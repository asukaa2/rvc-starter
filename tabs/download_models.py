"""Download-models tab for the Gradio web UI.

Lets the user pull pretrained Hubert embedders and pitch extractors from the
project's HuggingFace mirror on demand, instead of waiting for the
inference tab to fetch them on first use.
"""

import os
import sys
import traceback

# Make the rvc package importable when the tab is loaded from app.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import gradio as gr

from child.model_picker import MODELS_DIR, scan_models_dir
from child.status import status_box

# Lazily imported so the tab loads without torch / faiss / etc being
# installed — the bridge functions will raise a friendly error if the
# package isn't fully wired yet.
def _import_downloaders():
    from rvc.infer.utils import check_predictors, check_embedders
    return check_predictors, check_embedders


EMBEDDERS = [
    "contentvec_base",
    "hubert_base",
    "japanese_hubert_base",
    "korean_hubert_base",
    "chinese_hubert_base",
    "portuguese_hubert_base",
    "spin",
]

PREDICTORS = [
    "rmvpe", "rmvpe-legacy",
    "fcpe", "fcpe-legacy",
    "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full",
    "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium",
    "mangio-crepe-large", "mangio-crepe-full",
]


def nametabs():
    """Return ``("Download models", build_download_tab)`` for ``app.py``."""
    return "Download models", build_download_tab


def build_download_tab():
    """Mount the download-models tab UI inside the current ``gr.Tab`` context."""
    gr.Markdown(
        "## Download pretrained embedders & pitch extractors\n"
        "Files are pulled from the project's HuggingFace mirror and cached "
        f"under `{MODELS_DIR}/`. Use this tab once before running your first "
        "inference to avoid the long first-run download."
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Hubert embedders", open=True):
                embedder_choice = gr.Dropdown(
                    choices=EMBEDDERS, value=EMBEDDERS[0],
                    label="Embedder", interactive=True,
                )
                dl_embedder_btn = gr.Button("Download embedder", variant="primary")
                embedder_status = status_box(label="Embedder status", value="Idle.")

        with gr.Column(scale=1):
            with gr.Accordion("Pitch extractors", open=True):
                predictor_choice = gr.Dropdown(
                    choices=PREDICTORS, value=PREDICTORS[0],
                    label="Predictor", interactive=True,
                )
                dl_predictor_btn = gr.Button("Download predictor", variant="primary")
                predictor_status = status_box(label="Predictor status", value="Idle.")

    # ----- Currently-cached files panel -----
    with gr.Accordion("Currently cached in models/", open=False):
        cache_table = gr.Dataframe(
            headers=["file", "size_MB"],
            datatype=["str", "number"],
            value=_refresh_cache_table(),
            interactive=False,
            wrap=True,
        )
        refresh_cache_btn = gr.Button("Refresh cache listing", variant="secondary")
        refresh_cache_btn.click(
            fn=_refresh_cache_table,
            outputs=[cache_table],
        )

    # ----- Wiring -----
    dl_embedder_btn.click(
        fn=_download_embedder,
        inputs=[embedder_choice],
        outputs=[embedder_status],
    ).then(
        fn=_refresh_cache_table,
        outputs=[cache_table],
    )

    dl_predictor_btn.click(
        fn=_download_predictor,
        inputs=[predictor_choice],
        outputs=[predictor_status],
    ).then(
        fn=_refresh_cache_table,
        outputs=[cache_table],
    )


def _download_embedder(name):
    """Download a Hubert embedder weight file via the project's HF mirror."""
    if not name:
        return "[ERROR] No embedder selected."
    try:
        _, check_embedders = _import_downloaders()
        check_embedders(name)
        path = os.path.join(MODELS_DIR, f"{name}.pt")
        return f"[OK] Embedder downloaded: {path}"
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}"


def _download_predictor(name):
    """Download a pitch-extractor weight file via the project's HF mirror."""
    if not name:
        return "[ERROR] No predictor selected."
    try:
        check_predictors, _ = _import_downloaders()
        check_predictors(name)
        return f"[OK] Predictor '{name}' downloaded to {MODELS_DIR}/"
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}"


def _refresh_cache_table():
    """Return a list of (filename, size_MB) tuples for everything under models/."""
    rows = []
    if not os.path.isdir(MODELS_DIR):
        return rows

    for root, _dirs, files in os.walk(MODELS_DIR):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            try:
                size_mb = round(os.path.getsize(full) / (1024 * 1024), 2)
            except OSError:
                size_mb = 0.0
            rows.append([os.path.relpath(full, MODELS_DIR), size_mb])

    # Sort by filename for a stable view.
    rows.sort(key=lambda r: r[0].lower())
    return rows
