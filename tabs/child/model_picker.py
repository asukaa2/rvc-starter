"""Voice-model and feature-index dropdowns that scan ``./models/``.

The RVC pipeline reads the trained ``.pth`` and the optional ``.index``
files from a local ``models/`` directory at runtime.  These helpers scan
that directory and turn the contents into ``gr.Dropdown`` choices so the
UI is always in sync with whatever the user dropped on disk.
"""

import os

import gradio as gr

# Same convention as rvc/infer/utils.py — relative to wherever the python
# process was launched from, which is the project root when ``app.py`` runs.
MODELS_DIR = os.path.join(os.getcwd(), "models")


def scan_models_dir():
    """Walk ``./models/`` and return ``(pth_paths, index_paths)``.

    Returned paths are absolute so the user can copy-paste them elsewhere
    without thinking about the working directory.
    """
    pth_choices, index_choices = [], []

    if not os.path.isdir(MODELS_DIR):
        return pth_choices, index_choices

    for root, _dirs, files in os.walk(MODELS_DIR):
        for fn in sorted(files):
            full = os.path.abspath(os.path.join(root, fn))
            low = fn.lower()
            if low.endswith(".pth"):
                pth_choices.append(full)
            elif low.endswith(".index") or low.endswith(".bin"):
                # `.bin` shows up in some old RVC indices.
                index_choices.append(full)

    return pth_choices, index_choices


def model_picker(label="Voice model (.pth)"):
    """Dropdown listing every ``.pth`` under ``./models/``."""
    pth_choices, _ = scan_models_dir()
    return gr.Dropdown(
        choices=pth_choices,
        value=pth_choices[0] if pth_choices else None,
        label=label,
        interactive=True,
        allow_custom_value=True,  # let users paste a path outside models/
        info=f"Scanned from {MODELS_DIR}",
    )


def index_picker(label="Feature index (.index) — optional"):
    """Dropdown listing every ``.index`` / ``.bin`` under ``./models/``."""
    _, index_choices = scan_models_dir()
    return gr.Dropdown(
        choices=index_choices,
        value=None,
        label=label,
        interactive=True,
        allow_custom_value=True,
    )


def refresh_button(label="Re-scan models/"):
    """Small secondary button that callers wire to refresh the dropdowns."""
    return gr.Button(value=label, variant="secondary", size="sm")
