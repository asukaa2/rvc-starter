"""Inference tab for the Gradio web UI.

Exposes :func:`nametabs` which returns ``("Inference", build_inference_tab)``.
``app.py`` iterates over the registered tabs and mounts each one in a
single ``with gr.Tabs():`` block — see the README for the pattern.
"""

import os
import sys
import traceback

# Make sure the ``rvc`` package is importable when the tab is loaded
# (e.g. when running app.py from a different cwd than the repo root).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import gradio as gr

from tabs.child.audio_io import audio_in, audio_out
from tabs.child.model_picker import (
    index_picker,
    model_picker,
    refresh_button,
    scan_models_dir,
)
from tabs.child.f0_controls import (
    autotune_checkbox,
    autotune_strength_slider,
    f0_method_dropdown,
    filter_radius_slider,
    hop_length_slider,
    pitch_slider,
)
from tabs.child.conversion_settings import (
    embedder_dropdown,
    export_format_dropdown,
    index_rate_slider,
    protect_slider,
    resample_sr_dropdown,
    volume_envelope_slider,
)
from tabs.child.advanced_settings import (
    clean_audio_checkbox,
    clean_strength_slider,
    cpu_mode_checkbox,
    half_precision_checkbox,
    split_audio_checkbox,
)
from tabs.child.status import status_box

# Imported lazily inside the bridge function so the tab loads even before
# the rvc package's heavy deps (torch, faiss, ...) are installed; that
# keeps the Gradio UI previewable in lightweight CI environments.
def _import_infer_main():
    from rvc.infer.infer import infer_main
    return infer_main


def nametabs():
    """Return ``(tab_name, builder_fn)`` so ``app.py`` can mount us."""
    return "Inference", build_inference_tab


def build_inference_tab():
    """Mount the inference tab UI inside the current ``gr.Tab`` context."""
    gr.Markdown(
        "## Voice Conversion\n"
        "Upload an audio file, pick a trained `.pth` model, tweak f0 / retrieval "
        "settings, and click **Convert**."
    )

    # ----- Top row: input + output audio -----
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_audio = audio_in()
            with gr.Row():
                pth_picker = model_picker()
                idx_picker = index_picker()
            refresh_btn = refresh_button()
        with gr.Column(scale=1):
            output_audio = audio_out()

    # ----- F0 / pitch -----
    with gr.Accordion("F0 / pitch", open=True):
        with gr.Row():
            f0_method = f0_method_dropdown()
            pitch = pitch_slider()
        with gr.Row():
            filter_radius = filter_radius_slider()
            hop_length = hop_length_slider()
        with gr.Row():
            autotune = autotune_checkbox()
            autotune_strength = autotune_strength_slider()

    # ----- Retrieval / protection -----
    with gr.Accordion("Retrieval / protection", open=False):
        with gr.Row():
            index_rate = index_rate_slider()
            volume_envelope = volume_envelope_slider()
            protect = protect_slider()

    # ----- Embedder / export -----
    with gr.Accordion("Embedder / export", open=False):
        with gr.Row():
            embedder = embedder_dropdown()
            resample_sr = resample_sr_dropdown()
            export_format = export_format_dropdown()

    # ----- Advanced -----
    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            split_audio = split_audio_checkbox()
            clean_audio = clean_audio_checkbox()
            clean_strength = clean_strength_slider()
        with gr.Row():
            is_half = half_precision_checkbox()
            cpu_mode = cpu_mode_checkbox()

    status = status_box()

    with gr.Row():
        convert_btn = gr.Button("Convert", variant="primary")
        clear_btn = gr.Button("Clear", variant="stop")

    # ----- Wiring -----
    convert_inputs = [
        input_audio, pth_picker, idx_picker,
        f0_method, pitch, filter_radius, hop_length,
        autotune, autotune_strength,
        index_rate, volume_envelope, protect,
        embedder, resample_sr, export_format,
        split_audio, clean_audio, clean_strength,
        is_half, cpu_mode,
    ]
    convert_btn.click(
        fn=_run_inference,
        inputs=convert_inputs,
        outputs=[output_audio, status],
    )

    # Refresh button: re-scan ./models/ and push the new choices into both
    # dropdowns in a single click (one named function returning two
    # gr.Dropdown.update objects — cleaner than the click().then() chain
    # and avoids the Gradio 6 arg-count mismatch warning).
    refresh_btn.click(
        fn=_refresh_dropdowns,
        outputs=[pth_picker, idx_picker],
    )

    # Clear button: blank the output audio + reset status.
    clear_btn.click(
        fn=lambda: (None, "Cleared."),
        outputs=[output_audio, status],
    )


def _refresh_dropdowns():
    """Re-scan ``./models/`` and return updated Dropdown components.

    Called by the "Re-scan models/" button.  Returns a tuple of two
    ``gr.Dropdown(...)`` instances so Gradio pushes the new ``choices=``
    and ``value=`` into the model_picker / index_picker dropdowns in one
    round-trip.
    """
    pth_choices, index_choices = scan_models_dir()
    return (
        gr.Dropdown(choices=pth_choices, value=pth_choices[0] if pth_choices else None),
        gr.Dropdown(choices=index_choices, value=None),
    )


def _run_inference(
    input_path, pth_path, index_path,
    f0_method, pitch, filter_radius, hop_length,
    autotune, autotune_strength,
    index_rate, volume_envelope, protect,
    embedder, resample_sr, export_format,
    split_audio, clean_audio, clean_strength,
    is_half, cpu_mode,
):
    """Bridge: collect UI values -> call ``infer_main`` -> (audio_out, status)."""
    if not input_path:
        return None, "[ERROR] No input audio provided."
    if not pth_path or not os.path.isfile(pth_path):
        return None, "[ERROR] No voice model selected (or the path is invalid)."

    output_path = os.path.splitext(input_path)[0] + f"_rvc_output.{export_format}"

    try:
        infer_main = _import_infer_main()
        infer_main(
            is_half=bool(is_half),
            cpu_mode=bool(cpu_mode),
            pitch=int(pitch),
            filter_radius=int(filter_radius),
            index_rate=float(index_rate),
            volume_envelope=float(volume_envelope),
            protect=float(protect),
            hop_length=int(hop_length),
            f0_method=f0_method,
            input_path=input_path,
            output_path=output_path,
            pth_path=pth_path,
            index_path=index_path or None,
            export_format=export_format,
            embedder_model=embedder,
            resample_sr=int(resample_sr) if str(resample_sr) != "0" else 0,
            f0_autotune=bool(autotune),
            f0_autotune_strength=float(autotune_strength),
            split_audio=bool(split_audio),
            clean_audio=bool(clean_audio),
            clean_strength=float(clean_strength),
        )
        return output_path, f"[OK] Conversion complete. Output written to: {output_path}"
    except Exception as e:
        return None, f"[ERROR] {e}\n{traceback.format_exc()}"
