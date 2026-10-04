"""Advanced toggles: audio splitting, denoising, fp16, CPU mode.

These map 1:1 to the keyword arguments of
:func:`rvc.infer.infer.infer_main` that are typically left at their
defaults but are useful to expose in the UI.
"""

import gradio as gr


def split_audio_checkbox(label="Split long input at silences", value=False):
    """Splits input at silent gaps before conversion to keep memory bounded."""
    return gr.Checkbox(value=value, label=label)


def clean_audio_checkbox(label="Apply noise reduction to output", value=False):
    """Runs the built-in spectral-gating denoiser on the converted audio."""
    return gr.Checkbox(value=value, label=label)


def clean_strength_slider(label="Noise-reduction strength", value=0.7):
    """0 = no denoising, 1 = max denoising (can introduce artifacts)."""
    return gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=value, label=label)


def half_precision_checkbox(label="Use fp16 (CUDA / MPS only)", value=False):
    """Auto-disabled on CPU / OpenCL by Config.__init__."""
    return gr.Checkbox(value=value, label=label)


def cpu_mode_checkbox(label="Force CPU (overrides fp16)", value=False):
    """Skip GPU detection entirely — useful for debugging on a misbehaving GPU."""
    return gr.Checkbox(value=value, label=label)
