"""Pitch-extraction (f0) controls: method dropdown, pitch slider, autotune.

Every f0 method that ``rvc.lib.predictors.generator.Generator.compute_f0``
knows about is mirrored here so the UI never offers a method the backend
can't actually run.
"""

import gradio as gr

# Keep in sync with Generator.compute_f0's dispatch dict.
F0_METHODS = [
    "pm",
    "dio", "harvest",
    "yin", "pyin",
    "swipe",
    "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large", "crepe-full",
    "mangio-crepe-tiny", "mangio-crepe-small", "mangio-crepe-medium",
    "mangio-crepe-large", "mangio-crepe-full",
    "rmvpe", "rmvpe-legacy",
    "fcpe", "fcpe-legacy",
]


def f0_method_dropdown(label="F0 method", value="rmvpe"):
    return gr.Dropdown(choices=F0_METHODS, value=value, label=label, interactive=True)


def pitch_slider(label="Pitch shift (semitones)", value=0):
    """+12 raises by an octave, -12 drops by an octave."""
    return gr.Slider(minimum=-36, maximum=36, step=1, value=value, label=label)


def filter_radius_slider(label="F0 median filter radius", value=3):
    """Odd integer; 0 disables the median filter.  Higher = smoother f0."""
    return gr.Slider(minimum=0, maximum=7, step=1, value=value, label=label)


def hop_length_slider(label="Hop length (samples)", value=64):
    return gr.Slider(minimum=16, maximum=512, step=16, value=value, label=label)


def autotune_checkbox(label="Snap f0 to nearest note", value=False):
    return gr.Checkbox(value=value, label=label)


def autotune_strength_slider(label="Autotune strength", value=1.0):
    return gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=value, label=label)
