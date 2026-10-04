"""Reusable audio input / output Gradio components.

Each function here returns a single ``gr.Audio`` / ``gr.File`` component so
the tab builder can compose them freely.  Keeping them in a child module
means the inference tab (or any future tab) doesn't have to re-derive the
``sources=`` / ``type=`` choices — change once here, applies everywhere.
"""

import gradio as gr


def audio_in(label="Input audio", sources=("upload", "microphone")):
    """Audio input component that returns a filepath (not numpy)."""
    return gr.Audio(
        sources=list(sources),
        type="filepath",
        label=label,
        editable=False,
    )


def audio_out(label="Converted audio"):
    """Audio output component.  Use ``.click(outputs=[audio_out, ...])``."""
    return gr.Audio(type="filepath", label=label, interactive=False)


def batch_dir_in(label="…or batch-convert a directory"):
    """Optional directory input for batch mode (one file per audio inside)."""
    return gr.File(
        file_count="directory",
        file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus",
                    ".m4a", ".aac", ".aiff", ".webm"],
        label=label,
    )
