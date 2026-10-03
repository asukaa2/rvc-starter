"""Status / progress display components.

A simple ``gr.Textbox`` keeps the implementation dependency-free; we just
standardise on a non-interactive multi-line textbox with a "Ready." default.
"""

import gradio as gr


def status_box(label="Status", value="Ready.", lines=2):
    """Non-interactive status textbox.

    Use as the output target of the convert / download buttons so the user
    sees "[OK] ..." or "[ERROR] ..." messages from the pipeline.
    """
    return gr.Textbox(value=value, label=label, interactive=False, lines=lines)


def output_path_box(label="Output file", value=""):
    """Small read-only textbox for the written-file path."""
    return gr.Textbox(value=value, label=label, interactive=False, max_lines=1)
