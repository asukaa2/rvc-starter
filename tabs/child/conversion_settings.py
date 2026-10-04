"""General conversion settings: retrieval mix, protection, embedder, export.

These are the parameters that don't fit into "pitch" or "advanced toggles"
— index rate, RMS blend, protect ratio, Hubert embedder choice, output
sample-rate override, and export container.
"""

import gradio as gr

# Same set as rvc.infer.utils.check_embedders.
EMBEDDERS = [
    "contentvec_base",
    "hubert_base",
    "japanese_hubert_base",
    "korean_hubert_base",
    "chinese_hubert_base",
    "portuguese_hubert_base",
    "spin",
]

EXPORT_FORMATS = ["wav", "flac", "mp3", "ogg", "opus", "m4a", "aac", "aiff", "webm"]
RESAMPLE_RATES = [0, 16000, 22050, 32000, 40000, 44100, 48000]


def index_rate_slider(label="Index rate (retrieval mix)", value=0.5):
    """0 = pure Hubert features, 1 = pure retrieval index."""
    return gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=value, label=label)


def volume_envelope_slider(label="Volume envelope (RMS blend)", value=1.0):
    """0 = pure target RMS, 1 = pure source RMS."""
    return gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=value, label=label)


def protect_slider(label="Protect (low-freq consonant guard)", value=0.5):
    """0 = no protection (more conversion), 0.5 = max protection."""
    return gr.Slider(minimum=0.0, maximum=0.5, step=0.05, value=value, label=label)


def embedder_dropdown(label="Hubert embedder", value="contentvec_base"):
    return gr.Dropdown(choices=EMBEDDERS, value=value, label=label, interactive=True)


def resample_sr_dropdown(label="Resample output SR (0 = keep model SR)", value=0):
    return gr.Dropdown(
        choices=[str(r) for r in RESAMPLE_RATES],
        value="0",
        label=label,
        interactive=True,
    )


def export_format_dropdown(label="Export format", value="wav"):
    return gr.Dropdown(choices=EXPORT_FORMATS, value=value, label=label, interactive=True)
