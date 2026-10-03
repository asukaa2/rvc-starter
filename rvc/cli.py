"""Command-line entry point for the RVC inference pipeline.

Run ``python -m rvc.cli --help`` (or ``rvc-infer`` after installing the
package) to see the full list of options.  The CLI is a thin wrapper around
:func:`rvc.infer.infer.infer_main` — every argument here maps directly to
a parameter of that function.
"""

import argparse
import sys

from rvc.infer.infer import infer_main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rvc-infer",
        description="Run a single RVC inference pass on an audio file or directory.",
    )

    # Required inputs.
    p.add_argument("--input", "-i", dest="input_path", required=True,
                   help="Path to a single audio file or a directory of audio files for batch conversion.")
    p.add_argument("--output", "-o", dest="output_path", default="./output.wav",
                   help="Output path (single-file mode) or output directory stem (batch mode).")
    p.add_argument("--model", "-m", dest="pth_path", required=True,
                   help="Path to the .pth voice-conversion model.")

    # Optional index file used for retrieval.
    p.add_argument("--index", dest="index_path", default=None,
                   help="Path to the .index faiss file (retrieval-based enhancement).")

    # Pitch / f0 options.
    p.add_argument("--pitch", type=int, default=0,
                   help="Pitch shift in semitones (e.g. 12 for +1 octave, -12 for -1 octave).")
    p.add_argument("--f0-method", dest="f0_method", default="rmvpe",
                   choices=["pm", "dio", "harvest", "yin", "pyin", "swipe",
                            "crepe-tiny", "crepe-small", "crepe-medium",
                            "crepe-large", "crepe-full",
                            "mangio-crepe-tiny", "mangio-crepe-small",
                            "mangio-crepe-medium", "mangio-crepe-large",
                            "mangio-crepe-full",
                            "rmvpe", "rmvpe-legacy", "fcpe", "fcpe-legacy"],
                   help="Pitch-extraction algorithm to use.")
    p.add_argument("--filter-radius", dest="filter_radius", type=int, default=3,
                   help="Median filter radius applied to f0 (odd integer; 0 disables).")
    p.add_argument("--hop-length", dest="hop_length", type=int, default=64,
                   help="Hop length for f0 estimation, in samples.")
    p.add_argument("--f0-autotune", dest="f0_autotune", action="store_true",
                   help="Snap f0 to the nearest musical note.")
    p.add_argument("--f0-autotune-strength", dest="f0_autotune_strength", type=float, default=1.0,
                   help="Strength of the autotune snap (0-1).")

    # Retrieval / protection options.
    p.add_argument("--index-rate", dest="index_rate", type=float, default=0.5,
                   help="Mix ratio between retrieval features and Hubert features (0-1).")
    p.add_argument("--volume-envelope", dest="volume_envelope", type=float, default=1.0,
                   help="Source-target RMS blending (0 = pure target, 1 = pure source).")
    p.add_argument("--protect", type=float, default=0.5,
                   help="Protect ratio for low-frequency breaths/consonants (0-1).")

    # Embedder / vocoder / export options.
    p.add_argument("--embedder", dest="embedder_model", default="contentvec_base",
                   choices=["contentvec_base", "hubert_base", "japanese_hubert_base",
                            "korean_hubert_base", "chinese_hubert_base",
                            "portuguese_hubert_base", "spin"],
                   help="Pretrained Hubert embedder to use.")
    p.add_argument("--resample-sr", dest="resample_sr", type=int, default=0,
                   help="Resample output to this sample rate (0 = keep model's target SR).")
    p.add_argument("--export-format", dest="export_format", default="wav",
                   choices=["wav", "flac", "mp3", "ogg", "opus", "m4a", "aac", "aiff", "webm"],
                   help="Output container format.")

    # Audio splitting / cleanup.
    p.add_argument("--split-audio", dest="split_audio", action="store_true",
                   help="Split long input at silent gaps before conversion.")
    p.add_argument("--clean-audio", dest="clean_audio", action="store_true",
                   help="Apply spectral-gating noise reduction to the output.")
    p.add_argument("--clean-strength", dest="clean_strength", type=float, default=0.7,
                   help="Strength (0-1) of the output denoiser when --clean-audio is set.")

    # Performance / device options.
    p.add_argument("--half", dest="is_half", action="store_true",
                   help="Use fp16 inference (CUDA / MPS only; auto-disabled on CPU/OpenCL).")
    p.add_argument("--cpu", dest="cpu_mode", action="store_true",
                   help="Force CPU inference (overrides --half).")

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    infer_main(
        is_half=args.is_half,
        cpu_mode=args.cpu_mode,
        pitch=args.pitch,
        filter_radius=args.filter_radius,
        index_rate=args.index_rate,
        volume_envelope=args.volume_envelope,
        protect=args.protect,
        hop_length=args.hop_length,
        f0_method=args.f0_method,
        input_path=args.input_path,
        output_path=args.output_path,
        pth_path=args.pth_path,
        index_path=args.index_path,
        export_format=args.export_format,
        embedder_model=args.embedder_model,
        resample_sr=args.resample_sr,
        f0_autotune=args.f0_autotune,
        f0_autotune_strength=args.f0_autotune_strength,
        split_audio=args.split_audio,
        clean_audio=args.clean_audio,
        clean_strength=args.clean_strength,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
