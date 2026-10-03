# rvc-starter

A clean, self-contained starter pipeline for **Retrieval-based Voice Conversion
(RVC)** inference.  Given a trained `.pth` model (and optionally a `.index`
faiss file), this repo lets you convert an audio file or a directory of audio
files in a single command — no training, no GUI, no extra fluff.

This is the cleaned-up version of the project: all missing modules have been
filled in, every broken import has been repaired, and several real bugs
(a nested `forward` method, a hardcoded channel count, a stale `new_tensor`
call) have been fixed.

---

## Features

- **Single-file & batch inference** — pass either an audio file or a directory.
- **6+ pitch extractors** — `pm`, `dio`, `harvest`, `yin`, `pyin`, `swipe`,
  `crepe-{tiny,small,medium,large,full}`, `mangio-crepe-*`, `rmvpe`,
  `rmvpe-legacy`, `fcpe`, `fcpe-legacy`.
- **Multiple vocoders** — Default NSF-HiFiGAN, MRF-HiFiGAN, RefineGAN.
- **CPU / CUDA / MPS / OpenCL** — automatic backend selection.
- **Audio splitting** — splits long inputs at silent gaps before conversion
  so memory stays bounded.
- **Optional denoising** — built-in spectral-gating noise reducer applied
  after conversion.
- **Optional pitch autotune** — snaps f0 to the nearest equal-tempered note.

---

## Installation

```bash
git clone https://github.com/asukaa2/rvc-starter.git
cd rvc-starter
pip install -r requirements.txt
# Optionally install as a package so the `rvc-infer` CLI is on PATH:
pip install -e .
```

> **PyTorch note:** install the wheel that matches your CUDA version from
> <https://pytorch.org/get-started/locally/> first if you want GPU inference.
> The `requirements.txt` only lists `torch>=2.1.0` — the actual CUDA build
> comes from whichever wheel you install.

### AMD GPU support (OpenCL)

Install [`pytorch-ocl`](https://github.com/cupy/pytorch-ocl) and the rest of
the stack will pick it up automatically.

```bash
pip install -e ".[opencl]"
```

---

## Model files

RVC needs pretrained embedder and pitch-extractor weights, and your own
trained `.pth` voice model (+ optional `.index`).  By default the pipeline
expects these under a `./models/` directory next to where you run inference:

```
models/
├── contentvec_base.pt     # Hubert embedder (auto-downloaded on first run)
├── rmvpe.pt                # RMVPE pitch extractor (auto-downloaded)
├── fcpe.pt                 # FCPE pitch extractor (auto-downloaded)
├── crepe_*.pth             # CREPE pitch extractors (auto-downloaded)
├── my_voice.pth            # YOUR trained voice model
└── my_voice.index          # YOUR retrieval index (optional)
```

Embedder and predictor weights are auto-downloaded from the project's
HuggingFace mirror the first time you reference them, so you only need to
bring your own `.pth` / `.index`.

---

## Usage

### As a CLI (recommended)

```bash
# Single-file conversion
rvc-infer \
    -i input.wav \
    -o output.wav \
    -m models/my_voice.pth \
    --index models/my_voice.index \
    --pitch 12 \
    --f0-method rmvpe \
    --index-rate 0.75

# Batch conversion (whole directory of audio files)
rvc-infer \
    -i ./inputs/ \
    -o ./outputs/ \
    -m models/my_voice.pth \
    --split-audio --clean-audio --half
```

Run `rvc-infer --help` to see every flag.

### As a Python module

```python
from rvc.infer.infer import infer_main

infer_main(
    input_path="input.wav",
    output_path="output.wav",
    pth_path="models/my_voice.pth",
    index_path="models/my_voice.index",
    pitch=12,
    f0_method="rmvpe",
    index_rate=0.75,
    is_half=True,
)
```

---

## Project layout

```
rvc/
├── config.py                # runtime device / fp16 configuration
├── cli.py                   # argparse CLI entry point (rvc-infer)
├── opencl.py                # AMD GPU OpenCL helpers (STFT / GRU / group_norm)
├── infer/
│   ├── infer.py             # top-level infer_main() + VoiceConverter class
│   ├── pipeline.py          # per-chunk voice-conversion pipeline
│   └── utils.py             # load_audio / clear_gpu_cache / HF_download_file ...
├── lib/
│   ├── algo/                # synthesizer + encoder + generator + residual blocks
│   │   ├── synthesizers.py
│   │   ├── encoders.py
│   │   ├── attentions.py
│   │   ├── modules.py       # WaveNet
│   │   ├── residuals.py     # ResidualCouplingBlock + ResBlock
│   │   ├── commons.py
│   │   ├── normalization.py
│   │   └── generators/
│   │       ├── hifigan.py
│   │       ├── nsf_hifigan.py
│   │       ├── mrf_hifigan.py
│   │       └── refinegan.py
│   ├── predictors/          # pitch extractors + the unified Generator
│   │   ├── generator.py
│   │   ├── rmvpe.py
│   │   ├── fcpe.py
│   │   ├── crepe.py
│   │   ├── swipe.py
│   │   └── pyworld.py
│   └── tools/               # mediafire / mega / pixeldrain / gdrive downloaders
└── modules/
    ├── fairseq.py           # in-repo Hubert implementation (no fairseq dep)
    ├── opencl.py
    ├── cut.py               # split-audio / restore utilities
    ├── noisereduce.py       # built-in spectral-gating noise reducer
    └── utils/
        └── rms.py           # RMS energy extractor
```

---

## What was fixed

The original repo was a work-in-progress.  This fork fills in every missing
piece and repairs every runtime error so the package imports cleanly and the
CLI works end-to-end.  Specifically:

| # | File | Problem | Fix |
|---|------|---------|-----|
| 1 | all packages | missing `__init__.py` files | created them |
| 2 | `rvc/config.py` | imported by `infer.py` but didn't exist | implemented `Config` with device auto-detection + `device_config()` |
| 3 | `rvc/modules/cut.py` | imported by `infer.py` but didn't exist | implemented `cut()` / `restore()` using `librosa.effects.split` + chunk-aware zero-padding |
| 4 | `rvc/modules/noisereduce.py` | imported (as `modules.noisereduce`) by `infer.py` but didn't exist | implemented a self-contained spectral-gating `reduce_noise()` |
| 5 | `rvc/lib/algo/generators/mrf_hifigan.py` | imported by `synthesizers.py` but didn't exist | implemented a working `HiFiGANMRFGenerator` mirroring the NSF variant |
| 6 | `rvc/infer/pipeline.py` | three broken imports (`predictor` vs `predictors`, `lib.modules.utils.rms` vs `modules.utils.rms`, deleted `lib.modules.my_utils`) | rewired to the correct paths |
| 7 | `rvc/infer/infer.py` | `from modules.noisereduce ...` (bare `modules`) | fixed to `from rvc.modules.noisereduce ...` |
| 8 | `rvc/lib/algo/generators/hifigan.py` | (a) wrong import paths, (b) `forward()` was nested inside `__init__` | fixed imports; un-nested `forward` to be a class method |
| 9 | `rvc/lib/algo/generators/nsf_hifigan.py` | wrong import paths | rewired to `rvc.lib.algo.commons` / `residuals` |
| 10 | `rvc/lib/algo/generators/refinegan.py` | (a) hardcoded `nn.Conv1d(256, …)` for the conditioning input, (b) `block.remove_weight_norm()` called on plain `Conv1d` | use `gin_channels`; use the function-form `remove_weight_norm(block)` |
| 11 | `rvc/lib/predictors/crepe.py` | `cents.new_tensor(numpy_array)` — `new_tensor` rejects numpy arrays in modern PyTorch | convert numpy → torch tensor explicitly before adding |
| 12 | repo root | no `requirements.txt`, no `README`, no `pyproject.toml`, no CLI | added all four |

After the fixes, the whole package imports cleanly:

```bash
python -c "from rvc.infer.infer import infer_main; print('ok')"
# ok
```

---

## License

MIT — see [LICENSE](./LICENSE).
