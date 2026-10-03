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

## Web UI (Gradio)

For point-and-click usage, the project ships a Gradio app under `app.py`.
The UI is split into reusable components (`child/`) and tab builders
(`tabs/`); each tab exposes a `nametabs()` function returning
`(tab_name, builder_fn)` so `app.py` can mount every tab with a single loop.

```bash
python app.py
# Launches http://127.0.0.1:7860
```

Layout:

```
app.py                 # entry: mounts every tab via nametabs() and launches
tabs/
├── inference_tab.py   # nametabs() -> ("Inference", build_inference_tab)
└── download_models.py  # nametabs() -> ("Download models", build_download_tab)
child/                  # reusable Gradio components
├── audio_io.py         # input / output audio components
├── model_picker.py     # .pth / .index dropdowns + scan_models_dir()
├── f0_controls.py      # f0_method, pitch, autotune sliders + dropdowns
├── conversion_settings.py  # index_rate, protect, embedder, export_format, ...
├── advanced_settings.py    # split_audio, clean_audio, half / cpu toggles
└── status.py          # status textbox + progress bar
```

Add a new tab by dropping a `tabs/<name>.py` that defines:

```python
def nametabs():
    return "Tab name", build_tab

def build_tab():
    ...
```

…and registering it in `app.py`'s `TABS` list.  No other wiring needed.

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

## License

MIT — see [LICENSE](./LICENSE).
