"""Gradio web UI entry point for the rvc-starter project.

Mounts every tab registered in ``TABS`` from ``tabs/*.py`` (each exposes
``nametabs() -> (tab_name, builder_fn)``).  Adding a new tab = drop a new
file in ``tabs/``, add one import + one entry to ``TABS``.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.makedirs(os.path.join(_REPO_ROOT, "models"), exist_ok=True)

import gradio as gr  # noqa: E402  (after sys.path patching)

from tabs.inference_tab import nametabs as inference_tab_nametabs  # noqa: E402
from tabs.download_models import nametabs as download_tab_nametabs  # noqa: E402

TABS = [
    inference_tab_nametabs,
    download_tab_nametabs,
]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RVC Starter") as app:
        with gr.Tabs():
            for nametabs in TABS:
                name, builder = nametabs()
                with gr.Tab(name):
                    builder()
    return app


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="rvc-webui",
        description="Launch the rvc-starter Gradio web UI.",
    )
    p.add_argument("--host", default=os.environ.get("RVC_HOST", "127.0.0.1"),
                   help="Bind address (default 127.0.0.1; use 0.0.0.0 to expose).")
    p.add_argument("--port", type=int, default=int(os.environ.get("RVC_PORT", "7860")),
                   help="Port (default 7860).")
    p.add_argument("--share", action="store_true",
                   default=os.environ.get("RVC_SHARE", "0") == "1",
                   help="Create a public gradio.live share URL.")
    p.add_argument("--browser", action="store_true",
                   default=os.environ.get("RVC_OPEN_BROWSER", "0") == "1",
                   help="Open the UI in the default browser on launch.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.browser,
        show_error=True,
    )


if __name__ == "__main__":
    main()
