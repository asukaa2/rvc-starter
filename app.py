"""Gradio web UI entry point for the rvc-starter project.

The app is structured as:

    app.py                 <- this file; mounts every tab and launches
    tabs/<name>.py         <- each file exposes ``nametabs() -> (name, build)``
    child/<component>.py  <- reusable Gradio UI components

To add a new tab:

1. Drop ``tabs/my_tab.py`` with:
       def nametabs():
           return "My tab", build_my_tab
       def build_my_tab():
           ...
2. Add ``from tabs.my_tab import nametabs as my_tab_nametabs`` to the import
   block below and append ``my_tab_nametabs`` to the ``TABS`` list.

Nothing else needs to change — ``build_app()`` iterates over ``TABS`` and
mounts every entry inside a single ``gr.Tabs`` block.
"""

import os
import sys

# Make the rvc package importable when running app.py directly via
# ``python app.py`` (no install needed) — the project root is the parent
# of this file.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Ensure the local ``models/`` cache dir exists so the download tab can
# write into it on first run.
os.makedirs(os.path.join(_REPO_ROOT, "models"), exist_ok=True)

import gradio as gr  # noqa: E402  (after sys.path patching)

# Each entry here is the ``nametabs`` function from a tabs/*.py module.
# A function returning (name, builder) means app.py never has to know the
# concrete tab class — adding / removing tabs is a one-line change here.
from tabs.inference_tab import nametabs as inference_tab_nametabs
from tabs.download_models import nametabs as download_tab_nametabs

TABS = [
    inference_tab_nametabs,
    download_tab_nametabs,
]


def build_app() -> gr.Blocks:
    """Construct the full Blocks app, mounting every tab in ``TABS``."""
    # NOTE: theme= and css= were moved from Blocks() to launch() in Gradio 6.
    with gr.Blocks(title="RVC Starter") as app:
        gr.Markdown(
            "# RVC Starter\n"
            "Retrieval-based Voice Conversion — point-and-click inference."
        )

        with gr.Tabs():
            for nametabs in TABS:
                name, builder = nametabs()
                with gr.Tab(name):
                    builder()

        gr.Markdown(
            "---\n"
            "RVC Starter · [github.com/asukaa2/rvc-starter]"
            "(https://github.com/asukaa2/rvc-starter)"
        )

    return app


def main():
    """Entry point used by the ``rvc-webui`` console script."""
    host = os.environ.get("RVC_HOST", "127.0.0.1")
    port = int(os.environ.get("RVC_PORT", "7860"))
    share = os.environ.get("RVC_SHARE", "0") == "1"

    app = build_app()
    # Gradio 6 moved theme / css here from gr.Blocks().
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=os.environ.get("RVC_OPEN_BROWSER", "0") == "1",
        show_error=True,
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1200px !important; margin: 0 auto; }",
    )


if __name__ == "__main__":
    main()
