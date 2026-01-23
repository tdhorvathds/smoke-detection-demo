from __future__ import annotations

from pathlib import Path
import requests
import streamlit as st

# =========================
# EDIT THESE
# =========================
# Put your weights into a GitHub Release and paste the direct asset URLs here.
WEIGHTS = {
    "fasterrcnn_best.pth": "https://github.com/tdhorvathds/smoke-detection-demo/releases/download/v1.0-demo/fasterrcnn_best.pth",
    "yolo_best.pt": "https://github.com/tdhorvathds/smoke-detection-demo/releases/download/v1.0-demo/yolo_best.pt",
}


# This matches your current structure: streamlit_app/models/<file>
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# =========================


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        prog = st.progress(0.0, text=f"Downloading {dst.name}…")
        done = 0

        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total > 0:
                    prog.progress(min(done / total, 1.0))

        prog.empty()


@st.cache_resource(show_spinner=False)
def ensure_weights() -> dict[str, Path]:
    """
    Ensures weights exist in streamlit_app/models/.
    Downloads missing files from GitHub Releases once per app instance.
    Returns {filename: local_path}.
    """
    out: dict[str, Path] = {}
    for fname, url in WEIGHTS.items():
        dst = MODELS_DIR / fname
        if not dst.exists():
            _download(url, dst)
        out[fname] = dst
    return out
