from __future__ import annotations
from pathlib import Path
import zipfile
import requests
import streamlit as st

ASSETS_BASE = Path(__file__).resolve().parent.parent / "assets"
SAMPLES_DIR = ASSETS_BASE / "samples_demo"
XAI_DIR = ASSETS_BASE / "xai_demo"
GT_DIR = ASSETS_BASE / "gt"

# Point these to your HF dataset repo (replace USER/REPO)
HF_DATASET = "tdhorvath/smoke-detection-demo-assets"
SAMPLES_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/samples_demo.zip"
XAI_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/xai_demo.zip"
GT_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/gt.zip"


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise RuntimeError(f"Got HTML instead of file. URL may be wrong/private: {url}")

        total = int(r.headers.get("Content-Length") or 0)
        bar = st.progress(0.0, text=f"Downloading {out_path.name}…")
        done = 0

        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    bar.progress(min(done / total, 1.0))
        bar.empty()


def _unzip(zip_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst_dir.parent)  # zip contains folder name (samples_demo/ etc.)


@st.cache_resource(show_spinner=False)
def ensure_demo_assets() -> None:
    # Samples
    if not SAMPLES_DIR.exists():
        z = ASSETS_BASE / "samples_demo.zip"
        _download(SAMPLES_URL, z)
        _unzip(z, SAMPLES_DIR)

    # XAI
    if not XAI_DIR.exists():
        z = ASSETS_BASE / "xai_demo.zip"
        _download(XAI_URL, z)
        _unzip(z, XAI_DIR)

    # GT (optional)
    if not GT_DIR.exists():
        z = ASSETS_BASE / "gt.zip"
        try:
            _download(GT_URL, z)
            _unzip(z, GT_DIR)
        except Exception:
            # GT is optional in your UI; ignore if missing
            pass
