#!/usr/bin/env bash
set -e

BR_MAIN="main"
BR_HF="hf-clean"

git checkout "$BR_HF"
git rm -r --cached . >/dev/null 2>&1 || true

git checkout "$BR_MAIN" -- \
  README.md requirements.txt .gitignore \
  streamlit_app/app.py streamlit_app/src streamlit_app/utils

cat > app.py <<'PY'
import streamlit as st

st.set_page_config(
    page_title="Wildfire Smoke Detection Demo",
    layout="wide",
)

from streamlit_app.app import main

if __name__ == "__main__":
    main()
PY

git add -A
git commit -m "Deploy to Hugging Face" || echo "No changes to deploy."
git push hf "$BR_HF":main --force

git checkout "$BR_MAIN"
echo "[OK] Deployed to Hugging Face."
