import streamlit as st

st.set_page_config(
    page_title="Wildfire Smoke Detection Demo",
    layout="wide",
)

from streamlit_app.app import main

if __name__ == "__main__":
    main()
