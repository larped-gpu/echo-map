"""EchoMap companion app: live map viewer + robot status."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from config import CHIRP_MODES, MATERIALS, MODEL_DIR  # noqa: E402

st.set_page_config(page_title="EchoMap", page_icon="🦇", layout="wide")

st.title("EchoMap")
st.caption("Adaptive chirp echolocation room-mapping robot")

st.markdown(
    """
    A wheeled robot that maps indoor spaces and identifies wall materials using
    only sound. The **adaptive chirp policy** selects both *where to move* and
    *what chirp to emit* based on map uncertainty.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Chirp modes")
    for mode in CHIRP_MODES:
        if mode == "GEOMETRY":
            st.write(f"• **{mode}**: long-range wall detection (2-8 kHz)")
        elif mode == "MATERIAL":
            st.write(f"• **{mode}**: surface discrimination (8-20 kHz)")
        else:
            st.write(f"• **{mode}**: glass/mirror probe (12-24 kHz, oblique)")

with col2:
    st.subheader("Status")
    model_path = MODEL_DIR / "echomap.pt"
    if model_path.exists():
        st.success("EchoNet model found")
    else:
        st.warning("Train first: `python python/train.py --synthetic`")

    map_path = ROOT / "data" / "map_output.png"
    if map_path.exists():
        st.success("Map output available")
        st.image(str(map_path), caption="Latest room map")
    else:
        st.info("Run mapping: `python python/inference.py --synthetic`")

st.divider()

st.subheader("Materials")
mat_cols = st.columns(3)
for i, mat in enumerate(MATERIALS):
    with mat_cols[i % 3]:
        st.write(f"• {mat}")

demo_mode = st.selectbox("Demo chirp mode", CHIRP_MODES)

if st.button("Simulate chirp emission"):
    st.markdown(f"### Emitting `{demo_mode}` chirp...")
    st.progress(0.0)
    import time

    for pct in range(0, 101, 10):
        st.progress(pct / 100.0)
        time.sleep(0.05)
    st.success(f"{demo_mode} chirp complete, echo processed")

st.divider()
st.markdown(
    "**Hardware:** Pi 4 · 4× electret mics · USB audio · SG90 servo · 2WD chassis  \n"
    "**Pipeline:** chirp → matched filter → TOA/DOA → MFCC → EchoNet → map update → policy"
)
