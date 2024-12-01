import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Webrtc")
webrtc_streamer(key="Example")