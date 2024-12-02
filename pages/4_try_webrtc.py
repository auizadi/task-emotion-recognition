import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Streamlit WebRTC - Uji Kamera")
webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTCConfiguration({  
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
)
