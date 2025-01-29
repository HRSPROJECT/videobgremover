import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from PIL import Image
import requests
import tempfile
from io import BytesIO
from bokeh.models import Div  # Import for Explore button

# --- Configuration ---
DEFAULT_BACKGROUND_URL = "https://i.imgur.com/1q56L1X.png"
OUTPUT_PATH = 'output.mp4'
MASK_THRESHOLD = 0.6
OUTPUT_FPS = 30

# --- Function Definitions ---
# ... (Function definitions for convert_to_png, download_image, and process_video remain the same) ...

# --- Streamlit App ---

# Hide UI elements except Settings
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .css-eczf16 {visibility:hidden;}  
            .stActionButton {visibility: hidden;}
            .stApp {background-color: white !important; color: black !important;} 
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Background Remover")

# Explore button with JavaScript redirect
if st.button("Explore"):
    js = f"window.open('https://hrsproject.github.io/home/')"
    html = f'<img src onerror="{js}">'
    div = Div(text=html)
    st.bokeh_chart(div)  

# File uploaders and background selection
uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_video is not None:
    video_bytes = uploaded_video.read()

    background_option = st.radio("Background Option:", ("Default", "Upload"))
    if background_option == "Default":
        background_image = download_image(DEFAULT_BACKGROUND_URL)
    else:
        uploaded_background = st.file_uploader("Upload a background image", type=["png", "jpg", "jpeg"])
        if uploaded_background is not None:
            background_image = Image.open(uploaded_background)
        else:
            background_image = None

    # Video processing and display
    if background_image is not None:
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                video_data, output_path = process_video(video_bytes, background_image, MASK_THRESHOLD)
                if video_data:
                    st.video(video_data)
                    st.download_button(
                        label="Download Processed Video",
                        data=video_data,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )
