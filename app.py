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
from bokeh.models import Div
from bokeh.plotting import show

# --- Configuration ---
DEFAULT_BACKGROUND_URL = "https://whitescreen.online/image/green-background.png"
OUTPUT_PATH = 'output.mp4'
MASK_THRESHOLD = 0.6
OUTPUT_FPS = 30

# --- Function Definitions ---
# ... (Function definitions for convert_to_png, download_image, and process_video remain the same) ...


# --- Streamlit App ---

# Initialize session state for theme
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Define custom CSS for light and dark themes
light_theme = """
    <style>
    body {
        background-color: #f0f2f6;
        color: #262730;
    }
    .stButton>button {
        background-color: #f0f2f6;
        color: #262730;
        border: 1px solid #262730;
    }
    </style>
"""

dark_theme = """
    <style>
    body {
        background-color: #262730;
        color: #f0f2f6;
    }
    .stButton>button {
        background-color: #262730;
        color: #f0f2f6;
        border: 1px solid #f0f2f6;
    }
    </style>
"""

# Apply the selected theme
if st.session_state.theme == "light":
    st.markdown(light_theme, unsafe_allow_html=True)
else:
    st.markdown(dark_theme, unsafe_allow_html=True)

# Theme toggle button
if st.button("Toggle Theme"):
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.experimental_rerun()  # Rerun the app to apply the theme change


# Hide UI elements except Settings
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .css-eczf16 {visibility:hidden;}
            .stActionButton {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Background Remover")

# Explore button with JavaScript redirect
if st.button("Explore"):
    js = f"window.open('https://hrsproject.github.io/home/')"
    html = f'<img src onerror="{js}">'
    div = Div(text=html)
    show(div)

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
