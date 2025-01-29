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

# --- Configuration ---
DEFAULT_BACKGROUND_URL = "https://whitescreen.online/image/green-background.png"  # Public green screen image
OUTPUT_PATH = 'output.mp4'
MASK_THRESHOLD = 0.6
OUTPUT_FPS = 30

# --- Function Definitions ---

def convert_to_png(image_path, output_dir="."):
    """Converts an image to PNG and returns the new path."""
    try:
        img = Image.open(image_path)
        name, ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(output_dir, f"{name}.png")
        img.save(output_path, "PNG")
        return output_path
    except Exception as e:
        st.error(f"Error converting image: {e}")
        return None

def download_image(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        return Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading default background image: {e}")
        return None

def process_video(video_bytes, background_image, mask_threshold):
    """Processes a video for background replacement using MediaPipe."""
    # Create a temporary file to store the video bytes
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_bytes)
    tfile.close()

    segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    cap = cv2.VideoCapture(tfile.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)  # Initialize progress bar
    start_time = time.time()

    # Convert the background to RGB using cv2
    background_rgb = cv2.cvtColor(np.array(background_image), cv2.COLOR_RGBA2RGB)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = segmentation.process(RGB)
            mask = results.segmentation_mask

            rsm = np.stack((mask,) * 3, axis=-1)
            condition = (rsm > mask_threshold).astype(np.uint8)

            resized_background = cv2.resize(background_rgb, (width, height))

            output = np.where(condition, frame, resized_background)

            out.write(output)
            frame_count += 1

            # Update progress bar
            progress = int(frame_count / total_frames * 100)
            progress_bar.progress(progress)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        return None, None

    finally:
        cap.release()
        out.release()
        os.unlink(tfile.name)  # Remove temporary file

    end_time = time.time()
    total_time = end_time - start_time
    st.info(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
    st.info(f"Average FPS: {frame_count / total_time:.2f} FPS")
    st.success(f"Video processing complete.")

    # Read the processed video file into memory for display with st.video
    with open(OUTPUT_PATH, 'rb') as f:
        video_data = f.read()

    return video_data, OUTPUT_PATH  # Return video data and file path

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

# ... (File uploaders, background selection, and processing remain the same) ...
