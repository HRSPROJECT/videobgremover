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

# --- Configuration ---
DEFAULT_BACKGROUND_URL = "https://i.imgur.com/1q56L1X.png"  # Public green screen image
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
    start_time = time.time()
    
    progress_bar = st.progress(0)  # Add a progress bar

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

            resized_background = cv2.resize(np.array(background_image), (width, height))

            output = np.where(condition, frame, resized_background)

            out.write(output)
            frame_count += 1

            progress = int(frame_count / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100)
            progress_bar.update(progress)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        return None

    finally:
        cap.release()
        out.release()
        # Remove the temporary file
        os.unlink(tfile.name) 

    end_time = time.time()
    total_time = end_time - start_time
    st.info(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
    if frame_count > 0:
        st.info(f"Average FPS: {frame_count / total_time:.2f} FPS")
    st.success(f"Video processing complete.")
    return OUTPUT_PATH

# --- Streamlit App ---

st.title("Background Remover")

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
            output_path = process_video(video_bytes, background_image, MASK_THRESHOLD)
            if output_path:
                st.video(output_path)
