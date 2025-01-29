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
DEFAULT_BACKGROUND_URL = "https://whitescreen.online/image/green-background.png"
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
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading default background image: {e}")
        return None

def process_video(video_bytes, background_image, mask_threshold):
    """Processes a video for background replacement using MediaPipe."""
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
    progress_bar = st.progress(0)
    start_time = time.time()

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

            progress = int(frame_count / total_frames * 100)
            progress_bar.progress(progress)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        return None, None

    finally:
        cap.release()
        out.release()
        os.unlink(tfile.name)

    end_time = time.time()
    total_time = end_time - start_time
    st.info(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
    st.info(f"Average FPS: {frame_count / total_time:.2f} FPS")
    st.success(f"Video processing complete.")

    with open(OUTPUT_PATH, 'rb') as f:
        video_data = f.read()

    return video_data, OUTPUT_PATH

# --- Streamlit App ---

st.title("Background Remover")

# Add the redirect button
if st.button("Explore"):
    st.markdown(f'<a href="https://hrsproject.github.io/home/" target="_blank">Explore</a>', unsafe_allow_html=True)

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
