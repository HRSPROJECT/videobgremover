import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from PIL import Image
import requests
from io import BytesIO

# --- Configuration ---
DEFAULT_BACKGROUND_URL = "https://i.imgur.com/1q56L1X.png" #Green screen image from a public source
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

def process_video(video_path, background_path, output_path, mask_threshold, output_fps=None):
    """Processes a video for background replacement using MediaPipe."""
    if not os.path.exists(video_path):
        st.error(f"Error: Video file not found at {video_path}")
        return None

    segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    background = cv2.imread(background_path)
    if background is None:
        st.error(f"Error: Could not read background image at {background_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file at {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if output_fps is None:
       output_fps = fps

    st.write(f"Input video: {width}x{height} @ {fps:.2f} FPS")
    st.write(f"Output video: {width}x{height} @ {output_fps:.2f} FPS")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), output_fps, (width, height))

    frame_count = 0
    start_time = time.time()

    try:
       with st.spinner("Processing video..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = segmentation.process(RGB)
            mask = results.segmentation_mask

            rsm = np.stack((mask,) * 3, axis=-1)
            condition = (rsm > mask_threshold).astype(np.uint8)

            resized_background = cv2.resize(background, (width, height))

            output = np.where(condition, frame, resized_background)

            out.write(output)
            frame_count += 1

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    end_time = time.time()
    total_time = end_time - start_time
    st.write(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
    if frame_count > 0:
       st.write(f"Average FPS: {frame_count/total_time:.2f} FPS")
    st.success(f"Video processing complete. Output saved to {output_path}")
    return output_path

# --- Streamlit App ---

def main():
    st.title("Background Replacement App")

    # Video Upload
    video_file = st.file_uploader("Upload your video", type=['mp4', 'mov'])
    if not video_file:
        st.warning("Please upload a video.")
        return

    # Background Handling
    bg_options = ["Default Green Screen", "Custom Background"]
    bg_choice = st.radio("Select a Background Type:", bg_options)


    background_path = DEFAULT_BACKGROUND_URL  # Initialize with default

    if bg_choice == "Custom Background":
        uploaded_bg = st.file_uploader("Upload your background image", type=['png', 'jpg', 'jpeg'])
        if uploaded_bg:
            with open(uploaded_bg.name, "wb") as f:
                f.write(uploaded_bg.getbuffer())
            background_path = convert_to_png(uploaded_bg.name)
            if not background_path:
               st.error("Error converting the background image")
               return
        else:
            st.warning("Please upload a background image or select the default")
            return
    else:
        default_image = download_image(DEFAULT_BACKGROUND_URL)
        if default_image:
            # Temporarily save the image as PNG so the processing function can read it.
           default_image_path = "default_background.png"
           default_image.save(default_image_path,"PNG")
           background_path = default_image_path
        else:
            st.error("Failed to load default background image.")
            return

    # Processing Button
    if st.button("Process Video"):
        try:
            with open(video_file.name, "wb") as f:
                f.write(video_file.getbuffer())
            output_video_path = process_video(video_file.name, background_path, OUTPUT_PATH, MASK_THRESHOLD, OUTPUT_FPS)
            if output_video_path:
                with open(output_video_path, 'rb') as file:
                  video_bytes = file.read()
                  st.download_button(label="Download Processed Video",
                                    data=video_bytes,
                                    file_name="processed_video.mp4",
                                    mime="video/mp4")

        except FileNotFoundError as e:
            st.error(e)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
