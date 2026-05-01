import os
import shutil
import subprocess
import sys
import tempfile
import gdown
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import librosa

st.set_page_config(page_title="Audio Anomaly Detection", layout="wide", page_icon="🎙️")

def download_models():
    encoder_path = "encoder_model.h5"
    if not os.path.exists(encoder_path):
        st.info("Downloading encoder_model.h5 (first time only)...")
        gdown.download(id="1c1KdgVNx5L18pkuZjWQMHTqYMKUVyoiS", output=encoder_path, quiet=False)
        
    os.makedirs("alexnet", exist_ok=True)
    alexnet_path = os.path.join("alexnet", "tf_alexnet.keras")
    if not os.path.exists(alexnet_path):
        st.info("Downloading tf_alexnet.keras (first time only)...")
        gdown.download(id="1AaP8QskC3hmlA2QC0vuwNtvgyrBmOzHO", output=alexnet_path, quiet=False)

def execute_pipeline(input_dir):
    cmd = [sys.executable, "infer.py", input_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Exit Code: {result.returncode}\n\nSTDOUT:\n{result.stdout.strip()}\n\nSTDERR:\n{result.stderr.strip()}", "", result.stdout.strip()
    
    results_text, time_text = "", ""
    if os.path.exists("results.txt"):
        with open("results.txt", "r") as f:
            results_text = f.read().strip()
    if os.path.exists("time.txt"):
        with open("time.txt", "r") as f:
            time_text = f.read().strip()
            
    return results_text, time_text, result.stdout.strip()

# --- UI START ---
st.title("🎙️ Audio Anomaly Detection Pipeline")
st.markdown("Upload audio files to run through the **Autoencoder + 1D AlexNet** architecture.")

# Sidebar Options
st.sidebar.header("⚙️ Deployment Options")
show_logs = st.sidebar.checkbox("Show Inference Logs", value=False)
show_viz = st.sidebar.checkbox("Generate Waveform Plot", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Instructions:**\n"
    "- Upload a **Single WAV** to quickly test the pipeline.\n"
    "- Upload a **ZIP file** of WAVs for batch processing.\n"
    "- Models automatically pull from Google Drive on the first run."
)

tab1, tab2 = st.tabs(["🎵 Single File Inference", "📁 Batch Inference (ZIP)"])

# ====== TAB 1: SINGLE FILE ======
with tab1:
    st.header("Analyze a Single Audio File")
    single_file = st.file_uploader("Upload a single .wav file", type=["wav"])
    
    if single_file is not None:
        st.audio(single_file, format="audio/wav")
        
        if show_viz:
            with st.spinner("Generating audio waveform..."):
                y, sr = librosa.load(single_file, sr=None)
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#1f77b4', linewidth=0.5)
                ax.set_title("Raw Audio Waveform")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)
                plt.close(fig)
        
        if st.button("Run Prediction", key="btn_single"):
            with st.spinner("Analyzing audio..."):
                download_models()
                
                work_dir = tempfile.mkdtemp(prefix="streamlit_single_")
                input_dir = os.path.join(work_dir, "data")
                os.makedirs(input_dir, exist_ok=True)
                
                # Save the uploaded file as 1.wav so infer.py can read it
                file_path = os.path.join(input_dir, "1.wav")
                with open(file_path, "wb") as f:
                    f.write(single_file.getbuffer())
                    
                results, timing, logs = execute_pipeline(input_dir)
                
                if results and not results.startswith("Exit Code"):
                    st.success("✅ Analysis Complete")
                    st.metric(label="Predicted Class Output", value=f"Class {results}")
                    st.caption(f"Inference Time: {timing} seconds")
                else:
                    st.error("Inference failed.")
                    st.code(results)
                    
                if show_logs and logs:
                    with st.expander("Detailed Process Logs"):
                        st.text(logs)

# ====== TAB 2: BATCH ZIP ======
with tab2:
    st.header("Batch Processing")
    uploaded_zip = st.file_uploader("Upload WAV Files as a ZIP archive", type=["zip"])
    
    if uploaded_zip is not None:
        if st.button("Run Batch Inference", key="btn_batch"):
            with st.spinner("Processing batch jobs..."):
                download_models()
                
                work_dir = tempfile.mkdtemp(prefix="streamlit_zip_")
                zip_path = os.path.join(work_dir, "input.zip")
                input_dir = os.path.join(work_dir, "data")
                
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())
                    
                shutil.unpack_archive(zip_path, input_dir)
                
                results, timing, logs = execute_pipeline(input_dir)
                
                if results and not results.startswith("Exit Code"):
                    st.success("✅ Batch Inference Complete")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Predictions (results.txt)")
                        st.text_area("", results, height=250, label_visibility="collapsed")
                    with col2:
                        st.subheader("Timing Info (time.txt)")
                        st.text_area("", timing, height=250, label_visibility="collapsed")
                else:
                    st.error("Inference failed.")
                    st.code(results)
                    
                if show_logs and logs:
                    with st.expander("Detailed Process Logs"):
                        st.text(logs)
