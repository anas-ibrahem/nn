import os
import shutil
import subprocess
import sys
import tempfile
import gdown

import streamlit as st

def download_models():
    # Download encoder model
    encoder_path = "encoder_model.h5"
    if not os.path.exists(encoder_path):
        st.info("Downloading encoder_model.h5 (first time only)...")
        # ID: 1c1KdgVNx5L18pkuZjWQMHTqYMKUVyoiS
        gdown.download(id="1c1KdgVNx5L18pkuZjWQMHTqYMKUVyoiS", output=encoder_path, quiet=False)
        
    # Download AlexNet model
    os.makedirs("alexnet", exist_ok=True)
    alexnet_path = os.path.join("alexnet", "tf_alexnet.keras")
    if not os.path.exists(alexnet_path):
        st.info("Downloading tf_alexnet.keras (first time only)...")
        # ID: 1AaP8QskC3hmlA2QC0vuwNtvgyrBmOzHO
        gdown.download(id="1AaP8QskC3hmlA2QC0vuwNtvgyrBmOzHO", output=alexnet_path, quiet=False)

def run_inference(zip_file):
    if zip_file is None:
        return "Please upload a zip file containing .wav files.", "", ""

    work_dir = tempfile.mkdtemp(prefix="streamlit_input_")
    zip_path = os.path.join(work_dir, "input.zip")
    input_dir = os.path.join(work_dir, "data")

    with open(zip_path, "wb") as f:
        f.write(zip_file.getbuffer())

    try:
        shutil.unpack_archive(zip_path, input_dir)
    except Exception as e:
        return f"Error unpacking zip: {e}", "", ""

    cmd = [sys.executable, "infer.py", input_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        msg = stderr if stderr else stdout if stdout else "Inference failed."
        full_error = f"Exit Code: {result.returncode}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{msg}"
        return full_error, "", full_error

    results_text = ""
    time_text = ""

    if os.path.exists("results.txt"):
        with open("results.txt", "r") as f:
            results_text = f.read()

    if os.path.exists("time.txt"):
        with open("time.txt", "r") as f:
            time_text = f.read()

    return results_text, time_text, result.stdout.strip()


st.set_page_config(page_title="Audio Anomaly Detection", layout="centered")

st.title("Audio Anomaly Detection")
st.write("Upload a ZIP file containing your .wav recordings. This app runs the Autoencoder + AlexNet pipeline.")

uploaded = st.file_uploader("Upload WAV Files (ZIP)", type=["zip"])

if st.button("Run Inference"):
    with st.spinner("Running inference..."):
        download_models()
        results, timing, logs = run_inference(uploaded)

    if results and not results.startswith("Error"):
        st.success("Inference complete.")
    else:
        st.error(results or "Inference failed.")

    st.session_state["results"] = results
    st.session_state["timing"] = timing
    st.session_state["logs"] = logs

if "results" in st.session_state:
    st.subheader("Predictions (results.txt)")
    st.text_area("", st.session_state["results"], height=260)

if "timing" in st.session_state:
    st.subheader("Timing Info (time.txt)")
    st.text_area("", st.session_state["timing"], height=120)

if "logs" in st.session_state and st.session_state["logs"]:
    with st.expander("Inference Logs"):
        st.text(st.session_state["logs"])
