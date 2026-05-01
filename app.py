import gradio as gr
import os
import shutil
import subprocess
import time
import traceback

def process_audio(zip_file):
    if zip_file is None:
        return "Please upload a zip file containing .wav files.", ""

    # ✅ FIX 1: Handle both Gradio 3.x (object) and 4.x (string path)
    if hasattr(zip_file, "name"):
        zip_file_path = zip_file.name   # Gradio 3.x
    else:
        zip_file_path = zip_file        # Gradio 4.x returns string

    if not os.path.exists(zip_file_path):
        return f"Uploaded file not found at: {zip_file_path}", ""

    # 1. Setup workspace
    input_dir = "hf_input_data"
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)

    # 2. Extract files
    try:
        shutil.unpack_archive(zip_file_path, input_dir)
    except Exception as e:
        return f"Error unpacking zip: {str(e)}\n\nPath used: {zip_file_path}", ""

    # ✅ FIX 2: Verify infer.py exists before calling it
    if not os.path.exists("infer.py"):
        return "Error: infer.py not found in working directory.", ""

    # ✅ FIX 3: Show full stderr + stdout on failure
    try:
        result = subprocess.run(
            ["python", "infer.py", input_dir],
            capture_output=True,
            text=True,
            timeout=300  # prevent hanging
        )
        if result.returncode != 0:
            return (
                f"Inference Error (exit code {result.returncode}):\n"
                f"--- STDERR ---\n{result.stderr}\n"
                f"--- STDOUT ---\n{result.stdout}"
            ), ""
    except subprocess.TimeoutExpired:
        return "Error: Inference script timed out after 5 minutes.", ""
    except Exception as e:
        return f"Execution Error:\n{traceback.format_exc()}", ""

    # 4. Read results
    results_text = "No results.txt found after inference."
    if os.path.exists("results.txt"):
        with open("results.txt", "r") as f:
            results_text = f.read()

    time_text = "No time.txt found."
    if os.path.exists("time.txt"):
        with open("time.txt", "r") as f:
            time_text = f.read()

    return results_text, time_text


with gr.Blocks(title="Audio Anomaly Detection Pipeline") as demo:
    gr.Markdown("# 🚀 Audio Anomaly Detection")
    gr.Markdown("Upload a **ZIP file** containing your `.wav` recordings.")

    with gr.Row():
        file_input = gr.File(label="Upload WAV Files (ZIP)", file_types=[".zip"])

    with gr.Row():
        run_btn = gr.Button("Run Inference", variant="primary")

    with gr.Row():
        output_results = gr.Textbox(label="Predictions (results.txt)", lines=10)
        output_time = gr.Textbox(label="Timing Info (time.txt)", lines=2)

    run_btn.click(
        fn=process_audio,
        inputs=file_input,
        outputs=[output_results, output_time],
        api_name="run_inference"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)