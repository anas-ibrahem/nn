import gradio as gr
import os
import shutil
import subprocess
import time

def process_audio(zip_file):
    if zip_file is None:
        return "Please upload a zip file containing .wav files.", ""
    
    # 1. Setup workspace
    input_dir = "hf_input_data"
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)
    
    # 2. Extract files
    try:
        shutil.unpack_archive(zip_file.name, input_dir)
    except Exception as e:
        return f"Error unpacking zip: {str(e)}", ""

    # 3. Run the inference script
    # We use the existing infer.py logic
    start_time = time.time()
    try:
        # Pass the extracted directory to infer.py
        result = subprocess.run(["python", "infer.py", input_dir], capture_output=True, text=True)
        if result.returncode != 0:
            return f"Inference Error: {result.stderr}", ""
    except Exception as e:
        return f"Execution Error: {str(e)}", ""
    
    # 4. Read results
    results_text = "Processing Error"
    if os.path.exists("results.txt"):
        with open("results.txt", "r") as f:
            results_text = f.read()
            
    time_text = ""
    if os.path.exists("time.txt"):
        with open("time.txt", "r") as f:
            time_text = f.read()

    return results_text, time_text

# Define Gradio Interface
with gr.Blocks(title="Audio Anomaly Detection Pipeline") as demo:
    gr.Markdown("# 🚀 Hybrid Audio Anomaly Detection")
    gr.Markdown("Upload a **ZIP file** containing your `.wav` recordings. This Space runs the Autoencoder + AlexNet/XGBoost pipeline.")
    
    with gr.Row():
        file_input = gr.File(label="Upload WAV Files (ZIP)", file_types=[".zip"])
    
    with gr.Row():
        run_btn = gr.Button("Run Inference", variant="primary")
    
    with gr.Row():
        output_results = gr.Textbox(label="Predictions (results.txt)", lines=10)
        output_time = gr.Textbox(label="Timing Info (time.txt)", lines=2)

    run_btn.click(fn=process_audio, inputs=file_input, outputs=[output_results, output_time])

if __name__ == "__main__":
    demo.launch()
