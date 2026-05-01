import gradio as gr
import os
import shutil
import subprocess
import time
import traceback
import sys

# ─── DEBUG FUNCTION ───────────────────────────────────────────────
def debug_environment():
    lines = []

    lines.append("=== PYTHON ===")
    lines.append(sys.executable)
    lines.append(sys.version)

    lines.append("\n=== WORKING DIRECTORY ===")
    cwd = os.getcwd()
    lines.append(cwd)

    lines.append("\n=== FILES IN CWD ===")
    try:
        for f in sorted(os.listdir(cwd)):
            lines.append(f"  {f}")
    except Exception as e:
        lines.append(f"ERROR: {e}")

    lines.append("\n=== infer.py EXISTS? ===")
    lines.append(str(os.path.exists("infer.py")))

    lines.append("\n=== ENVIRONMENT VARIABLES ===")
    for k, v in sorted(os.environ.items()):
        lines.append(f"  {k}={v}")

    lines.append("\n=== INSTALLED PACKAGES (pip list) ===")
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "list"],
                           capture_output=True, text=True, timeout=30)
        lines.append(r.stdout)
        if r.stderr:
            lines.append("STDERR: " + r.stderr)
    except Exception as e:
        lines.append(f"ERROR: {e}")

    lines.append("\n=== TEST: can python run at all? ===")
    try:
        r = subprocess.run([sys.executable, "-c", "print('subprocess OK')"],
                           capture_output=True, text=True, timeout=10)
        lines.append(f"stdout: {r.stdout.strip()}")
        lines.append(f"stderr: {r.stderr.strip()}")
        lines.append(f"returncode: {r.returncode}")
    except Exception as e:
        lines.append(f"ERROR: {e}")

    lines.append("\n=== TEST: import infer ===")
    try:
        r = subprocess.run([sys.executable, "-c", "import infer; print('import OK')"],
                           capture_output=True, text=True, timeout=15)
        lines.append(f"stdout: {r.stdout.strip()}")
        lines.append(f"stderr: {r.stderr.strip()}")
    except Exception as e:
        lines.append(f"ERROR: {e}")

    return "\n".join(lines)


# ─── MAIN FUNCTION ────────────────────────────────────────────────
def process_audio(zip_file):
    if zip_file is None:
        return "Please upload a zip file.", ""

    zip_file_path = zip_file.name if hasattr(zip_file, "name") else zip_file

    if not os.path.exists(zip_file_path):
        return f"Uploaded file not found at path: {zip_file_path}", ""

    input_dir = "hf_input_data"
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedi