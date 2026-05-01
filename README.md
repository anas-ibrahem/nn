---
title: Team 19 Audio Anomaly Detection
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
python_version: 3.10.13
---

# Audio Anomaly Detection Pipeline

This Space runs a deep learning pipeline for detecting anomalies in industrial audio recordings.

## Pipeline Overview
1. **Denoising:** High-pass filtering and spectral subtraction.
2. **Feature Extraction:** Mel-spectrogram generation and normalization.
3. **Dimensionality Reduction:** Keras Autoencoder (Encoder) extracts 256D embeddings.
4. **Classification:** 1D AlexNet predicts the machine state.

## How to use
Upload a `.zip` file containing `.wav` files. The space will return a `results.txt` with predictions and `time.txt` with execution latency.
