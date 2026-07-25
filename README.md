# Analysis of EEG Signals for Emotion Detection

I classify whether a student **understood an online lecture** from their raw EEG
signals — using classical signal processing for preprocessing and feature
extraction, and deep learning models for classification.

Internship project at SSN College of Engineering, under the guidance of
Dr. S. Kavitha (Associate Professor, Dept. of CSE), 2024.

## Problem

EEG recordings from **8 students** watching online lecture videos, each labelled:
`1` = understood the lecture, `0` = did not understand. The goal is to predict this
label from the EEG signal.

## Pipeline

See [`source_code.ipynb`](source_code.ipynb):

1. **Filtering** (per subject, per channel) — Butterworth bandpass (0.5–40 Hz),
   median (kernel 3), and notch (50 Hz). Uses the 14 Emotiv channels.
2. **Feature extraction** — sliding window (128 samples, 50% overlap) with
   time-domain (statistical, Hjorth), frequency-domain (PSD per band), time–frequency
   (wavelet), and nonlinear (entropy, Hurst, fractal dimension) features.
3. **Modeling** — 1D CNN, LSTM, and GRU.

## Results

| Model | Accuracy | Recall | Precision | F1 |
| --- | --- | --- | --- | --- |
| SVM | 84 | 82 | 74 | 77 |
| Random Forest | 88 | 91 | 78 | 84 |
| CNN | 86 | 93 | 74 | 82 |

Random Forest gave me the best results (~88% accuracy). SVM and Random Forest were
run separately and are not in this notebook.

## Running it

I wrote the notebook for **Google Colab**: upload the EEG CSV to your Drive, run the
mount cell, adjust the file path, and run top to bottom. To run locally, drop the
`google.colab` cell, `pip install -r requirements.txt`, and point the path at your
local file.

> The dataset (`EEG_data(1).csv`) isn't included — the notebook loads it from Google
> Drive. It has ~68,831 samples × 87 columns before feature selection.
