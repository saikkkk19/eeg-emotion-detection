# Analysis of EEG Signals for Emotion Detection

In this project I classify whether a student **understood an online lecture** from
their raw EEG signals, using classical signal processing for preprocessing and
feature extraction, and deep learning models for classification.

This is my final-year B.Sc. (AI & DA) project at Hindustan Institute of Technology
and Science, which I carried out under the guidance of **Dr. S. Kavitha**,
Associate Professor, Dept. of CSE, SSN College of Engineering (July 2024).

## Problem

I worked with EEG recordings collected from **8 students** watching online lecture
videos during the COVID-19 lockdown. For each (subject, video) the data is labelled
with a binary target:

- `1` — the subject **understood** the lecture
- `0` — the subject **did not understand** the lecture

My goal is to predict this understanding label from the EEG signal.

## Pipeline

The notebook [`eeg_filtering_feature_extraction_modeling.ipynb`](eeg_filtering_feature_extraction_modeling.ipynb)
implements my full pipeline:

### 1. Preprocessing — filtering
I apply the following per subject, per EEG channel:
- **Bandpass filter** — Butterworth, 0.5–40 Hz (keeps delta→low-gamma bands)
- **Median filter** — kernel size 3 (removes transient spikes)
- **Notch filter** — 50 Hz (removes power-line interference)

The 14 Emotiv channels I use: `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`.

### 2. Feature extraction
I slide a window (128 samples, 50% overlap) over each channel and extract the
following feature families per window:

| Category | Features |
| --- | --- |
| Time domain | Statistical (mean, median, mode, std, variance, min, max, RMS, distance), Hjorth (activity, mobility, complexity) |
| Frequency domain | Power Spectral Density per band (delta, theta, low/mid/high beta, low/high gamma) via Welch |
| Time–frequency | Discrete Wavelet Transform (db4) — approximation & detail coefficient stats/energy |
| Nonlinear | Differential / Sample / Approximate entropy, Hurst exponent, Higuchi fractal dimension |

### 3. Modeling
I train and evaluate three deep-learning models:
- **1D CNN** (Conv1D → MaxPooling → Dense)
- **LSTM**
- **GRU**

> Note: my written report also compares **SVM** and **Random Forest** (Random Forest
> gave me the best results, ~88% accuracy). I ran those classical models separately,
> so they are not included in this notebook.

## Results (from my project report)

| Model | Accuracy | Recall | Precision | F1 |
| --- | --- | --- | --- | --- |
| SVM | 84 | 82 | 74 | 77 |
| Random Forest | 88 | 91 | 78 | 84 |
| CNN | 86 | 93 | 74 | 82 |

## Dataset

I did **not** include the dataset (`EEG_data(1).csv`) in this repository — the
notebook reads it from my Google Drive. To reproduce my work, place your EEG CSV
somewhere accessible and update the file path in the loading cells. The full dataset
has ~68,831 samples × 87 columns before feature selection.

Expected columns include: `subject_id`, `video_id`, `subject_understood`, the 14
`EEG.*` channel columns, plus demographic/metadata fields.

## Running the notebook

I wrote the notebook for **Google Colab**. To run it there:
1. Upload `EEG_data(1).csv` to your Google Drive.
2. Open the notebook in Colab and run the first cell to mount Drive.
3. Adjust the CSV path if needed and run the cells top to bottom.

To run it locally instead, remove the `google.colab` mount cell, install the
dependencies below, and point the CSV path at your local file.

## Dependencies

```bash
pip install -r requirements.txt
```

Core libraries: `numpy`, `pandas`, `scipy`, `PyWavelets`, `antropy`, `nolds`,
`nitime`, `spectrum`, `pyedflib`, `scikit-learn`, `tensorflow`, `matplotlib`,
`seaborn`.

## References

Key sources from my report:
1. Bhatt et al., *Machine learning for cognitive behavioral analysis*, Brain Informatics (2023).
2. Samal & Hashmi, *Role of ML/DL in EEG-based BCI emotion recognition*, Artificial Intelligence Review (2024).
3. Wang et al., *Deep learning-based EEG emotion recognition*, Frontiers in Psychology (2023).
4. Huang et al., *An Online Teaching Video Evaluation Scheme Based on EEG Signals and ML*, Complexity (2022).
5. Zhou & Dou, *EEG4Students*, arXiv:2208.11743 (2022).
