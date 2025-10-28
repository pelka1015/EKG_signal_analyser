## ❤️ ECG Signal Analysis and Filtering

This project provides a Python-based tool for analyzing and processing ECG (electrocardiogram) signals from CSV files. It uses digital filtering techniques to enhance signal quality and automatically detects key features of the ECG waveform, including P, Q, R, S, and T waves. The script also estimates the heart rate based on detected R-peaks.

## 📋 Features

- High-pass and band-stop filtering of ECG signals

- Time and frequency domain visualizations

- Automatic detection of:

- P, Q, R, S, T peaks

- Heart rate calculation (beats per minute)

- Interactive file selection using a GUI file dialog

- Clear, annotated plots for analysis and validation

## 🧠 How It Works

1. Load Data: Uses a file dialog to select a CSV containing raw ECG data (one float per line).

2. Filtering:

    - Applies a high-pass filter to remove low-frequency drift.

    - Applies a notch (band-stop) filter to eliminate line noise (e.g., 100 Hz).

3. Peak Detection:

    - Detects local maxima and classifies them as P, Q, R, S, or T peaks based on amplitude and position.

4. Plotting:

    - Time-domain and frequency-domain plots before and after filtering.

    - Final plot highlights detected peaks and displays the estimated heart rate.


## 📦 Requirements
Install required libraries using pip:

```bash
pip install numpy matplotlib scipy
```
This script also uses tkinter, which is included with most Python installations.

## 📊 Example Output

<img width="1429" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/0bed75d9-05a3-4b6e-a1e2-409089bae8d9" />

## ✒️ Authors
- Patryk Pełka
- Robert Wewersowicz


## 🧠 Citation

If you use this project in academic work, please cite or reference this repository.
