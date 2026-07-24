# Real-Time Sign Language to Text Converter 🖐️💬

A real-time Computer Vision and Machine Learning application that detects hand sign gestures via webcam and translates them into text in real-time using **MediaPipe** landmark tracking and a **Scikit-Learn** classifier built into an interactive **Streamlit** web dashboard.

---

## 🌟 Key Features

- **Real-Time Hand Tracking**: Detects 21 3D hand landmark coordinates in real-time using MediaPipe Hands.
- **Scale-Invariant Feature Extraction**: Computes 210 pairwise normalized Euclidean distances between all landmark points.
- **Machine Learning Classification**: Trained model predicts gesture letters with confidence probability scores.
- **Interactive Dashboard**: Clean Streamlit user interface displaying live camera feed, predictions, and confidence levels.
- **Dataset Collection & Training Pipeline**: Scripts included to collect custom gesture datasets and train custom Random Forest models.

---

## 🚀 Quick Start Guide

### 1. Create & Activate Virtual Environment (Python 3.10)

```powershell
python -m venv venv310
.\venv310\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```

---

## 📜 License
This project is open-source under the MIT License.
