# ✋ Real-Time Sign Language to Text Converter

A real-time computer vision system that detects hand gestures using MediaPipe and converts them into text using a trained Machine Learning model.

---

## 🚀 Features

- Real-time hand tracking using webcam
- Gesture recognition using ML model
- Scale-invariant feature extraction (210 distances)
- Confidence-based predictions
- Streamlit-based UI
- Custom dataset collection support
- Kaggle dataset integration

---

## 🧠 How It Works

1. MediaPipe detects 21 hand landmarks
2. 210 pairwise distances are computed
3. Features are normalized (scale invariant)
4. Model predicts gesture using Random Forest
5. Output is displayed in real-time

---

## 📁 Project Structure

sign-language-system/
│
├── app.py               # Streamlit app
├── train_model.py       # Model training
├── collect_data.py      # Data collection
├── check_kaggle.py      # Dataset validation
├── model.pkl            # Trained model
├── dataset.csv          # Dataset
└── requirements.txt

---

## ⚙️ Installation

git clone https://github.com/yourusername/sign-language-detector.git
cd sign-language-detector

pip install -r requirements.txt

---

## ▶️ Run the App

streamlit run app.py

---

## 🧪 Train the Model

python train_model.py

- Loads dataset (Kaggle or local)
- Normalizes features
- Trains Random Forest model
- Saves model as model.pkl

---

## 📸 Collect Custom Data

python collect_data.py

- Uses webcam to capture gestures
- Stores 210-distance features
- Appends to dataset

---

## 🔍 Validate Dataset (Optional)

python check_kaggle.py

- Compares extracted features with Kaggle dataset
- Ensures consistency

---

## 📊 Model Details

Algorithm: Random Forest  
Features: 210 pairwise distances  
Normalization: Per-sample scaling  
Input: 3D hand landmarks  

---

## ⚠️ Limitations

- Single-hand detection only
- Limited gesture vocabulary
- Accuracy depends on lighting
- No sentence formation

---

## 🔧 Future Improvements

- Sequence modeling (LSTM)
- Larger dataset (A–Z + words)
- Sentence builder
- Text-to-speech output
- Multi-hand detection
- Mobile deployment

---

## 🧠 Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Streamlit

---

## 📌 Conclusion

This project demonstrates real-time computer vision, feature engineering, and machine learning deployment.

To make it production-ready:
- Add FastAPI backend
- Improve model (XGBoost / Deep Learning)
- Deploy on cloud
