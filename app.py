import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Set Streamlit page config
st.set_page_config(page_title="Sign Language to Text", layout="wide", page_icon="👋")

MODEL_PATH = 'model.pkl'

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

@st.cache_resource(show_spinner="Loading scale-invariant model...")
def load_sign_model():
    """Loads the trained model."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    return None

def get_landmarks(hand_landmarks):
    """Extracts 210 pairwise distances between 21 hand landmarks to match Kaggle dataset format."""
    landmarks = []
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            landmarks.append(dist)
            
    max_dist = max(landmarks) if landmarks else 0
    if max_dist > 0:
        landmarks = [d / max_dist for d in landmarks]
            
    return landmarks

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.model = load_sign_model()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(img_rgb)
        
        prediction_text = "N/A"
        conf_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                if self.model:
                    landmarks = get_landmarks(hand_landmarks)
                    try:
                        prediction = self.model.predict([landmarks])[0]
                        if hasattr(self.model, "predict_proba"):
                            probabilities = self.model.predict_proba([landmarks])[0]
                            confidence = max(probabilities)
                            conf_text = f" ({confidence*100:.1f}%)"
                            if confidence > 0.6:
                                prediction_text = prediction
                            else:
                                prediction_text = "Uncertain"
                        else:
                            prediction_text = prediction
                    except Exception as e:
                        prediction_text = f"Error: {e}"

        overlay_text = f"Sign: {prediction_text}{conf_text}" if prediction_text != "N/A" else "Waiting for hand..."
        color = (0, 255, 0) if prediction_text not in ["N/A", "Uncertain"] else ((0, 165, 255) if prediction_text == "Uncertain" else (200, 200, 200))
        
        cv2.rectangle(img, (10, 10), (550, 65), (0, 0, 0), -1)
        cv2.putText(img, overlay_text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Real-Time Sign Language to Text Converter 👋")
    st.markdown("This dashboard uses your webcam to detect hand gestures and converts them to text using a trained machine learning model.")

    model = load_sign_model()
    
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Instructions")
        st.markdown("""
        1. Click **START** on the video player on the left.
        2. Allow camera access in your browser.
        3. Perform sign language gestures inside the camera frame.
        4. Real-time predictions will appear live on the video feed.
        """)
        
        if not model:
            st.warning("Model not found! Please train the model first by running `train_model.py`.")

    with col1:
        st.markdown("### WebRTC Live Webcam Stream")
        webrtc_streamer(
            key="sign-language-detection",
            video_processor_factory=SignLanguageProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

if __name__ == '__main__':
    main()



