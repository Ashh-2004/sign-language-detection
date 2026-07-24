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

# Check MediaPipe API version
HAS_SOLUTIONS = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

def draw_landmarks_cv(image, hand_landmarks):
    height, width, _ = image.shape
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = (int(hand_landmarks[start_idx].x * width), int(hand_landmarks[start_idx].y * height))
        pt2 = (int(hand_landmarks[end_idx].x * width), int(hand_landmarks[end_idx].y * height))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * width), int(lm.y * height)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

class HandTracker:
    def __init__(self):
        self.use_solutions = HAS_SOLUTIONS
        self.hands = None
        self.detector = None
        
        if self.use_solutions:
            try:
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"Solutions init error: {e}")
        else:
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
                if not os.path.exists(model_path):
                    import urllib.request
                    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
                    urllib.request.urlretrieve(url, model_path)
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                print(f"HandLandmarker init error: {e}")

    def process(self, img_bgr):
        landmarks_list = []
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = np.ascontiguousarray(img_rgb)
            
            if self.use_solutions and self.hands:
                results = self.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            img_bgr,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        landmarks_list.append(hand_landmarks.landmark)
            elif self.detector:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = self.detector.detect(mp_image)
                if result.hand_landmarks:
                    for hand_landmarks in result.hand_landmarks:
                        draw_landmarks_cv(img_bgr, hand_landmarks)
                        landmarks_list.append(hand_landmarks)
        except Exception as e:
            print(f"Tracker process error: {e}")
                    
        return landmarks_list

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
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
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
        self.tracker = None
        try:
            self.tracker = HandTracker()
        except Exception as e:
            print(f"Processor tracker init error: {e}")
        self.model = load_sign_model()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1) # Mirror image
            
            prediction_text = "N/A"
            conf_text = ""

            if self.tracker:
                landmarks_list = self.tracker.process(img)
                if landmarks_list:
                    for landmark_points in landmarks_list:
                        if self.model:
                            landmarks = get_landmarks(landmark_points)
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
        except Exception as e:
            return frame

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




