import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Set Streamlit page config
st.set_page_config(page_title="Sign Language to Text", layout="wide", page_icon="👋")

MODEL_PATH = 'model.pkl'

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
        self.detector = None
        self.hands = None
        self.error_msg = None
        
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
                self.error_msg = str(e)
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
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                self.error_msg = str(e)

    def process(self, frame_rgb):
        landmarks_list = []
        if self.use_solutions and self.hands:
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_rgb,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    landmarks_list.append(hand_landmarks.landmark)
        elif self.detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.detector.detect(mp_image)
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks_cv(frame_rgb, hand_landmarks)
                    landmarks_list.append(hand_landmarks)
        return landmarks_list

    def close(self):
        if self.use_solutions and hasattr(self, 'hands'):
            self.hands.close()

@st.cache_resource(show_spinner="Loading scale-invariant model...")
def load_sign_model():
    """Loads the trained model."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    return None

def get_landmarks(landmark_points):
    """Extracts 210 pairwise distances between 21 hand landmarks to match Kaggle dataset format."""
    landmarks = []
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmark_points])
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            landmarks.append(dist)
            
    max_dist = max(landmarks) if landmarks else 0
    if max_dist > 0:
        landmarks = [d / max_dist for d in landmarks]
            
    return landmarks

def main():
    st.title("Real-Time Sign Language to Text Converter 👋")
    st.markdown("This dashboard uses your webcam to detect hand gestures and converts them to text using a trained machine learning model.")

    model = load_sign_model()
    
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Prediction")
        placeholder = st.empty()
        
        if not model:
            st.warning("Model not found! Please train the model first by running `train_model.py`.")
            
        st.markdown("### Instructions")
        st.markdown("""
        1. Make sure your webcam is enabled.
        2. Check the 'Run Webcam' box to start.
        3. Perform the sign inside the camera frame.
        """)

    with col1:
        run = st.checkbox('Run Webcam')
        FRAME_WINDOW = st.image([])
        
    if run:
        camera = cv2.VideoCapture(0)
        tracker = HandTracker()
        
        while run:
            success, frame = camera.read()
            if not success:
                st.error("Failed to read from webcam.")
                break
                
            # Process Frame
            frame = cv2.flip(frame, 1) # Mirror image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            landmarks_list = tracker.process(frame_rgb)
            
            prediction_text = "N/A"
            conf_text = ""

            if landmarks_list:
                for landmark_points in landmarks_list:
                    if model:
                        landmarks = get_landmarks(landmark_points)
                        try:
                            # Predict
                            prediction = model.predict([landmarks])[0]
                            # Get probability if available
                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba([landmarks])[0]
                                confidence = max(probabilities)
                                conf_text = f" ({confidence*100:.1f}%)"
                                
                                if confidence > 0.6:
                                    prediction_text = prediction
                                else:
                                    prediction_text = "Uncertain"
                            else:
                                prediction_text = prediction
                        except Exception as e:
                            st.write(f"Prediction error: {e}")
                            
            # Update UI
            FRAME_WINDOW.image(frame_rgb)
            
            if prediction_text != "N/A" and prediction_text != "Uncertain":
                placeholder.markdown(f"<h1 style='text-align: center; color: green;'>{prediction_text}{conf_text}</h1>", unsafe_allow_html=True)
            elif prediction_text == "Uncertain":
                 placeholder.markdown(f"<h1 style='text-align: center; color: orange;'>{prediction_text}</h1>", unsafe_allow_html=True)
            else:
                placeholder.markdown(f"<h1 style='text-align: center; color: gray;'>Waiting for hand...</h1>", unsafe_allow_html=True)

        camera.release()
        tracker.close()
        st.write("Stopped")

if __name__ == '__main__':
    main()

