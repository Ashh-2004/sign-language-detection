import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Set Streamlit page config
st.set_page_config(page_title="Sign Language to Text", layout="wide", page_icon="👋")

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

MODEL_PATH = 'model.pkl'

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
    
    # Calculate all pairwise distances (21 * 20 / 2 = 210 features)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            landmarks.append(dist)
            
    # Normalize by max distance to make it scale invariant
    max_dist = max(landmarks)
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
        
    camera = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while run:
            success, frame = camera.read()
            if not success:
                st.error("Failed to read from webcam.")
                break
                
            # Process Frame
            frame = cv2.flip(frame, 1) # Mirror image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            prediction_text = "N/A"
            conf_text = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    if model:
                        landmarks = get_landmarks(hand_landmarks)
                        try:
                            # Predict
                            prediction = model.predict([landmarks])[0]
                            # Get probability if available
                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba([landmarks])[0]
                                confidence = max(probabilities)
                                conf_text = f" ({confidence*100:.1f}%)"
                                
                                # Use a confidence threshold
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
            
            # Use HTML/CSS to make the text large and visible
            if prediction_text != "N/A" and prediction_text != "Uncertain":
                placeholder.markdown(f"<h1 style='text-align: center; color: green;'>{prediction_text}{conf_text}</h1>", unsafe_allow_html=True)
            elif prediction_text == "Uncertain":
                 placeholder.markdown(f"<h1 style='text-align: center; color: orange;'>{prediction_text}</h1>", unsafe_allow_html=True)
            else:
                placeholder.markdown(f"<h1 style='text-align: center; color: gray;'>Waiting for hand...</h1>", unsafe_allow_html=True)

        else:
            if camera.isOpened():
                camera.release()
                st.write("Stopped")

if __name__ == '__main__':
    main()
