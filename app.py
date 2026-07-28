import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Set Streamlit page config
st.set_page_config(page_title="Sign Language to Text", layout="wide", page_icon="👋")

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

MODEL_PATH = 'model.pkl'

# RTC config so it works reliably on cloud hosts (uses a public STUN server)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


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


class SignLanguageProcessor:
    """Handles per-frame hand detection + prediction for streamlit-webrtc."""

    def __init__(self, model):
        self.model = model
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Shared state read by the main thread to update the prediction UI
        self.prediction_text = "N/A"
        self.conf_text = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        prediction_text = "N/A"
        conf_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                if self.model:
                    landmarks = get_landmarks(hand_landmarks)
                    try:
                        prediction = self.model.predict([landmarks])[0]

                        if hasattr(self.model, "predict_proba"):
                            probabilities = self.model.predict_proba([landmarks])[0]
                            confidence = max(probabilities)
                            conf_text = f" ({confidence * 100:.1f}%)"

                            if confidence > 0.6:
                                prediction_text = prediction
                            else:
                                prediction_text = "Uncertain"
                        else:
                            prediction_text = prediction
                    except Exception as e:
                        prediction_text = f"Error: {e}"

        self.prediction_text = prediction_text
        self.conf_text = conf_text

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


def main():
    st.title("Real-Time Sign Language to Text Converter 👋")
    st.markdown(
        "This dashboard uses your webcam to detect hand gestures and converts them to text using a trained machine learning model."
    )

    model = load_sign_model()

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Prediction")
        placeholder = st.empty()

        if not model:
            st.warning("Model not found! Please train the model first by running `train_model.py`.")

        st.markdown("### Instructions")
        st.markdown(
            """
        1. Click **Start** below and allow camera access in your browser.
        2. Perform the sign inside the camera frame.
        3. The predicted letter will appear here.
        """
        )

    with col1:
        ctx = webrtc_streamer(
            key="sign-detect",
            video_processor_factory=lambda: SignLanguageProcessor(model),
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

    # Poll the processor's latest prediction and update the sidebar text
    if ctx.video_processor:
        prediction_text = ctx.video_processor.prediction_text
        conf_text = ctx.video_processor.conf_text

        if prediction_text not in ("N/A", "Uncertain"):
            placeholder.markdown(
                f"<h1 style='text-align: center; color: green;'>{prediction_text}{conf_text}</h1>",
                unsafe_allow_html=True,
            )
        elif prediction_text == "Uncertain":
            placeholder.markdown(
                "<h1 style='text-align: center; color: orange;'>Uncertain</h1>",
                unsafe_allow_html=True,
            )
        else:
            placeholder.markdown(
                "<h1 style='text-align: center; color: gray;'>Waiting for hand...</h1>",
                unsafe_allow_html=True,
            )


if __name__ == '__main__':
    main()
