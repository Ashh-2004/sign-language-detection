import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Must match the Kaggle dataset format so data can be merged / model reused
DATASET_PATH = 'sign_data.csv'

def get_landmarks(hand_landmarks):
    """
    Extracts 210 pairwise distances between 21 hand landmarks.
    This matches the Kaggle 'hand-sign-dataset' feature format exactly.
    """
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    dists = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dists.append(np.linalg.norm(points[i] - points[j]))
    return dists  # 210 values, un-normalized (train_model.py normalizes per-sample)

def collect_data_for_label(label, num_samples=200):
    cap = cv2.VideoCapture(0)

    print(f"\nGet ready to sign: '{label}'")
    print("Capturing starts in 3 seconds... (Press ESC to stop early)")
    time.sleep(3)

    samples_collected = 0
    all_data = []

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened() and samples_collected < num_samples:
            success, image = cap.read()
            if not success:
                continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    landmarks = get_landmarks(hand_landmarks)
                    all_data.append([label] + landmarks)
                    samples_collected += 1

            cv2.putText(image, f"Sign: {label}  Samples: {samples_collected}/{num_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow('Data Collection', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    if not all_data:
        print("No samples collected (no hand detected). Try again.")
        return

    # 210 pairwise distance columns — same schema as Kaggle sign_data.csv
    columns = ['Sign'] + [f'dist_{i}' for i in range(210)]
    df = pd.DataFrame(all_data, columns=columns)

    if not os.path.exists(DATASET_PATH):
        df.to_csv(DATASET_PATH, index=False)
        print(f"Created new dataset '{DATASET_PATH}' with {samples_collected} samples for '{label}'.")
    else:
        df.to_csv(DATASET_PATH, mode='a', header=False, index=False)
        print(f"Appended {samples_collected} samples for '{label}' to '{DATASET_PATH}'.")

if __name__ == '__main__':
    while True:
        sign_label = input("\nEnter sign label to collect (or 'q' to quit): ").strip()
        if sign_label.lower() == 'q':
            break
        num_str = input("How many samples? (default 200): ").strip()
        num = int(num_str) if num_str.isdigit() else 200
        collect_data_for_label(sign_label, num_samples=num)