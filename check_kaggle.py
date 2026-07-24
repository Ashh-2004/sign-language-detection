import pandas as pd
import numpy as np
import cv2
import os
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands

df = pd.read_csv(r'C:\Users\ashis\.cache\kagglehub\datasets\harshitpathak18\hand-sign-dataset\versions\2\sign_data.csv')
row_A = df[df['Sign'] == 'A'].iloc[:, :-1].values # All 100 A signatures

folder = r'C:\Users\ashis\.cache\kagglehub\datasets\harshitpathak18\hand-sign-dataset\versions\2\data\data\A'

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)

found = False
for img_name in os.listdir(folder):
    img_path = os.path.join(folder, img_name)
    img = cv2.imread(img_path)
    if img is None: continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        print(f"Found hand in {img_name}")
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # 3D
        points_3d = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        dists_3d = []
        for i in range(len(points_3d)):
            for j in range(i+1, len(points_3d)):
                dists_3d.append(np.linalg.norm(points_3d[i] - points_3d[j]))
                
        # 2D
        points_2d = np.array([[lm.x, lm.y] for lm in landmarks])
        dists_2d = []
        for i in range(len(points_2d)):
            for j in range(i+1, len(points_2d)):
                dists_2d.append(np.linalg.norm(points_2d[i] - points_2d[j]))
                
        print("Our 3D mean:", np.mean(dists_3d), "Kaggle A mean:", np.mean(row_A))
        print("Our 2D mean:", np.mean(dists_2d))
        
        found = True
        break
        
if not found:
    print("Could not find any hands in any image!")
