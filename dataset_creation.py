import mediapipe as mp
import os
import cv2 as cv
import pickle

DATA_DIR = './Air Draw/data'

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6)

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for j in os.listdir(os.path.join(DATA_DIR,dir_)):
        data_aux = []
        img = cv.imread(os.path.join(DATA_DIR,dir_,str(j)))
        img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(dir_)

f = open('./Air Draw/data.pickle','wb')
pickle.dump({'data':data, 'labels':labels},f)
f.close()