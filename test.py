import mediapipe as mp
import cv2 as cv
import numpy as np
import pickle

model_dict = pickle.load(open('./Air Draw/model.p','rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6)

# Create a blank image with the same dimensions as the video frames
blank = np.zeros((500,500,3),dtype='uint8')

# Initialize previous coordinates
prev_x, prev_y = 0, 0

# Open the video capture
cap = cv.VideoCapture('http://192.168.1.2:4747/video')

while True:
    ret, img = cap.read()
     # Flip the image horizontally for a mirror effect
    img = cv.flip(img, 1)
    img = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)
    img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
            img_rgb,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())  

            data_aux = []  # Reset the data_aux list for each frame
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_aux.append(x)
                data_aux.append(y)

            data_aux = data_aux[:42]
            prediction = model.predict([np.array(data_aux)])
            print(prediction)

            # Only draw if the prediction is 0
            if prediction == '0':
                # Get the coordinates of the tip of the index finger (landmark 8)
                x = int(hand_landmarks.landmark[8].x * img.shape[1])
                y = int(hand_landmarks.landmark[8].y * img.shape[0])

                # If this is the first frame, initialize prev_x and prev_y
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                # Draw a line from the previous point to the current point
                blank = cv.line(blank, (prev_x, prev_y), (x, y), (255,255,255), 2)

                # Update previous coordinates
                prev_x, prev_y = x, y

    # Create a copy of the blank canvas to draw the pointer
    blank_pointer = blank.copy()

    # Draw a circle at the tip of the index finger on the blank canvas copy
    blank_pointer = cv.circle(blank_pointer, (prev_x, prev_y), 15, (0, 255, 0), 2)

    # Display the image with the pointer
    cv.imshow("Blank", blank_pointer)
    cv.imshow("Image", img)

    # Break the loop if 'q' is pressedq
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv.destroyAllWindows()
