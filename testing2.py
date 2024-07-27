# Model testing
# Testing

import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open(r'C:\Users\hp\Desktop\Computer Vision\model_svm2.p', 'rb'))
model = model_dict['model']

# Open the video capture device (webcam)
cap = cv2.VideoCapture(0)

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Dictionary to map predicted class indices to characters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

while True:
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Store hand landmarks in lists
            x_coords = []
            y_coords = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # Append coordinates to the lists
                x_coords.append(x)
                y_coords.append(y)

            # Normalize coordinates
            x_min = min(x_coords)
            y_min = min(y_coords)

            normalized_coords = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                normalized_x = x - x_min
                normalized_y = y - y_min

                normalized_coords.append(normalized_x)
                normalized_coords.append(normalized_y)

            # Calculate bounding box coordinates
            x1 = int(x_min * W) - 10
            y1 = int(y_min * H) - 10
            x2 = int(max(x_coords) * W) - 10
            y2 = int(max(y_coords) * H) - 10

            # Predict the hand sign
            prediction = model.predict([np.asarray(normalized_coords)])
            predicted_character = labels_dict[int(prediction[0])]

            # Display bounding box and predicted character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Show the frame
    cv2.imshow('frame', frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
#sbcr