import cv2
import mediapipe
import torch
import pandas as pd
import numpy as np
from CNNModel import CNNModel

# Load model
model = CNNModel()
model.load_state_dict(torch.load("CNN_model_alphabet_SIBI.pth"))
model.eval()

# Camera
cap = cv2.VideoCapture(0)
handTracker = mediapipe.solutions.hands
drawing = mediapipe.solutions.drawing_utils
drawingStyles = mediapipe.solutions.drawing_styles

handDetector = handTracker.Hands(static_image_mode=True, min_detection_confidence=0.2)

# Classes
classes = {
    'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'J': 9,
    'K': 10,'L': 11,'M': 12,'N': 13,'O': 14,'P': 15,'Q': 16,'R': 17,
    'S': 18,'T': 19,'U': 20,'V': 21,'W': 22,'X': 23,'Y': 24,'Z': 25
}
reverse_classes = {v: k for k, v in classes.items()}

current_word = ""
current_predicted_letter = None
sentence_text = ""

# Create sentence window
sentence_window = "Sentence"
cv2.namedWindow(sentence_window)

def show_sentence(sentence):
    blank = 255 * np.ones((200, 800, 3), np.uint8)
    cv2.putText(blank, sentence, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.imshow(sentence_window, blank)

show_sentence(sentence_text)

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = handDetector.process(frameRGB)

    predicted_letter = None
    x_list, y_list, z_list = [], [], []

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand,
                handTracker.HAND_CONNECTIONS,
                drawingStyles.get_default_hand_landmarks_style(),
                drawingStyles.get_default_hand_connections_style()
            )

            # Extract landmarks
            for lm in hand.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)
                z_list.append(lm.z)

            # Normalize
            data = {}
            for i, lm in enumerate(hand.landmark):
                data[f"LM_{i}_x"] = lm.x - min(x_list)
                data[f"LM_{i}_y"] = lm.y - min(y_list)
                data[f"LM_{i}_z"] = lm.z - min(z_list)

            df = pd.DataFrame([data])
            coords = np.reshape(df.values, (df.shape[0], 63, 1))
            coords = torch.from_numpy(coords).float()

            # Predict
            with torch.no_grad():
                out = model(coords)
                _, pred = torch.max(out.data, 1)
                predicted_letter = reverse_classes[pred.item()]
                current_predicted_letter = predicted_letter

            # Bounding box
            x1 = int(min(x_list) * w) - 10
            y1 = int(min(y_list) * h) - 10
            x2 = int(max(x_list) * w) + 10
            y2 = int(max(y_list) * h) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.putText(frame, predicted_letter, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    # -----------------------------
    # KEYBOARD CONTROLS
    # -----------------------------
    key = cv2.waitKey(1)

    # Press C → add predicted letter to the current word
    if key == ord('c') and current_predicted_letter:
        current_word += current_predicted_letter

    # SPACE → finalize word and add to sentence
    if key == 32:  # Spacebar
        if current_word.strip() != "":
            sentence_text += current_word + " "
            current_word = ""
            show_sentence(sentence_text)

    # ENTER → clear full sentence
    if key == 13:
        sentence_text = ""
        show_sentence(sentence_text)

    # BACKSPACE → delete last character
    if key == 8:
        current_word = current_word[:-1]

    # Quit
    if key == ord('q'):
        break

    # Display current predicted letter
    cv2.putText(frame, f"Letter: {current_predicted_letter}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    # Display current word
    cv2.putText(frame, f"Word: {current_word}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Show main camera window
    cv2.imshow("Camera", frame)

cap.release()
cv2.destroyAllWindows()
