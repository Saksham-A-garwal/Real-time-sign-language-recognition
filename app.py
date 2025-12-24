import streamlit as st
import cv2
import mediapipe as mp
import torch
import pandas as pd
import numpy as np
import sys
import os
from PIL import Image
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
import threading

# Add the directory containing CNNModel.py to the system path
sys.path.append(os.path.join(os.getcwd(), 'Sign Language Recognition'))

try:
    from CNNModel import CNNModel
except ImportError:
    st.error("Could not import CNNModel. Please make sure 'Sign Language Recognition/CNNModel.py' exists.")
    st.stop()

# Page Config
st.set_page_config(
    page_title="ASL Recognition Pro",
    page_icon="👐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Styling
st.markdown("""
    <style>
    /* Main Background and Text */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #00d4ff;
        color: #000000;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00a3cc;
        color: #ffffff;
        transform: scale(1.02);
    }
    
    /* Cards/Containers */
    .css-1r6slb0 {
        background-color: #1f2129;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Success/Info Messages */
    .stSuccess, .stInfo {
        background-color: #1f2129 !important;
        color: #fafafa !important;
        border-left: 5px solid #00d4ff;
    }
    
    /* Image Captions */
    .stImage > div > div > div {
        color: #cccccc;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Sidebar Navigation
st.sidebar.title("Navigation")

# Model Selection
model_type = st.sidebar.radio("Model Type", ["Alphabet (A-Z)", "Numbers (1-9)"])

# Load Model
@st.cache_resource
def load_model(path):
    try:
        # Always try to load as 26 classes first, as training.py defaults to 26
        model = CNNModel(num_classes=26)
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                model.eval()
                return model
            except RuntimeError:
                # If that fails, try 10 classes (in case it was trained correctly elsewhere)
                try:
                    model = CNNModel(num_classes=10)
                    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                    model.eval()
                    return model
                except RuntimeError:
                     # Try 9 classes
                    try:
                        model = CNNModel(num_classes=9)
                        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                        model.eval()
                        return model
                    except Exception as e:
                         st.error(f"Model architecture mismatch: {e}")
                         return None
        else:
            st.error(f"Model file not found at {path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define model path based on selection
if model_type == "Alphabet (A-Z)":
    model_path = os.path.join('Sign Language Recognition', 'CNN_model_alphabet_SIBI.pth')
else:
    model_path = os.path.join('Sign Language Recognition', 'CNN_model_number_SIBI.pth')

model = load_model(model_path)

# Define Classes based on selection
if model_type == "Alphabet (A-Z)":
    classes = {
        'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'J': 9,
        'K': 10,'L': 11,'M': 12,'N': 13,'O': 14,'P': 15,'Q': 16,'R': 17,
        'S': 18,'T': 19,'U': 20,'V': 21,'W': 22,'X': 23,'Y': 24,'Z': 25
    }
else:
    # Numbers 1-9 (Index 0 -> '1')
    # User reported "1 less" error when 0 was included.
    # This implies the model predicts Index 0 for '1'.
    classes = {
        '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8
    }

reverse_classes = {v: k for k, v in classes.items()}

app_mode = st.sidebar.radio("Go to", ["Real-Time Recognition", "Text to Sign", "Image Prediction"])
def predict_frame(frame, hands_detector):
    h, w, _ = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frameRGB)
    
    predicted_letter = None
    annotated_frame = frame.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            x_list = []
            y_list = []
            z_list = []
            
            for lm in hand_landmarks.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)
                z_list.append(lm.z)
                
            # Normalize
            data = {}
            for i, lm in enumerate(hand_landmarks.landmark):
                data[f"LM_{i}_x"] = lm.x - min(x_list)
                data[f"LM_{i}_y"] = lm.y - min(y_list)
                data[f"LM_{i}_z"] = lm.z - min(z_list)
                
            df = pd.DataFrame([data])
            coords = np.reshape(df.values, (df.shape[0], 63, 1))
            coords = torch.from_numpy(coords).float()
            
            # Predict
            with torch.no_grad():
                out = model(coords)
                
                # If using Number model but it has 26 outputs, slice it to first 10
                if model_type == "Numbers (1-9)" and out.shape[1] == 26:
                    out = out[:, :10]
                
                _, pred = torch.max(out.data, 1)
                
                # Handle case where prediction index is out of range for our current class map
                pred_idx = pred.item()
                if pred_idx in reverse_classes:
                    predicted_letter = reverse_classes[pred_idx]
                else:
                    predicted_letter = "?"
            
            # Draw Bounding Box
            x1 = int(min(x_list) * w) - 10
            y1 = int(min(y_list) * h) - 10
            x2 = int(max(x_list) * w) + 10
            y2 = int(max(y_list) * h) + 10
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 212, 255), 3)
            cv2.putText(annotated_frame, predicted_letter, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 212, 255), 3)
                        
    return annotated_frame, predicted_letter

# Cleanup: Release camera if not in Real-Time mode
if app_mode != "Real-Time Recognition":
    if 'cap' in st.session_state:
        st.session_state.cap.release()
        del st.session_state.cap

# --------------------------------------------------------------------------------
# Feature 1: Real-Time Recognition
# --------------------------------------------------------------------------------
if app_mode == "Real-Time Recognition":
    st.title("🎥 Real-Time ASL Recognition")
    st.markdown("Use your webcam to predict ASL signs in real-time and build sentences.")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Controls")
        run = st.checkbox('Start Camera', value=False)
        
        if 'current_word' not in st.session_state:
            st.session_state.current_word = ""
        if 'sentence' not in st.session_state:
            st.session_state.sentence = ""
        if 'last_pred' not in st.session_state:
            st.session_state.last_pred = None
            
        # Control Buttons
        if st.button("Add Letter (C)"):
            if st.session_state.last_pred:
                st.session_state.current_word += st.session_state.last_pred
                
        if st.button("Add Word (Space)"):
            if st.session_state.current_word:
                st.session_state.sentence += st.session_state.current_word + " "
                st.session_state.current_word = ""
                
        if st.button("Backspace"):
            st.session_state.current_word = st.session_state.current_word[:-1]
            
        if st.button("Clear Sentence"):
            st.session_state.sentence = ""
            st.session_state.current_word = ""

        # Text-to-Speech Button
        if TTS_AVAILABLE:
            if st.button("Speak Sentence 🗣️"):
                if st.session_state.sentence.strip():
                    def speak_text(text):
                        try:
                            engine = pyttsx3.init()
                            engine.say(text)
                            engine.runAndWait()
                        except Exception as e:
                            st.error(f"Error in TTS: {e}")
                    
                    # Run in a separate thread to avoid blocking the UI too much
                    threading.Thread(target=speak_text, args=(st.session_state.sentence,), daemon=True).start()
                else:
                    st.warning("Sentence is empty!")
        else:
            st.warning("Text-to-Speech library not found. Please run `pip install pyttsx3` in your terminal.")

        st.markdown("---")
        st.markdown("### Current Status")
        st.info(f"**Current Word:** {st.session_state.current_word}")
        st.success(f"**Sentence:** {st.session_state.sentence}")

    with col1:
        FRAME_WINDOW = st.image([])
        
        if run:
            if 'cap' not in st.session_state or not st.session_state.cap.isOpened():
                st.session_state.cap = cv2.VideoCapture(0)
            
            cap = st.session_state.cap
            
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to grab frame")
                    break
                
                frame = cv2.flip(frame, 1)
                annotated_frame, predicted_letter = predict_frame(frame, hands)
                
                if predicted_letter:
                    st.session_state.last_pred = predicted_letter
                
                # Display
                FRAME_WINDOW.image(annotated_frame, channels="BGR")
                
        else:
            if 'cap' in st.session_state:
                st.session_state.cap.release()
                del st.session_state.cap
            st.write("Camera is stopped.")

# --------------------------------------------------------------------------------
# Feature 2: Text to Sign
# --------------------------------------------------------------------------------
elif app_mode == "Text to Sign":
    st.title("📝 Text to Sign Language")
    st.markdown("Enter text below to convert it into ASL sign language images.")
    
    text_input = st.text_input("Enter Text:", "").lower()
    
    if text_input:
        st.subheader("Sign Language Output")
        
        # Filter only valid characters (a-z and 0-9)
        valid_chars = [c for c in text_input if c.isalnum()]
        
        if valid_chars:
            # Display images in a grid
            cols = st.columns(min(len(valid_chars), 6))
            
            for i, char in enumerate(valid_chars):
                # Determine image path
                # Assuming dataset structure: Data/asl_dataset/a/hand1_a_bot_seg_1_cropped.jpeg
                # We need to pick one representative image for each character
                
                char_dir = os.path.join("Data", "asl_dataset", char)
                if os.path.exists(char_dir):
                    images = os.listdir(char_dir)
                    if images:
                        img_path = os.path.join(char_dir, images[0]) # Pick the first image
                        
                        # Display in grid (wrapping)
                        col_idx = i % 6
                        if col_idx == 0 and i > 0:
                            cols = st.columns(min(len(valid_chars) - i, 6))
                        
                        with cols[col_idx]:
                            st.image(img_path, caption=char.upper(), use_container_width=True)
                    else:
                        with cols[i % 6]:
                            st.warning(f"No image for '{char}'")
                else:
                     # Handle numbers or missing folders if any
                    with cols[i % 6]:
                        st.warning(f"'{char}' not found")
        else:
            st.info("Please enter alphanumeric characters.")

# --------------------------------------------------------------------------------
# Feature 3: Image Prediction
# --------------------------------------------------------------------------------
elif app_mode == "Image Prediction":
    st.title("🖼️ Image Prediction")
    st.markdown("Upload an image of a hand sign to predict the alphabet.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button("Predict"):
            with st.spinner('Processing...'):
                # Convert PIL image to cv2
                img_array = np.array(image)
                if img_array.shape[-1] == 4: # RGBA to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                # If image is RGB from PIL, cv2 expects BGR usually, but our predict_frame converts BGR to RGB.
                # So if we pass RGB, we should convert to BGR first so predict_frame can convert it back to RGB?
                # Or just modify predict_frame to handle it. 
                # Let's just convert to BGR to match video frame format.
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Attempt 1: Standard Static Mode
                hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
                annotated_image, prediction = predict_frame(img_bgr, hands)
                hands.close()
                
                # Attempt 2: Stream Mode (sometimes works better for certain angles/lighting)
                if not prediction:
                    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
                    annotated_image, prediction = predict_frame(img_bgr, hands)
                    hands.close()
                
                # Attempt 3: Add Padding (helps if hand is too close to edge)
                if not prediction:
                    h, w, _ = img_bgr.shape
                    pad = int(max(h, w) * 0.2) # 20% padding
                    img_padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
                    
                    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
                    annotated_image_pad, prediction_pad = predict_frame(img_padded, hands)
                    hands.close()
                    
                    if prediction_pad:
                        annotated_image = annotated_image_pad
                        prediction = prediction_pad
                
                # Convert back to RGB for display
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_image_rgb, caption='Processed Image', use_container_width=True)
                
                if prediction:
                    st.success(f"### Predicted Letter: **{prediction}**")
                else:
                    st.error("No hand detected. Please try an image with better lighting or the hand fully visible.")

# Footer
st.markdown("---")
#st.markdown("<div style='text-align: center; color: #666;'>Built with ❤️ using Streamlit, MediaPipe, and PyTorch</div>", unsafe_allow_html=True)
