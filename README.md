# 👐 Real-Time Sign Language Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00BACC?style=for-the-badge)

A  Real-time Computer Vision application designed to bridge the communication gap using American Sign Language (ASL) . This project leverages **MediaPipe** for hand tracking and a **1D Convolutional Neural Network (CNN)** for high-accuracy gesture classification.

---

## 🚀 Features

### 🎥 1. Real-Time Recognition
- **Live Detection**: Uses your webcam to detect hand gestures in real-time.
- **Accurate Prediction**: Classifies ASL Alphabets (A-Z) and Numbers (1-9) with high precision.
- **Sentence Building**: Construct words and sentences dynamically by chaining together predicted signs.
- **Text-to-Speech (TTS)**: Integrated speech synthesis to read out your constructed sentences aloud.

### 📝 2. Text to Sign
- **Reverse Translation**: Type any text, and the application instantly generates the corresponding sign language sequence.
- **Visual Learning**: Excellent tool for learning how to spell words in sign language.

### 🖼️ 3. Image Prediction
- **Static Analysis**: Upload existing images to identify the sign language gesture present.
- **Robust Processing**: Includes advanced preprocessing (padding, multi-mode detection) to handle various image qualities and angles.

---

## 🛠️ Technical Architecture

The system is built on a robust pipeline:
1.  **Input**: Webcam video feed or static images.
2.  **Hand Tracking**: **MediaPipe Hands** detects 21 3D landmarks per hand.
3.  **Feature Extraction**: Landmark coordinates (x, y, z) are normalized relative to the hand's position.
4.  **Classification**: A custom **1D CNN (Convolutional Neural Network)** implemented in PyTorch processes the coordinate data to predict the class (Letter vs Number).
5.  **Interface**: A sleek, dark-themed **Streamlit** UI provides a seamless user experience.

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time features)

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/real-time-sign-language-recognition.git
    cd real-time-sign-language-recognition
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🎯 Usage

To interpret the model correctly, ensure you have the trained model files (`.pth`) in the `Sign Language Recognition/` directory.

**Run the Application:**
```bash
streamlit run app.py
```

### Navigation
Use the **Sidebar** to switch between modes:
- **Model Type**: Toggle between "Alphabet (A-Z)" and "Numbers (1-9)".
- **Go to**: Select the active feature (Real-Time, Text to Sign, Image Prediction).

---

## 📂 Project Structure

```
├── Data/                       # Dataset files (Excel, Images)
├── Sign Language Recognition/  # Model Source Code & Trained Weights
│   ├── CNNModel.py             # PyTorch Model Architecture
│   ├── training.py             # Training script
│   └── *.pth                   # Saved Model Weights
├── app.py                      # Main Streamlit Application
├── requirements.txt            # Python Dependencies
└── README.md                   # Documentation
```

---

## 🤝 Acknowledgments

- **MediaPipe** by Google for the robust hand tracking solution.
- **Streamlit** for the rapid application development framework.
- The open-source community for the datasets and tools used in this project.
