# 😃 Real-Time Emotion Detection Using MobileNetV2

This project detects human emotions in real time using a webcam. It uses **OpenCV** for face detection and a **MobileNetV2**-based deep learning model (trained using transfer learning) for emotion classification.

---

## 🔍 Features

- Real-time face detection using Haar cascades
- Emotion prediction using MobileNetV2
- 7 supported emotions:  
  `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`
- Works with standard webcam
- Lightweight and efficient

---

## 🧠 Emotion Classes

| Label     | Index |
|-----------|-------|
| Angry     | 0     |
| Disgust   | 1     |
| Fear      | 2     |
| Happy     | 3     |
| Neutral   | 4     |
| Sad       | 5     |
| Surprise  | 6     |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 🧠 Model File

```bash
EmotionModel.h5
```

## ▶️ Usage

```bash
python detect_emotion.py
```
The webcam will open and start detecting emotions in real-time.

Press q to quit.

## 🗂️ Project Structure

```bash
emotion-detection/
├── main.py        # Real-time detection script
├── EmotionModel.h5          # Trained Keras model
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
└── dataPreprocessing.ipynb  # Making Model process
```

## 💡 How It Works
Captures live video using OpenCV.

Detects faces using Haar Cascades.

Crops and preprocesses each face to 224x224 RGB format.

Predicts emotion using the MobileNetV2-based model.

Displays the predicted emotion label on the frame.

## 📜 License
This project is licensed under the MIT License.
Feel free to fork and improve it! 🔥

## 🙋‍♂️ Author
Divyansu Giri
