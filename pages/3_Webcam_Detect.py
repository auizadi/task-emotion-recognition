import streamlit as st
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from tensorflow.keras.models import load_model

# Judul aplikasi
st.title("Real-Time Emotion Recognition")
st.sidebar.caption("Real-time emotion recognition and face detection using MTCNN")

# Load model dan label emosi
model = load_model('model_file.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Inisialisasi MTCNN
detector = MTCNN()

# Fungsi untuk memproses wajah
def preprocess_face(face, target_size=(48, 48)):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, target_size)
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

# Opsi untuk menggunakan webcam
enable_camera = st.checkbox("Aktifkan Kamera")

if enable_camera:
    # Menggunakan OpenCV untuk akses webcam
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Tidak dapat mengakses kamera.")
            break

        # Deteksi wajah menggunakan MTCNN
        result = detector.detect_faces(frame)
        if result:
            for person in result:
                bounding_box = person['box']
                keypoints = person['keypoints']

                x, y, width, height = bounding_box
                x, y = max(0, x), max(0, y)

                face = frame[y:y+height, x:x+width]

                try:
                    # Preprocessing wajah
                    preprocessed_face = preprocess_face(face)
                    prediction = model.predict(preprocessed_face)
                    emotion_index = np.argmax(prediction)
                    emotion = emotion_labels[emotion_index]

                    # Menampilkan label emosi di atas kotak bounding
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Menampilkan keypoints
                    cv2.circle(frame, keypoints['left_eye'], 4, (0, 255, 0), -1)
                    cv2.circle(frame, keypoints['right_eye'], 4, (0, 255, 0), -1)
                    cv2.circle(frame, keypoints['nose'], 4, (0, 255, 0), -1)
                    cv2.circle(frame, keypoints['mouth_left'], 4, (0, 255, 0), -1)
                    cv2.circle(frame, keypoints['mouth_right'], 4, (0, 255, 0), -1)
                except Exception as e:
                    print(f"Error processing face: {e}")

        # Konversi frame ke RGB untuk Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    # Lepaskan webcam setelah loop selesai
    cap.release()
    cv2.destroyAllWindows()