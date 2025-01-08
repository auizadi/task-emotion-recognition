import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from mtcnn import MTCNN


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model_fer2013.keras")  
    return model


class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def preprocess_image(image):
    image = cv2.resize(image, ( 47,55))  
    image = image / 255.0  
    if image.shape[-1] == 1:  
        image = np.repeat(image, 3, axis=-1)
    image = np.expand_dims(image, axis=0)  
    return image


def real_time_emotion_detection():
    st.title("Real-Time Emotion Detection")

    st.write("Starting webcam...")
    run = st.checkbox("Run Webcam")
    
    if run:

        model = load_model()
        mtcnn = MTCNN()

        cap = cv2.VideoCapture(0)

        frame_placeholder = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Gagal untuk menangkap video.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detections = mtcnn.detect_faces(rgb_frame)

            for detection in detections:
                x, y, width, height = detection['box']
                face = rgb_frame[max(0, y):y + height, max(0, x):x + width]  # Ensur indices are valid

                if face.size > 0:
                    processed_face = preprocess_image(face)
                    predictions = model.predict(processed_face)
                    predicted_label = class_labels[np.argmax(predictions)]

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            frame_placeholder.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_emotion_detection()
