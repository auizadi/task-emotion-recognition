import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw
from mtcnn import MTCNN

# Load pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model_fer2013.keras")  # Replace with your model file path
    return model

# Define emotion labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Preprocess the image to match model input
def preprocess_image(image):
    image = cv2.resize(image, ( 47,55))  # Resize to the target size
    image = image / 255.0  # Normalize pixel values
    if image.shape[-1] == 1:  # If grayscale, convert to RGB
        image = np.repeat(image, 3, axis=-1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Real-time emotion detection function
def real_time_emotion_detection():
    st.title("Real-Time Emotion Detection")

    # Initialize webcam
    st.write("Starting webcam...")
    run = st.checkbox("Run Webcam")
    
    if run:
        # Load model
        model = load_model()
        mtcnn = MTCNN()

                # Start video capture
        cap = cv2.VideoCapture(0)

        # Create a placeholder for the video frame
        frame_placeholder = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture video. Please check your webcam.")
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            detections = mtcnn.detect_faces(rgb_frame)

            # Draw bounding boxes and predict emotion
            for detection in detections:
                x, y, width, height = detection['box']
                face = rgb_frame[max(0, y):y + height, max(0, x):x + width]  # Ensure indices are valid

                # Preprocess the detected face
                if face.size > 0:
                    processed_face = preprocess_image(face)
                    predictions = model.predict(processed_face)
                    predicted_label = class_labels[np.argmax(predictions)]

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Update the frame in the placeholder
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)

        # Release webcam
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_emotion_detection()
