import streamlit as st
from mtcnn import MTCNN
# import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title("Pengenalan Emosi dengan Foto")
st.sidebar.caption("Emotion recognition and face detection using MTCNN")
uploaded_file = st.file_uploader("Pilih foto dari penyimpanan perangkat Anda.", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load model
    model = load_model('deepid_60.keras')

    # Label emosi
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # Read the uploaded file as an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Load the image (BGR format)

    # Convert BGR to RGB for compatibility with Matplotlib and MTCNN
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MTCNN detector
    mtcnn = MTCNN()

    # Detect faces in the image
    result = mtcnn.detect_faces(image_rgb)

    # Prepare a copy of the image for drawing results
    image_with_results = image_rgb.copy()

    for face in result:
        # Get bounding box coordinates
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)
        x2, y2 = x + width, y + height

        # Extract the face region
        face_region = image_rgb[y:y2, x:x2]

        # Resize the face region to match the input shape of the model (e.g., 48x48 or 64x64)
        face_region_resized = cv2.resize(face_region, (39, 31))  # Adjust size as per your model
        face_region_gray = cv2.cvtColor(face_region_resized, cv2.COLOR_RGB2GRAY)
        face_region_normalized = face_region_gray / 255.0  # Normalize pixel values
        face_region_input = np.expand_dims(face_region_normalized, axis=0)
        face_region_input = np.expand_dims(face_region_input, axis=-1)  # Add channel dimension

        # Predict emotion
        prediction = model.predict(face_region_input)
        emotion_index = np.argmax(prediction)
        emotion_label = emotion_labels[emotion_index]

        # Draw the bounding box and label on the image
        cv2.rectangle(image_with_results, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_with_results, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

        # landmark
        keypoints = face['keypoints']
        cv2.circle(image_with_results,(keypoints['left_eye']), 4, (0, 255, 0), -1)
        cv2.circle(image_with_results,(keypoints['right_eye']), 4, (0, 255, 0), -1)
        cv2.circle(image_with_results,(keypoints['nose']), 4, (0, 255, 0), -1)
        cv2.circle(image_with_results,(keypoints['mouth_left']), 4, (0, 255, 0), -1)
        cv2.circle(image_with_results,(keypoints['mouth_right']), 4, (0, 255, 0), -1)

    # Display the results
    st.image(image_with_results, use_container_width=True)
    st.write("Hasil deteksi emosi : ", emotion_label)
   