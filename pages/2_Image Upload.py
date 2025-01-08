import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from mtcnn import MTCNN

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model_fer2013.keras")  
    return model

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_image(image):
    image = image.resize(( 47,55))  
    image = np.array(image) / 255.0  
    if image.shape[-1] == 1: 
        image = np.repeat(image, 3, axis=-1)
    image = np.expand_dims(image, axis=0)  
    return image

# Streamlit app
def main():
    st.title("Emotion Detection App")
    st.sidebar.caption("Emotion recognition and face detection using MTCNN")
    st.write("Upload an image, and the app will predict the emotion.")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
 
        image = Image.open(uploaded_file)

        mtcnn = MTCNN()

        # Detect faces
        image_np = np.array(image)
        detections = mtcnn.detect_faces(image_np)

        draw = ImageDraw.Draw(image)
        for detection in detections:
            x, y, width, height = detection['box']
            draw.rectangle([x, y, x + width, y + height], outline="red", width=2)

            for key, point in detection['keypoints'].items():
                draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill="red")

        st.image(image, caption="Detected Faces with Landmarks", use_container_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Load model and make prediction
        model = load_model()
        predictions = model.predict(processed_image)

        # Get the predicted emotion
        predicted_label = class_labels[np.argmax(predictions)]

        st.write(f"Predicted Emotion: **{predicted_label}**")

if __name__ == "__main__":
    main()
