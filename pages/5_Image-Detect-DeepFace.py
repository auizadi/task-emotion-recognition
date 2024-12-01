import streamlit as st  
import cv2  
from deepface import DeepFace   
import numpy as np  


st.title("Deteksi Emosi Menggunakan DeepFace")  

# Upload a single image file  
upload = st.file_uploader(label='Upload foto', type=['png', 'jpg', 'jpeg'])  

if upload is not None:  
    # Read the uploaded image  
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)  
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Load the image (BGR format)  

    if image is not None:  
        # Analyze emotions  
        results = DeepFace.analyze(image, actions=['emotion'], detector_backend='mtcnn')  

        # Get the dominant emotion and its percentage  
        emotion = results[0]['dominant_emotion']  
        emotion_percentage = results[0]['emotion'][emotion]  

        # Convert BGR to RGB for display  
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

        # Display the image  
        st.image(image_rgb, caption='Uploaded Image', use_container_width=True)  

        # Display the result  
        st.write(f'Dominant Emotion: {emotion} ({emotion_percentage:.2f}%)', font_size=24)  

    else:  
        st.error("Gambar tidak dapat dibaca.")  
else:  
    st.info("Silakan unggah gambar untuk mendeteksi emosi.")