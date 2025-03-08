import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load Model Efficiently
@st.cache_resource
def load_trained_model():
    model_path = "emotion_model.h5"
    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found! Please ensure the path is correct.")
        st.stop()
    return load_model(model_path)

model = load_trained_model()

# Define Class Labels
class_labels = ["Anger", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]

# Streamlit UI
st.title("😊 Emotion Recognition")
st.write("Upload an image to recognize the **Emotion**.")

# File Uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

# Image Preprocessing
def preprocess_image(image):
    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize Image
    img_resized = cv2.resize(gray, (64, 64))
    
    # Normalize
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    return gray, img_array

# Image Processing & Prediction
if uploaded_file:
    try:
        # Load Image
        image = Image.open(uploaded_file).convert("RGB")

        # Preprocess Image
        gray_image, img_array = preprocess_image(image)
        
        # Predict Emotion
        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_label = class_labels[predicted_index]
        confidence = predictions[predicted_index] * 100  # Convert to %

        # Layout Columns
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.image(image, caption="📷 Original Image", use_column_width=True)

        with col2:
            st.image(gray_image, caption="📸 Grayscale Image", use_column_width=True, channels="GRAY")

        with col3:
            st.subheader(f"🎭 Emotion: **{predicted_label}**")
            st.write(f"🧠 Confidence: **{confidence:.2f}%**")

        # 📊 Plot Probabilities
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=class_labels, y=predictions, palette="coolwarm", ax=ax)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Emotions")
        ax.set_title("📊 Confidence Levels for Each Emotion")
        ax.set_xticklabels(class_labels, rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")

st.write("👆 Upload an image to analyze emotions.")
