import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

# Load the optimized TFLite model
interpreter = tf.lite.Interpreter(model_path="emotion_model.tflite")
interpreter.allocate_tensors()

# Get input & output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define emotion labels
CLASS_NAMES = ["Anger", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
IMG_SIZE = 64  # Smaller image size for optimization

# Streamlit UI
st.title("🧠 Autism Emotion Detection (Optimized)")

# Upload Image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Preprocess Image
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray / 255.0
    img_final = np.expand_dims(img_normalized, axis=[0, -1])  # Reshape for model input
    
    # Display Image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run inference with TFLite Model
    interpreter.set_tensor(input_details[0]['index'], img_final.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    emotion_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100  # Confidence score

    # Show Result
    st.markdown(f"### 🎭 Detected Emotion: **{CLASS_NAMES[emotion_index]}**")
    st.markdown(f"### 🔹 Confidence: **{confidence:.2f}%**")

    # Show Probability Bar Chart
    st.bar_chart(prediction[0])
