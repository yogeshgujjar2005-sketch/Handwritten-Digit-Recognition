import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

# -----------------------------
# 1. Load the trained model
# -----------------------------
model_path = "cnn_model.keras"  # Make sure this file exists in the same folder as app.py
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Please check the path!")
else:
    model = load_model(model_path)

    # -----------------------------
    # 2. Streamlit App UI
    # -----------------------------
    st.title("Handwritten Digit Recognition (MNIST CNN)")
    st.write("Upload an image of a handwritten digit (0-9)")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width="stretch")

        # Convert image to numpy array
        img = np.array(image)

        # Preprocessing steps:
        # 1. Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Resize to 28x28
        img = cv2.resize(img, (28, 28))

        # 3. Invert colors (MNIST: white digit on black background)
        img = 255 - img

        # 4. Normalize
        img = img / 255.0

        # 5. Reshape for model input
        img = img.reshape(1, 28, 28, 1)

        # Display processed image (optional)
        st.image(img.reshape(28, 28), caption="Processed Image", width=150)

        # -----------------------------
        # 3. Predict
        # -----------------------------
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # Display results
        st.write(f"### Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence:.2f}")