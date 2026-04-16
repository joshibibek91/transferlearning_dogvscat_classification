import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image

# App title and description
st.title("Cat vs Dog Image Classifier App")
st.write(
    "Upload an image of a cat or a dog, and the model will predict the class."
)

# Load model
model = keras.models.load_model("./my_model.keras")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

# Prediction
if uploaded_file is not None and st.button("Predict"):
    # Class labels
    class_labels = {0: "Cat", 1: "Dog"}

    # Load and preprocess image
    image = load_img(uploaded_file, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    result = np.argmax(prediction)

    # Output
    output = class_labels[result]

    # Display result
    st.write(f"Prediction: {output}")
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)