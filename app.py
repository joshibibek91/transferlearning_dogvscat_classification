import streamlit as st



st.title("Cat vs Dog Image Classifier App ")
st.write("This is a simple image classifier app that can classify images of cats and dogs. You can upload an image of a cat or a dog, and the app will predict whether it is a cat or a dog.")  



#Testing the models
import tensorflow
# import keras
import numpy as np
from tensorflow import keras
# from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as img

# creating a object
model = keras.models.load_model('./my_model.keras')
path = st.file_uploader("Upload an image of a cat or a dog", type=["jpg", "jpeg", "png"])


if st.button("Predict"):
    class_labels = {0: "Cat", 1: "Dog"}

    # image_file = Image.open(path)
    # image_file = image_file.resize((224, 224))
    image_file = load_img(path, target_size=(224, 224))
    test_image = tensorflow.keras.preprocessing.image.img_to_array(image_file)/255
    test_image = np.expand_dims(test_image, 0)
    prediction = model.predict(test_image)
    result = np.argmax(prediction)
   
    # result is now 0 (Cat) or 1 (Dog)
    if result == 0:
        output = "Cat"
    else:
        output = "Dog"
    
    st.write(f"The image is predicted to be a {output}.")
    st.image(path, caption='Uploaded Image', width="stretch")
   
