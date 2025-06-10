import os
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Set page configuration for a cleaner look
st.set_page_config(layout="wide", page_title="Anomaly Detection System")

# --- MODEL LOADING ---
# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def load_keras_model():
    """Loads the Keras model and labels from the disk."""
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

# --- PREDICTION FUNCTION ---
def predict(image_to_predict, model, labels):
    """
    Takes a PIL image and returns the predicted class and confidence score.
    """
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resize and crop the image to 224x224
    size = (224, 224)
    image = image_to_predict.resize(size)

    # Convert image to numpy array and normalize it
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

# --- STREAMLIT INTERFACE ---
st.title("Anomaly Detection for Manufactured Products 🏭")
st.write("Upload an image of a product (e.g., a bottle cap) to check if it's normal or an anomaly.")

# Load the model and labels
model, labels = load_keras_model()

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Your Image")
        st.image(image, use_column_width=True)

    with col2:
        st.header("Prediction")
        # Make a prediction when the button is clicked
        with st.spinner('Analyzing the image...'):
            class_name, confidence_score = predict(image, model, labels)

        # Display the result
        if class_name.lower() == "anomaly":
            st.error(f"Status: {class_name}")
        else:
            st.success(f"Status: {class_name}")

        st.write(f"**Confidence:** {confidence_score:.2%}")
