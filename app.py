import streamlit as st
from PIL import Image
import numpy as np
import keras # We now need to import keras directly

# NOTE: The os.environ line is no longer needed as we are using the new Keras 3 loading method.

# Set page configuration
st.set_page_config(layout="wide", page_title="Anomaly Detection System")

# --- MODEL LOADING (MODIFIED) ---
# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def load_keras_model():
    """
    Loads the Keras model and labels.
    The model is loaded as a TFSMLayer, which is the Keras 3 way for SavedModels.
    """
    labels_path = "labels.txt"
    model_path = "my_model"

    # Load the labels
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # --- THIS IS THE KEY CHANGE for loading the model ---
    # We load the SavedModel as a special Keras Layer
    model_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

    # We return the layer itself, not a full model object
    return model_layer, labels

# --- PREDICTION FUNCTION (MODIFIED) ---
def predict(image_to_predict, model_layer, labels):
    """
    Takes a PIL image and a TFSMLayer, and returns the predicted class and confidence score.
    """
    # Create the array of the right shape to feed into the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Resize and crop the image to 224x224
    size = (224, 224)
    image = image_to_predict.resize(size)
    
    # Convert image to numpy array and normalize it
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # --- THIS IS THE KEY CHANGE for making a prediction ---
    # A TFSMLayer is called like a function, not with .predict()
    # The output is a dictionary, so we get the tensor from its values.
    prediction_output = model_layer(data)
    prediction_tensor = list(prediction_output.values())[0]
    
    # Get the prediction array
    prediction = prediction_tensor.numpy()[0]

    # Find the index of the highest probability
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = prediction[index]
    
    return class_name, confidence_score

# --- STREAMLIT INTERFACE (No changes needed here) ---
st.title("Anomaly Detection for Manufactured Products 🏭")
st.write("Upload an image of a product (e.g., a bottle cap) to check if it's normal or an anomaly.")

# Load the model and labels
model_layer, labels = load_keras_model()

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
        # Make a prediction
        with st.spinner('Analyzing the image...'):
            class_name, confidence_score = predict(image, model_layer, labels)
        
        # Display the result
        if class_name.lower() == "anomaly":
            st.error(f"Status: {class_name}")
        else:
            st.success(f"Status: {class_name}")

        st.write(f"**Confidence:** {confidence_score:.2%}")
