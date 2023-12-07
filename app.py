# Import necessary libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from skimage import transform
import io
import os
import base64


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to preprocess the image (modify based on your preprocessing requirements)
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img_resized = transform.resize(np.array(img), (225, 225, 3), mode='reflect', anti_aliasing=True)
    processed_image = img_resized.reshape((1, 225, 225, 3))
    return processed_image

# Load the model
def load_model():
    model_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(model_dir, 'my_dense_model.h5')
    model = tf.keras.models.load_model(model_path)
    return model

# Function to make predictions
def predict_image(model, processed_image):
    prediction = model.predict(processed_image)
    return np.round(prediction, 2)

# Function to display prediction results
def display_prediction(prediction):
    tumor_types = {0: "No Tumor", 1: "Benign Tumor", 2: "Malignant Tumor"}
    predicted_class = np.argmax(prediction)
    st.write(f"Prediction: {tumor_types[predicted_class]}")
    for class_index, tumor_type in tumor_types.items():
        st.write(f"{tumor_type}: {prediction[0][class_index] * 100:.2f}%")

# Main Streamlit app
def main():
    col1, col2 = st.columns([1, 1])
    # Column 1: Cancer detection and upload button
    
    placeholder_image = Image.open("placeholder_image.jpg")
    placeholder_base64 = image_to_base64(placeholder_image)
    
    with col1:
        st.title("MammogrAI")
        st.subheader("Cancer detection")
        uploaded_file = st.file_uploader("Choose a mammogram image...", type=["jpg", "png", "jpeg"])

    # Column 2: Display mammography image
    # Column 2: Display mammography image
    with col2:
        if uploaded_file is not None:
            # If uploaded_file is available, display the uploaded image
            image = Image.open(uploaded_file)
        else:
            # If uploaded_file is not available, display the placeholder image
            image = placeholder_image

        image_with_style = f'<img src="data:image/png;base64,{image_to_base64(image)}" class="rounded-image" style="width: 100%; border: 3px solid #2E3135; border-radius: 10px;">'
        st.markdown(image_with_style, unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner("Classifying..."):
            # Display progress bar while the model is loading
            processed_image = preprocess_image(uploaded_file)
            model = load_model()
            prediction = predict_image(model, processed_image)
        st.success("Classification complete!")

        st.subheader("Prediction Results:")
        
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("No Tumor:")
            st.subheader(f"{prediction[0][0] * 100:.0f}%")
        with col2:
            st.write("Benign:")
            st.subheader(f"{prediction[0][1] * 100:.0f}%")
        with col3:
            st.write("Malignant:")
            st.subheader(f"{prediction[0][2] * 100:.0f}%")

# Run the Streamlit app
if __name__ == "__main__":
    main()
