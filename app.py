# Import necessary libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from skimage import transform
import io
import os

# Function to preprocess the image (modify based on your preprocessing requirements)
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img_resized = transform.resize(np.array(img), (225, 225, 3), mode='reflect', anti_aliasing=True)
    processed_image = img_resized.reshape((1, 225, 225, 3))
    return processed_image

# load the model
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

    # Create a layout with three columns
    col1, col2 = st.columns([1, 1])

    # Column 1: Cancer detection and upload button
    with col1:
        # Set app title and header
        st.title("MammogrAI")
        st.subheader("Cancer detection")
        # File uploader for mammogram images
        uploaded_file = st.file_uploader("Choose a mammogram image...", type=["jpg", "png", "jpeg"])

    # Column 2: Display mammography image
    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
         
    

            
    if uploaded_file is not None:
        # Preprocess the image
        processed_image = preprocess_image(uploaded_file)

        # Load the pre-trained model
        model = load_model()

        # Make predictions
        with st.spinner("Classifying..."):
            # Display progress bar while the model is loading
            prediction = predict_image(model, processed_image)

        # Remove progress bar and display prediction results
        st.success("Classification complete!")

        # Display prediction results in columns
        st.subheader("Prediction Results")
        
        
        with open ("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
        
        col1, col2, col3 = st.columns(3)

        # Column 1: Chance for normal
        with col1:
            st.write("No Tumor:")
            st.subheader(f"{prediction[0][0] * 100:.0f}%")

        # Column 2: Chance for benign
        with col2:
            st.write("Benign:")
            st.subheader(f"{prediction[0][1] * 100:.0f}%")

        # Column 3: Chance for malignant
        with col3:
            st.write("Malignant:")
            st.subheader(f"{prediction[0][2] * 100:.0f}%")



# Run the Streamlit app
if __name__ == "__main__":
    main()
