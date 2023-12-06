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

    # Set app title and header
    st.title("MammogrAI")
    st.subheader("Cancer Detection")
    st.write("This is the final project completed by three Le Wagon students after a nine-week data science bootcamp.")

    # File uploader for mammogram images
    uploaded_file = st.file_uploader("Choose a mammogram image...", type=["jpg", "png", "jpeg"])

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    # Column 1: Cancer detection information
    with col1:
        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), caption="Uploaded Image.", use_column_width=True)
        else:
            st.warning("Please upload a valid image file.")

    # Column 2: Display mammography image
    with col2:
        if uploaded_file is not None:
            processed_image = preprocess_image(uploaded_file)
            model = load_model()

            with st.spinner("Classifying..."):
                prediction = predict_image(model, processed_image)

            st.success("Classification complete!")
            st.subheader("Prediction Results")
            
            # Display chance for normal, benign, malignant
            st.write("Chance for Normal:")
            st.write(f"{prediction[0][0] * 100:.2f}%")

            st.write("Chance for Benign:")
            st.write(f"{prediction[0][1] * 100:.2f}%")

            st.write("Chance for Malignant:")
            st.write(f"{prediction[0][2] * 100:.2f}%")

# Run the Streamlit app
if __name__ == "__main__":
    main()
