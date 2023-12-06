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

    # Include the Montserrat link in the head section
    head = """
        <head>
            <link rel="stylesheet" type="text/css" href="https://fonts.googleapis.com/css?family=Montserrat">
        </head>
    """
    st.markdown(head, unsafe_allow_html=True)

    # Set font style directly in Streamlit
    font_style = """
        <style>
            /* Add the Montserrat font directly to the body */
            body {
                font-family: 'Montserrat', sans-serif;
            }
        </style>
    """
    st.markdown(font_style, unsafe_allow_html=True)

    # Set app title and header
    st.title("MammogrAI")
    st.header("Cancer Detection")
    st.write("This is the final project completed by three Le Wagon students after a nine-week data science bootcamp.")


    # About button to go to the "About the Team" page
    if st.button("About"):
        st.write("About the Team page content goes here.")

    # File uploader for mammogram images
    uploaded_file = st.file_uploader("Choose a mammogram image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display top right: Uploaded Image with stroke and rounded edges
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

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
        display_prediction(prediction)
    else:
        st.warning("Please upload a valid image file.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
