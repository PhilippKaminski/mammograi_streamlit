# Import necessary libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from skimage import transform
import matplotlib.image as mpimg
import io

# Function to preprocess the image (modify based on your preprocessing requirements)
def preprocess_image(uploaded_file):
    # Convert the Image object to a byte stream
    img_byte_array = io.BytesIO()
    uploaded_file.save(img_byte_array, format="PNG")

    # Open the image directly from the byte stream
    img = Image.open(io.BytesIO(img_byte_array.getvalue()))

    # Resize the image
    img_resized = transform.resize(np.array(img), (225, 225, 3), mode='reflect', anti_aliasing=True)

    # Reshape the image to match the model input shape
    processed_image = img_resized.reshape((1, 225, 225, 3))

    return processed_image



def load_model():
    model_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(model_dir, 'my_model')
    # Print the absolute path for debugging

    model = tf.keras.models.load_model(model_path)
    return model


# Main Streamlit app
def main():
    # Set app title
    st.title("Mammography Model Web App")

    # File uploader for mammogram images
    uploaded_file = st.file_uploader("Choose a mammogram image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Load the pre-trained model
        model = load_model()

        # Make predictions
        prediction = model.predict(processed_image)
        prediction = np.round(prediction, 2)
        if prediction[0][0] > prediction[0][1] and prediction[0][0] > prediction[0][2]:
            st.write(f"Prediction: you have a {np.round(prediction[0][0], 2)*100}% chance that the image contains a benign tumor")
            st.write(f"A malignant tumor has a chance of {np.round(prediction[0][1], 2)*100}% and there is a {np.round(prediction[0][2], 2)*100}% chance of no tumor")
        elif prediction[0][1] > prediction[0][0] and prediction[0][1] > prediction[0][2]:
            st.write(f"Prediction: you have a {np.round(prediction[0][1], 2)*100}% chance that the image contains a malignant tumor")
            st.write(f"A benign tumor has a chance of {np.round(prediction[0][0], 2)*100}% and there is a {np.round(prediction[0][2], 2)*100}% chance of no tumor")
        elif prediction[0][2] > prediction[0][1] and prediction[0][2] > prediction[0][0]:
            st.write(f"Prediction: you have a {np.round(prediction[0][2], 2)*100}% chance that the image contains no tumor")
            st.write(f"A malignant tumor has a chance of {np.round(prediction[0][0], 2)*100}% and the malignant tumor has a chance of {np.round(prediction[0][1], 2)*100}%")
        else:
            st.write(f"Prediction: no tumor:{np.round(prediction[0][2], 2)*100}%; benign: {np.round(prediction[0][0], 2)*100}%; malignant: {np.round(prediction[0][1], 2)*100}%" )
    else:
        return "file = None"

# Run the Streamlit app
if __name__ == "__main__":
    main()
