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
        
        
def display_consultation(prediction):
    tumor_types = {0: "Normal", 1: "Benign", 2: "Malignant"}
    predicted_class = np.argmax(prediction)
    
    if predicted_class == 0:
        st.subheader("Result: Normal (no evidence of breast cancer detected)")
        st.write("It's important to note that AI classifications may have a margin of error, and regular mammograms, as recommended by healthcare professionals, are still advised. Consult with your healthcare provider for personalized guidance on screening intervals, typically recommended annually for women aged 40 and older.")
    elif predicted_class == 1:
        st.subheader("Result: Benign (no signs of malignant features observed)")
        st.write("While the AI assessment suggests a benign condition, it's essential to acknowledge the possibility of false positives. Options to proceed include:")
        st.write("1. **Consultation with a Specialist:** Schedule an appointment with a breast health specialist for a thorough clinical evaluation.")
        st.write("2. **Follow-up Imaging:** Additional imaging studies, such as ultrasound or MRI, may be recommended to further assess the nature of the identified benign features.")
        st.write("3. **Biopsy if Necessary:** If uncertainties persist, a biopsy might be suggested to obtain a definitive diagnosis.")
        st.write("Always consult with your healthcare provider to determine the most appropriate course of action based on your individual circumstances.")
    elif predicted_class == 2:
        st.subheader("Result: Malignant (indications of potential breast cancer detected)")
        st.write("It's crucial to understand that AI results may have false negatives, and confirmation by medical professionals is essential. Options to consider following a malignant diagnosis:")
        st.write("1. **Consultation with Oncologist:** Schedule an immediate consultation with an oncologist to discuss the diagnosis and develop a personalized treatment plan.")
        st.write("2. **Biopsy Confirmation:** Confirm the malignancy through a biopsy, which provides detailed information about the type and stage of the cancer.")
        st.write("3. **Treatment Planning:** Collaborate with healthcare professionals to formulate a comprehensive treatment strategy, including surgery, chemotherapy, radiation, or a combination, depending on the specifics of the case.")
        st.write("Prompt and comprehensive medical intervention is crucial; consult with your healthcare team for guidance tailored to your specific situation.")


# Main Streamlit app
def main():
    
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden; }
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    
    
    logo = Image.open("logo.png")
    logo_base64 = image_to_base64(logo)
    st.set_page_config(page_title="MammogrAI", page_icon=logo_base64)
    
    placeholder_image = Image.open("placeholder_image.jpg")
    placeholder_base64 = image_to_base64(placeholder_image)
    
    col1, col2 = st.columns([1, 1])
    # Column 1: Cancer detection and upload button
    
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

        predicted_class = np.argmax(prediction)
        tumor_types = {0: "No Tumor", 1: "Benign Tumor", 2: "Malignant Tumor"}
        st.subheader(f"Prediction Result: {tumor_types[predicted_class]}")

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
            
            
        if st.button("Consultation"):
            with st.spinner("Generating consultation information..."):
                display_consultation(prediction)


# Run the Streamlit app
if __name__ == "__main__":
    main()
