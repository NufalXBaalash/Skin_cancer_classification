import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import gdown
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

# App title and description
st.title("🔬 Skin Cancer Classification Tool")
st.markdown("""
This application helps classify skin lesion images into 7 different categories:
- **Akiec**: Actinic Keratosis / Intraepithelial Carcinoma 🔍
- **BCC**: Basal Cell Carcinoma 🔬
- **BKL**: Benign Keratosis 👁️
- **DF**: Dermatofibroma 🔎
- **MEL**: Melanoma ⚠️
- **NV**: Melanocytic Nevus 🧿
- **Vasc**: Vascular Lesion 🩸
""")

st.warning("""
⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult with a qualified healthcare professional for proper diagnosis and treatment of skin conditions.
""")

# Sidebar customization
st.sidebar.header("💡 About This Tool")
st.sidebar.info("This app uses AI to classify skin lesions into 7 different categories based on visual patterns. Upload an image to get started!")
confidence_threshold = st.sidebar.slider("🎯 Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

# Fixed image dimensions
img_height = 150
img_width = 150

# Class definitions with descriptions and emojis
class_info = {
    "Akiec": "🔍 Actinic Keratosis / Intraepithelial Carcinoma - A pre-cancerous growth caused by sun damage",
    "BCC": "🔬 Basal Cell Carcinoma - The most common type of skin cancer, usually appears on sun-exposed areas",
    "BKL": "👁️ Benign Keratosis - A non-cancerous growth that may look concerning but is harmless",
    "DF": "🔎 Dermatofibroma - A common benign skin growth that is usually firm and brownish",
    "MEL": "⚠️ Melanoma - A serious form of skin cancer that develops from pigment-producing cells",
    "NV": "🧿 Melanocytic Nevus - A common mole, usually harmless",
    "Vasc": "🩸 Vascular Lesion - A growth or abnormality of blood vessels in the skin"
}

classes = ["Akiec", "BCC", "BKL", "DF", "MEL", "NV", "Vasc"]

@st.cache_resource
def load_keras_model():
    """Load the Keras model once and cache it."""
    model_path = "model2.h5"
    if not os.path.exists(model_path):
        # Corrected Google Drive file ID and URL format
        file_id = "1mm_o84dvdeQ20mAssJ4K3grlO8pdrsfW"
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None

    if os.path.getsize(model_path) < 1_000_000:
        st.error("❌ Model file appears incomplete or corrupted. Please check the download.")
        return None

    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading the model: {e}")
        return None

def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, img_array):
    predictions = model.predict(img_array)
    return predictions[0]

def display_prediction(predictions, classes, threshold):
    predicted_class_idx = np.argmax(predictions)
    predicted_class = classes[predicted_class_idx]
    confidence = predictions[predicted_class_idx]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 Prediction Results")
        if confidence >= threshold:
            st.success(f"**Predicted Class:** {predicted_class}")
            st.info(f"**Description:** {class_info[predicted_class]}")
            st.metric("Confidence", f"{confidence*100:.2f}%")
        else:
            st.warning(f"⚠️ Low confidence prediction: {predicted_class} ({confidence*100:.2f}%)")
            st.info("🩺 Consider consulting a specialist or uploading a clearer image.")

    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        y_pos = np.arange(len(classes))

        sorted_idx = np.argsort(predictions)
        sorted_classes = [classes[i] for i in sorted_idx]
        sorted_probs = [predictions[i] for i in sorted_idx]

        bars = ax.barh(y_pos, sorted_probs, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_classes)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('Classification Probabilities')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_position = width + 0.01
            ax.text(label_position, bar.get_y() + bar.get_height()/2,
                    f'{width*100:.1f}%', va='center')

        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("📊 Detailed Probabilities")
    prob_df = {"Class": classes, "Probability": [f"{p*100:.2f}%" for p in predictions]}
    st.table(prob_df)

def main():
    model = load_keras_model()

    if model is None:
        st.error("❌ Critical error: Model failed to load. The app cannot continue.")
        return

    uploaded_file = st.file_uploader("📤 Upload a skin lesion image", type=["jpg", "jpeg", "png"])
    camera_input = st.camera_input("📷 Or take a photo of the skin lesion")

    selected_image = None

    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
    elif camera_input is not None:
        selected_image = Image.open(io.BytesIO(camera_input.getvalue()))

    if selected_image is not None:
        st.image(selected_image, caption="📸 Uploaded Image", use_column_width=True)

        target_size = (img_width, img_height)
        img_array = preprocess_image(selected_image, target_size)

        if st.button("🔬 Classify Image"):
            with st.spinner("🔄 Analyzing the image..."):
                predictions = predict_image(model, img_array)
                display_prediction(predictions, classes, confidence_threshold)

    st.markdown("---")
    st.subheader("ℹ️ How to Use This Tool")
    st.markdown("""
    1. 📤 **Upload** a clear image of the skin lesion or 📷 **take a photo**
    2. 🔍 Click the **Classify Image** button
    3. 📊 Review the results and probability distribution
    4. 🩺 Remember to consult with a healthcare professional
    """)

if __name__ == "__main__":
    main()
