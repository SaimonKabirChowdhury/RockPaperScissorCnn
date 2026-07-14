from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path(__file__).parent / "bismillah.keras"
IMAGE_SIZE = (300, 300)
CLASS_NAMES = ["paper", "rock", "scissors"]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

st.set_page_config(
    page_title="Rock Paper Scissors CNN",
    page_icon="✋",
)

st.title("Rock Paper Scissors CNN")
st.write("Upload or capture a hand gesture to classify it.")

input_method = st.radio(
    "Choose input method",
    ["Upload image", "Use camera"],
)

if input_method == "Upload image":
    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
    )
else:
    image_file = st.camera_input("Take a picture")

if image_file is not None:
    image = Image.open(image_file).convert("RGB")

    st.image(image, caption="Input image", width=350)

    processed_image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(processed_image, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    st.subheader(f"Prediction: {predicted_class.title()}")
    st.write(f"Confidence: {confidence * 100:.2f}%")

    st.write("All predictions")

    for class_name, probability in zip(CLASS_NAMES, probabilities):
        st.write(f"{class_name.title()}: {probability * 100:.2f}%")
        st.progress(float(probability))