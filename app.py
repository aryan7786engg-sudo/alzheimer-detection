import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Alzheimer Detection", layout="centered")

st.title("🧠 Alzheimer Disease Detection Using Deep Learning")
st.write("Upload an MRI image to detect the stage of Alzheimer's Disease.")

model = load_model("best_model.keras")
classes = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]

uploaded = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])

if uploaded:
    st.image(uploaded, caption="Uploaded MRI Image", width=250)

    img = image.load_img(uploaded, target_size=(224,224))
    img_arr = image.img_to_array(img)/255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    predictions = model.predict(img_arr)
    pred_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    st.success(f"### ✅ Prediction: **{classes[pred_index]}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
