import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from googleapiclient.discovery import build

# ----------------------------
# ✅ CONFIG
# ----------------------------
YOUTUBE_API_KEY = "AIza......your real key here..."  # <-- PUT YOUR YOUTUBE API KEY HERE

st.set_page_config(page_title="Alzheimer Detection", layout="centered")

st.title("🧠 Alzheimer Disease Detection Using Deep Learning")
st.write("Upload an MRI image to detect the stage of Alzheimer's Disease.")


# ----------------------------
# ✅ Model Overview
# ----------------------------
with st.expander("📘 Model Overview"):
    st.write("""
    **Model:** MobileNetV2 (Transfer Learning)

    **Classification Categories:**
    - Mild Dementia  
    - Moderate Dementia  
    - Very Mild Dementia  
    - Non Demented  

    **Training Details:**
    - Epochs: 8  
    - Image Size: 224 × 224  
    - Optimizer: Adam  
    - Loss Function: Categorical Crossentropy  
    - Dataset: Kaggle Alzheimer MRI Dataset  
    """)


# ----------------------------
# ✅ LOAD MODEL
# ----------------------------
if not os.path.exists("best_model.keras"):
    st.error("❌ Model file 'best_model.keras' not found. Upload it to your GitHub repo.")
    st.stop()

model = load_model("best_model.keras")
classes = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]


# ----------------------------
# ✅ YouTube API Function
# ----------------------------
def fetch_youtube_videos(query):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            q=query,
            part="snippet",
            maxResults=4,
            type="video"
        )
        response = request.execute()

        videos = []
        for item in response["items"]:
            title = item["snippet"]["title"]
            vid_id = item["id"]["videoId"]
            thumb = item["snippet"]["thumbnails"]["medium"]["url"]
            link = f"https://www.youtube.com/watch?v={vid_id}"
            videos.append((title, thumb, link))

        return videos

    except Exception as e:
        st.error("⚠️ Could not fetch YouTube videos.")
        return []


# ----------------------------
# ✅ IMAGE UPLOAD
# ----------------------------
uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Uploaded MRI Image", width=250)

    img = image.load_img(uploaded, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    predictions = model.predict(img_arr)
    pred_index = np.argmax(predictions)
    confidence = float(np.max(predictions) * 100)

    st.success(f"### ✅ Prediction: **{classes[pred_index]}**")
    st.info(f"Confidence: **{confidence:.2f}%**")


    # -----------------------------------------
    # ✅ Probability Bar Chart
    # -----------------------------------------
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(classes, predictions[0], color="skyblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    plt.xticks(rotation=20)
    st.pyplot(fig)


    # -----------------------------------------
    # ✅ YouTube Recommendations
    # -----------------------------------------
    st.subheader("🎥 Recommended YouTube Videos")

    topic = classes[pred_index] + " Alzheimer awareness"
    videos = fetch_youtube_videos(topic)

    for title, thumb, link in videos:
        st.markdown(f"""
        <a href="{link}" target="_blank">
            <img src="{thumb}" width="240"><br>
            {title}
        </a>
        <br><br>
        """, unsafe_allow_html=True)



