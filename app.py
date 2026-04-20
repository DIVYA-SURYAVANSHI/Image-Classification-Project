import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Image Classifier", layout="wide")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Class labels
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# -------------------- CUSTOM UI --------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.title {
    text-align: center;
    font-size: 70px;
    font-weight: bold;
    color: #00FFAA;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #AAAAAA;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; font-size: 70px; color:#00FFAA;'>
🚀 AI Image Classifier
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<p style='
text-align:center;
font-size:30px;
color:#FFD700;
font-weight:500;
'>
Upload or capture an image and let AI classify it
</p>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# -------------------- LAYOUT --------------------
col1, col2 = st.columns(2)

# -------------------- IMAGE INPUT --------------------
with col1:
    st.subheader("📤 Upload or Capture Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    # -------- CAMERA CONTROL --------
    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🟢 Turn ON Camera"):
            st.session_state.camera_on = True

    with col_btn2:
        if st.button("🔴 Turn OFF Camera"):
            st.session_state.camera_on = False

    image = None

    if uploaded_file:
        image = Image.open(uploaded_file)

    elif st.session_state.camera_on:
        st.success("Camera is ON")
        camera_image = st.camera_input("Take a picture")

        if camera_image:
            image = Image.open(camera_image)
            st.session_state.camera_on = False  # auto OFF
            st.toast("📸 Image captured!")

    else:
        st.warning("Camera is OFF")

    if image:
        st.image(image, caption="Selected Image", use_column_width=True)


# -------------------- PREDICTION --------------------
with col2:
    st.subheader("🧠 Prediction Result")

    if image:
        try:
            # Preprocess
            img_resized = image.resize((32,32))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            with st.spinner("🔍 Analyzing Image..."):
                prediction = model.predict(img_array)

            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Output
            if confidence > confidence_threshold:
                st.success(f"✅ Prediction: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}")
            else:
                st.warning("⚠️ Low confidence prediction")

            # Chart
            st.subheader("📊 Prediction Probabilities")
            st.bar_chart(prediction[0])

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.info("Please upload or capture an image.")

# -------------------- FOOTER --------------------
st.markdown("---")

