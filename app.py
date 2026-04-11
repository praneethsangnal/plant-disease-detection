import streamlit as st
from PIL import Image
from src.predict import predict_image  # reuse your logic

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect disease and visualize model attention.")

# Upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            label, conf, output_path = predict_image(img)

        # Display result
        if "___" in label:
            plant, disease = label.split("___")
            st.success(f"🌿 Plant: {plant}")
            st.error(f"🦠 Disease: {disease}")
        else:
            st.write(f"Prediction: {label}")

        st.write(f"Confidence: {conf:.2f}")

        # Grad-CAM
        st.subheader("🔥 Grad-CAM Visualization")
        st.image(Image.open(output_path), caption="Grad-CAM")