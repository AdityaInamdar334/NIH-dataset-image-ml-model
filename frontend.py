import streamlit as st
import requests
import base64
from PIL import Image
import io

# FastAPI Backend URL
# Uncomment for local testing: API_URL = "http://localhost:8000"
API_URL = "https://aditya1209-nih-chest-app.hf.space"

st.set_page_config(page_title="AI Chest X-Ray Assistant", page_icon="🩻", layout="wide")

st.title("🩻 AI-Powered Medical Image Analysis Assistant")
st.markdown("""
Upload a Chest X-Ray image to get a multi-label disease prediction using our fine-tuned ResNet50 model.
The AI will also generate a **Grad-CAM heatmap** showing exactly where it's looking to make its diagnosis.
""")

# Health check to see if backend is running
try:
    health = requests.get(f"{API_URL}/health")
    if health.status_code == 200:
        st.sidebar.success("Backend API Status: Online")
    else:
        st.sidebar.warning("Backend API Status: Offline")
except requests.exceptions.ConnectionError:
    st.sidebar.error("Backend API is unreachable. Make sure FastAPI is running on port 8000.")

uploaded_file = st.file_uploader("Choose an X-Ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original X-Ray")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("AI Analysis")
        with st.spinner("Analyzing image..."):
            # Prepare file for upload
            # Reset file pointer
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                response = requests.post(f"{API_URL}/predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success(f"Top Prediction: **{result['top_disease']}**")
                    
                    # Decode and show Grad-CAM
                    heatmap_bytes = base64.b64decode(result["heatmap_base64"])
                    heatmap_image = Image.open(io.BytesIO(heatmap_bytes))
                    
                    st.image(heatmap_image, caption=f"Grad-CAM Attention Map for {result['top_disease']}", use_column_width=True)
                    
                    st.markdown("### Confidence Scores (All Diseases)")
                    
                    # Create a nice progress bar style chart for probabilities
                    for pred in result["predictions"]:
                        disease = pred["disease"]
                        prob = pred["probability"]
                        st.write(f"**{disease}**: {prob*100:.1f}%")
                        st.progress(prob)
                else:
                    st.error(f"Error from API: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend API. Please run `uvicorn app:app --reload` in your terminal.")
