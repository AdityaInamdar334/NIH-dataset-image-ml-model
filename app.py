import io
import os
import base64
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

# Grad-CAM dependencies
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data_loader import DISEASE_LABELS

app = FastAPI(title="NIH Chest X-Ray AI Assistant", version="1.0.0")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
device = None
model = None
cam = None

# Standard ImageNet normalization used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.on_event("startup")
def load_model():
    global device, model, cam
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading model on device: {device}")
    
    # Initialize Model architecture
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(DISEASE_LABELS))
    
    checkpoint_path = os.getenv("MODEL_PATH", "./checkpoints/best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded successfully.")
    else:
        print("WARNING: Checkpoint not found. Using random weights for demonstration.")
        
    model = model.to(device)
    model.eval()
    
    # Initialize Grad-CAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

def image_to_base64(img_array):
    """Convert numpy image array to base64 string"""
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Original image numpy array for Grad-CAM overlay
        orig_img_np = np.array(image.resize((224, 224)))
        
        # Transform for model input
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
        # Format predictions
        predictions = []
        for i, label in enumerate(DISEASE_LABELS):
            predictions.append({
                "disease": label,
                "probability": float(probs[i])
            })
            
        # Sort predictions by probability descending
        predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
        top_prediction_idx = DISEASE_LABELS.index(predictions[0]["disease"])
        
        # Generate Grad-CAM for the top predicted class
        # (We turn on grad temporarily for Grad-CAM)
        with torch.enable_grad():
            targets = [ClassifierOutputTarget(top_prediction_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
        # Create overlay visualization
        vis_image = show_cam_on_image(orig_img_np / 255.0, grayscale_cam, use_rgb=True)
        
        # Convert visualization to base64 to send to frontend
        heatmap_base64 = image_to_base64(vis_image)
        
        return {
            "predictions": predictions,
            "top_disease": predictions[0]["disease"],
            "heatmap_base64": heatmap_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
