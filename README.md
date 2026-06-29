# AI-Powered Medical Image Analysis Assistant 🩻

This project is an end-to-end Machine Learning Operations (MLOps) pipeline for multi-label chest X-ray classification using the **NIH ChestX-ray14 dataset**. It features a fine-tuned ResNet50 model, Grad-CAM explainability, Weights & Biases experiment tracking, a FastAPI backend, a Streamlit frontend, and a LangChain LLM agent for natural language report generation.

## Features

- **Phase 1: Data Pipeline (`data_loader.py`)** - Automated extraction, data integrity checks, and a robust PyTorch `Dataset` & `DataLoader` implementation with image augmentations.
- **Phase 2: Model Training (`train.py`)** - Transfer learning with `ResNet50`. Uses `BCEWithLogitsLoss` for 14-class multi-label classification. Integrated with **Weights & Biases** for loss and AUC-ROC tracking.
- **Phase 3: Explainability (`evaluate.py`)** - Computes detailed metrics (Precision, Recall, F1, AUC) and uses **Grad-CAM** to generate attention heatmaps overlaid on the original X-rays.
- **Phase 4 & 5: Deployment (`app.py`, `frontend.py`, `Dockerfile`)** - A **FastAPI** inference server that serves predictions and base64 encoded Grad-CAM heatmaps. A **Streamlit** frontend for an interactive user interface. Dockerized for easy deployment to AWS, Render, or Fly.io.
- **Phase 6: Agentic AI (`agent.py`)** - A **LangChain** agent that ingests the vision model's predictions and generates a professional, human-readable radiological report.

## Setup Instructions

### 1. Environment Setup
Install the required dependencies using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Get the Dataset
The complete NIH dataset is ~42GB. To download the dataset and the metadata CSV, simply run the download script (you can edit it to only download the first 2GB for rapid prototyping):
```bash
python3 download_data.py
```

### 3. Extract and Verify Data
Run the data loader to unpack the `.tar.gz` files into the `data/images/` folder and verify data integrity against the CSV:
```bash
python3 data_loader.py
```

## Usage

### Training the Model
Make sure you are logged into Weights & Biases:
```bash
wandb login
```
Start training! You can use `--subset_size` for quick local testing:
```bash
python3 train.py --epochs 10 --batch_size 32 --subset_size 1000
```
This will automatically save the best model to `./checkpoints/best_model.pth`.

### Evaluating the Model
To generate the classification report and save Grad-CAM heatmaps for the test set:
```bash
python3 evaluate.py
```
Check `./output/gradcam/` for the generated visual explanations.

### Running the Application Locally
**Start the FastAPI Backend:**
```bash
uvicorn app:app --reload --port 8000
```

**Start the Streamlit Frontend (in a new terminal):**
```bash
streamlit run frontend.py
```

### Running the Agent (Optional)
If you want to test the LangChain report generator, export your OpenAI API key and run the script:
```bash
export OPENAI_API_KEY="your-api-key"
python3 agent.py
```

## Docker Deployment
To build and run the entire API in a Docker container:
```bash
docker build -t nih-chest-xray-api .
docker run -p 8000:8000 nih-chest-xray-api
```
