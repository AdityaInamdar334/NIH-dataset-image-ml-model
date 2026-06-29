<div align="center">

# 🩻 AI-Powered Medical Image Analysis Assistant

### End-to-End MLOps Pipeline for Explainable Multi-Label Chest X-ray Classification and LLM-Powered Radiology Report Generation

<p align="center">
An end-to-end deep learning system that combines computer vision, explainable AI, MLOps, cloud-native deployment, and large language models for automated chest X-ray interpretation.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</p>

<p align="center">

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge)
![License](https://img.shields.io/github/license/AdityaInamdar334/REPOSITORY_NAME?style=for-the-badge)

</p>

---

### 🚀 Built With

**PyTorch • ResNet50 • FastAPI • Streamlit • Docker • Grad-CAM • LangChain • OpenAI • Weights & Biases**

</div>

---

# Overview

Medical image analysis requires more than accurate predictions. Modern clinical AI systems must be **interpretable, reproducible, scalable, and deployable**.

This project implements an end-to-end machine learning pipeline for **multi-label thoracic disease classification** using the NIH ChestX-ray14 dataset. It demonstrates the complete lifecycle of a production-oriented computer vision application, from data ingestion and model training to explainability, API deployment, and AI-assisted report generation.

The repository combines deep learning with modern MLOps practices to build a deployable inference system capable of:

- Detecting multiple thoracic abnormalities from a single chest X-ray
- Explaining predictions using Grad-CAM visualizations
- Serving predictions through a production-ready FastAPI backend
- Providing an interactive Streamlit interface
- Generating structured radiology reports using an LLM

---

# Key Highlights

- End-to-end Machine Learning pipeline
- Multi-label chest X-ray classification (14 disease classes)
- Transfer Learning using ResNet50
- Explainable AI using Grad-CAM
- Experiment tracking with Weights & Biases
- FastAPI inference service
- Interactive Streamlit dashboard
- Dockerized deployment
- LLM-powered radiology report generation using LangChain
- Modular architecture following production software engineering practices

---

# System Architecture

```text
                                        NIH ChestX-ray14 Dataset
                                                  │
                                                  │
                                      Data Validation Pipeline
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         │                                                 │
                         ▼                                                 ▼
                Metadata Verification                            Image Preprocessing
                         │                                                 │
                         └────────────────────────┬────────────────────────┘
                                                  │
                                                  ▼
                                     PyTorch Dataset & DataLoader
                                                  │
                                                  ▼
                                    ResNet50 Transfer Learning
                                                  │
                        ┌─────────────────────────┴──────────────────────────┐
                        │                                                    │
                        ▼                                                    ▼
               Model Checkpoints                             Weights & Biases Tracking
                        │
                        ▼
                 Model Evaluation
                        │
                        ▼
                Explainability Layer
                    (Grad-CAM)
                        │
                        ▼
                 FastAPI Inference API
                        │
                        ▼
               Streamlit Web Dashboard
                        │
                        ▼
           LangChain Radiology Report Generator
```

---

# Project Features

## Deep Learning

- Transfer learning with **ResNet50** pretrained on ImageNet
- Multi-label classification across **14 thoracic diseases**
- Binary Cross Entropy with Logits (`BCEWithLogitsLoss`)
- Configurable training pipeline
- GPU and Apple Silicon (MPS) acceleration
- Automatic checkpointing

---

## Data Engineering

- Automated dataset download
- Dataset extraction and organization
- Metadata validation
- Missing image detection
- Custom PyTorch Dataset implementation
- Efficient DataLoader pipeline
- Data augmentation and normalization

---

## Explainable AI

Model interpretability is a critical requirement in medical AI systems.

This project integrates **Grad-CAM** to generate visual explanations showing the image regions that contribute most strongly to each prediction.

Generated visualizations can be used to:

- Inspect model reasoning
- Validate prediction quality
- Improve model transparency
- Assist clinical interpretation

---

## MLOps

The project incorporates engineering practices commonly used in production machine learning systems.

Features include:

- Experiment tracking with Weights & Biases
- Hyperparameter logging
- Automatic checkpointing
- Model versioning
- Reproducible training pipeline
- Docker-based deployment

---

## Backend

A RESTful inference API built using FastAPI provides:

- Image upload endpoint
- Multi-label predictions
- Confidence scores
- Grad-CAM generation
- JSON responses
- Automatic OpenAPI documentation

---

## Frontend

An interactive Streamlit application enables users to:

- Upload chest X-rays
- Visualize predictions
- Display confidence scores
- Inspect Grad-CAM heatmaps
- Generate AI-assisted radiology reports

---

## LLM Integration

The vision model is integrated with a LangChain pipeline that transforms raw prediction probabilities into structured radiology reports.

Generated reports include:

- Clinical findings
- Predicted abnormalities
- Confidence-aware interpretation
- Diagnostic summary

---

# Technology Stack

| Category | Technologies |
|------------|--------------|
| Programming Language | Python 3.10+ |
| Deep Learning | PyTorch, TorchVision |
| Computer Vision | OpenCV, Pillow |
| CNN Backbone | ResNet50 |
| Explainability | Grad-CAM |
| Data Processing | NumPy, Pandas |
| Experiment Tracking | Weights & Biases |
| API Framework | FastAPI |
| Frontend | Streamlit |
| LLM Framework | LangChain |
| Model Serving | Uvicorn |
| Containerization | Docker |
| Visualization | Matplotlib |
| Dataset | NIH ChestX-ray14 |

---

# Repository Structure

```text
medical-image-analysis/
│
├── data/
│   ├── images/
│   ├── processed/
│   └── Data_Entry_2017.csv
│
├── checkpoints/
│   └── best_model.pth
│
├── outputs/
│   ├── gradcam/
│   ├── predictions/
│   └── reports/
│
├── app.py
├── frontend.py
├── train.py
├── evaluate.py
├── agent.py
├── data_loader.py
├── download_data.py
│
├── requirements.txt
├── Dockerfile
├── LICENSE
└── README.md
```

---

# Project Workflow

```text
Dataset
   │
   ▼
Data Validation
   │
   ▼
Preprocessing
   │
   ▼
Model Training
   │
   ▼
Evaluation
   │
   ▼
Grad-CAM
   │
   ▼
FastAPI
   │
   ▼
Streamlit
   │
   ▼
LLM Report Generation
```

---

# Why This Project?

This repository demonstrates practical experience across multiple areas of modern AI engineering, including:

- Computer Vision
- Deep Learning
- Explainable AI (XAI)
- Machine Learning Operations (MLOps)
- REST API Development
- Docker Containerization
- Production Model Deployment
- LLM Integration
- Software Engineering Best Practices

Rather than focusing solely on model performance, the project emphasizes building a complete, deployable machine learning system that is modular, reproducible, interpretable, and ready for real-world inference workflows.
# Getting Started

## Prerequisites

Before running the project, ensure that the following software is installed.

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| PyTorch | 2.x |
| CUDA (Optional) | 11.8+ |
| Docker | Latest |
| Git | Latest |

> **Recommended Hardware**
>
> - NVIDIA GPU (CUDA) **or**
> - Apple Silicon (MPS) **or**
> - Modern multi-core CPU

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/<your-username>/medical-image-analysis.git
cd medical-image-analysis
```

---

## Create a Virtual Environment

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Verify the installation.

```bash
python -c "import torch; print(torch.__version__)"
```

---

# Dataset

The project uses the **NIH ChestX-ray14** dataset published by the National Institutes of Health (NIH).

## Dataset Statistics

| Property | Value |
|-----------|------:|
| Images | 112,120 |
| Patients | 30,805 |
| Disease Labels | 14 |
| Image Type | Frontal Chest X-rays |
| Task | Multi-label Classification |

The dataset includes annotations for:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Fibrosis
- Hernia
- Infiltration
- Mass
- Nodule
- Pleural Thickening
- Pneumonia
- Pneumothorax

---

## Download Dataset

Run

```bash
python download_data.py
```

The download script automatically

- downloads the dataset archives
- verifies downloads
- extracts images
- organizes the directory structure

For quick experimentation, a subset of the dataset can be downloaded.

---

# Data Pipeline

The data ingestion pipeline performs the following steps automatically.

```text
Compressed Archives
        │
        ▼
Archive Extraction
        │
        ▼
Metadata Verification
        │
        ▼
Missing Image Detection
        │
        ▼
Image Transformations
        │
        ▼
PyTorch Dataset
        │
        ▼
DataLoader
```

## Data Augmentation

The training pipeline applies several image augmentations to improve generalization.

| Transformation | Purpose |
|---------------|---------|
| Random Rotation | Robustness to orientation |
| Horizontal Flip | Data diversity |
| Random Resized Crop | Scale invariance |
| Color Jitter | Brightness variation |
| Image Normalization | Stable optimization |

---

# Training

Authenticate with Weights & Biases.

```bash
wandb login
```

Launch training.

```bash
python train.py \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4
```

For rapid experimentation:

```bash
python train.py \
    --subset_size 1000 \
    --epochs 2
```

---

## Training Pipeline

```text
Training Dataset
        │
        ▼
Image Augmentation
        │
        ▼
ResNet50 Backbone
        │
        ▼
Classification Head
        │
        ▼
Loss Computation
        │
        ▼
Backpropagation
        │
        ▼
Checkpoint Saving
        │
        ▼
Weights & Biases Logging
```

---

## Model Architecture

The project fine-tunes a ResNet50 backbone pretrained on ImageNet.

```text
Input Image
     │
     ▼
ResNet50 Feature Extractor
     │
     ▼
Global Average Pooling
     │
     ▼
Fully Connected Layer
     │
     ▼
14 Disease Logits
     │
     ▼
Sigmoid Activation
     │
     ▼
Disease Probabilities
```

### Loss Function

Since multiple diseases may appear in the same image, the project uses

```
Binary Cross Entropy with Logits Loss
(BCEWithLogitsLoss)
```

instead of categorical cross entropy.

---

# Experiment Tracking

All experiments are automatically tracked using **Weights & Biases**.

The following metrics are logged during training.

- Training Loss
- Validation Loss
- Learning Rate
- Mean ROC-AUC
- Per-class ROC-AUC
- Precision
- Recall
- F1 Score
- Model Artifacts

Example dashboard:

```
Epoch 10

Training Loss
Validation Loss
Learning Rate
ROC-AUC

Best Model Saved
```

---

# Evaluation

Run

```bash
python evaluate.py
```

The evaluation pipeline computes

- ROC-AUC
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Statistics
- Grad-CAM visualizations

Generated outputs are stored in

```text
outputs/

├── gradcam/
├── predictions/
└── reports/
```

---

# Explainable AI

Medical AI systems require transparency to support clinical decision making.

This project integrates **Grad-CAM** to visualize image regions contributing to each disease prediction.

```text
Chest X-ray
      │
      ▼
Forward Pass
      │
      ▼
Predicted Disease
      │
      ▼
Gradient Computation
      │
      ▼
Activation Maps
      │
      ▼
Heatmap Overlay
```

The generated heatmaps assist in understanding model behavior by highlighting the spatial regions that most strongly influenced the prediction.

---

# Running the Inference API

Launch the FastAPI server.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at

```
http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|----------|----------|-------------|
| GET | / | Service information |
| GET | /health | Health check |
| POST | /predict | Disease prediction |

---

## Example Request

```http
POST /predict
Content-Type: multipart/form-data
```

Upload a chest X-ray image.

---

## Example Response

```json
{
  "predictions": [
    {
      "label": "Pneumonia",
      "confidence": 0.94
    },
    {
      "label": "Effusion",
      "confidence": 0.82
    }
  ],
  "gradcam": "<base64-image>",
  "report": "Findings suggest..."
}
```

---

# Streamlit Dashboard

Launch the interface.

```bash
streamlit run frontend.py
```

The dashboard provides

- Image upload
- Real-time inference
- Disease probabilities
- Confidence visualization
- Grad-CAM overlays
- AI-generated radiology report

---

# Docker

Build the container.

```bash
docker build -t chestxray-ai .
```

Run

```bash
docker run -p 8000:8000 chestxray-ai
```

The API becomes available at

```
http://localhost:8000
```

---

# Deployment

The project can be deployed to cloud platforms supporting Docker containers.

Supported environments include

- AWS ECS
- AWS EC2
- Google Cloud Run
- Azure Container Apps
- Render
- Fly.io
- Railway
- DigitalOcean App Platform

---

# Reproducibility

To improve reproducibility across experiments, the training pipeline supports

- Fixed random seeds
- Deterministic dataset splits
- Configurable hyperparameters
- Saved checkpoints
- Versioned experiment tracking

These practices make experiments easier to compare, reproduce, and extend.
---

# Performance

The model is evaluated on the NIH ChestX-ray14 test split using standard multi-label classification metrics.

| Metric | Score |
|---------|------:|
| Mean ROC-AUC | TBD |
| Macro F1 Score | TBD |
| Precision | TBD |
| Recall | TBD |
| Validation Loss | TBD |
| Test Loss | TBD |

> **Note**
> Replace these values with results from your final trained model. Avoid publishing benchmark numbers that have not been experimentally verified.

---

# Example Inference

### Input

Chest X-ray Image

↓

### Predictions

| Disease | Confidence |
|----------|-----------:|
| Pneumonia | 94.3% |
| Pleural Effusion | 87.6% |
| Atelectasis | 73.1% |

↓

### Explainability

Grad-CAM highlights the regions that contributed most strongly to the prediction.

↓

### Generated Report

```
Findings

Patchy left lower lobe opacity with a moderate pleural effusion.

Impression

Findings are suspicious for left lower lobe pneumonia
with associated pleural effusion.
Clinical correlation is recommended.
```

---

# Sample Workflow

```text
            Chest X-ray
                 │
                 ▼
      Image Preprocessing
                 │
                 ▼
      ResNet50 Inference
                 │
                 ▼
 Disease Probability Scores
                 │
      ┌──────────┴───────────┐
      ▼                      ▼
 Grad-CAM             LangChain Agent
      │                      │
      ▼                      ▼
 Explainability      Radiology Report
      │                      │
      └──────────┬───────────┘
                 ▼
          FastAPI Response
                 │
                 ▼
         Streamlit Dashboard
```

---

# Repository Highlights

This repository demonstrates practical experience across several domains of machine learning engineering.

### Machine Learning

- Transfer Learning
- Multi-label Classification
- Deep Convolutional Networks
- Model Evaluation
- Explainable AI

### Software Engineering

- Modular project structure
- REST API development
- Object-oriented Python
- Configuration management
- Reusable training pipeline

### MLOps

- Experiment tracking
- Model checkpointing
- Docker containerization
- Reproducible workflows
- Deployment-ready inference service

### AI Applications

- Medical Image Analysis
- Computer Vision
- Explainable AI
- LLM Integration
- AI-assisted Report Generation

---

# Roadmap

Future improvements planned for the project.

- [ ] Vision Transformer (ViT) backbone
- [ ] DenseNet121 comparison
- [ ] EfficientNet benchmark
- [ ] ONNX export
- [ ] TensorRT inference optimization
- [ ] Quantization
- [ ] Batch inference endpoint
- [ ] DICOM image support
- [ ] Hugging Face Spaces deployment
- [ ] Kubernetes deployment
- [ ] GitHub Actions CI/CD
- [ ] Model Registry
- [ ] Continuous Monitoring
- [ ] MLflow integration
- [ ] Distributed Training
- [ ] Multi-GPU support

---

# Research References

If this repository contributes to your work, please consider citing the following publications.

### ChestX-ray14

```
Wang, X. et al.

ChestX-ray8:
Hospital-scale Chest X-ray Database
and Benchmarks on Weakly Supervised
Classification and Localization
of Common Thorax Diseases.

CVPR 2017.
```

---

### ResNet

```
He, K.
Zhang, X.
Ren, S.
Sun, J.

Deep Residual Learning
for Image Recognition.

CVPR 2016.
```

---

### Grad-CAM

```
Selvaraju, R. R.

Grad-CAM:
Visual Explanations from Deep Networks
via Gradient-based Localization.

ICCV 2017.
```

---

### PyTorch

```
Paszke, A. et al.

PyTorch:
An Imperative Style,
High-Performance Deep Learning Library.

NeurIPS 2019.
```

---

# Contributing

Contributions are welcome.

If you would like to improve the project:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a Pull Request.

Bug reports, feature requests, and documentation improvements are always appreciated.

---

# Development

Run formatting tools

```bash
black .
isort .
flake8
```

Run tests

```bash
pytest
```

---

# Citation

If you use this repository in your research, please cite it.

```bibtex
@software{medical_image_analysis_assistant,
  author = {Aditya Inamdar},
  title = {AI-Powered Medical Image Analysis Assistant},
  year = {2026},
  url = {https://github.com/AdityaInamdar334/<repository-name>}
}
```

---

# License

This project is distributed under the MIT License.

See the **LICENSE** file for details.

---

# Acknowledgements

This project builds upon the work of the following open-source communities.

- PyTorch
- TorchVision
- FastAPI
- Streamlit
- LangChain
- Weights & Biases
- NIH Clinical Center

Special thanks to the researchers who created and released the ChestX-ray14 dataset.

---

# Author

## Aditya Inamdar

Machine Learning Engineer • AI Researcher • Full-Stack Developer

- GitHub: https://github.com/AdityaInamdar334
- LinkedIn: https://www.linkedin.com/in/adityainamdar1/
- Portfolio: https://adityainamdar.com *(update if applicable)*

---

<div align="center">

**If you found this project useful, consider giving it a ⭐.**

It helps increase the visibility of the project and supports future development.

</div>
