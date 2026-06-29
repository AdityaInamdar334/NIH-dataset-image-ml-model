import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import wandb
from tqdm import tqdm
import ssl

# Fix SSL certificate verify failed error on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Grad-CAM dependencies
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data_loader import get_data_loaders, DISEASE_LABELS

def evaluate_model(model, test_loader, device, log_wandb=False):
    """
    Evaluates the model on the test set, computing AUC-ROC, Precision, Recall, F1, 
    and Confusion Matrices per disease. Logs to wandb if specified.
    """
    model.eval()
    all_targets = []
    all_probs = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    
    metrics = {}
    auc_scores = []
    
    print("\n--- Detailed Metrics per Disease ---")
    for i, label in enumerate(DISEASE_LABELS):
        # 1. Calculate AUC-ROC
        try:
            if len(np.unique(all_targets[:, i])) > 1:
                auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
                auc_scores.append(auc)
                metrics[f"{label}/auc"] = auc
            else:
                auc = float('nan')
        except ValueError:
            auc = float('nan')
            
        # 2. Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets[:, i], all_preds[:, i], average='binary', zero_division=0
        )
        metrics[f"{label}/precision"] = precision
        metrics[f"{label}/recall"] = recall
        metrics[f"{label}/f1"] = f1
        
        # 3. Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(all_targets[:, i], all_preds[:, i], labels=[0, 1]).ravel()
        metrics[f"{label}/tp"] = tp
        metrics[f"{label}/fp"] = fp
        metrics[f"{label}/fn"] = fn
        metrics[f"{label}/tn"] = tn
        
        print(f"{label:<18} | AUC: {auc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
        print(f"                   | TP: {tp:<4} | FP: {fp:<4} | FN: {fn:<4} | TN: {tn:<4}")
            
    # Mean AUC
    valid_aucs = [score for score in auc_scores if not np.isnan(score)]
    mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    metrics["mean_auc"] = mean_auc
    print(f"\nMean AUC-ROC across all diseases: {mean_auc:.4f}")
    
    # 4. Global Classification Report
    print("\n--- Global Classification Report ---")
    report = classification_report(all_targets, all_preds, target_names=DISEASE_LABELS, zero_division=0)
    print(report)
    
    if log_wandb:
        wandb.log(metrics)
        print("Metrics successfully logged to Weights & Biases.")
        
    return all_targets, all_probs, all_preds


def generate_gradcam(model, image_tensor, target_class_idx, original_image_np):
    """
    Generates and overlays a Grad-CAM heatmap on the original image targeting the last conv layer.
    """
    # Target the final convolutional layer of ResNet50
    target_layers = [model.layer4[-1]]
    
    # Construct the CAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Define the target class
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    # Generate the heatmap
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay on the original image (expects original image to be in [0, 1])
    visualization = show_cam_on_image(original_image_np / 255.0, grayscale_cam, use_rgb=True)
    return visualization


def run_explainability(model, test_loader, device, output_dir="./output/gradcam", log_wandb=False):
    """
    Runs Grad-CAM on a few sample images and saves the visualizations locally (and to wandb).
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    print("\nGenerating Grad-CAM visualizations...")
    
    # Take the first batch for visualization
    images, targets = next(iter(test_loader))
    images, targets = images.to(device), targets.to(device)
    
    outputs = model(images)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).int()
    
    # Inverse transform to get original visual image from normalized tensor
    from torchvision import transforms
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    wandb_images = []
    
    # Generate 5 sample examples
    for i in range(min(5, images.size(0))):
        img_tensor = images[i].unsqueeze(0)
        true_labels = targets[i].cpu().numpy()
        pred_labels = preds[i].cpu().numpy()
        pred_probs = probs[i].cpu().detach().numpy()
        
        # Original image array
        orig_img_tensor = inv_normalize(images[i])
        orig_img_np = orig_img_tensor.permute(1, 2, 0).cpu().numpy()
        orig_img_np = np.clip(orig_img_np * 255, 0, 255).astype(np.uint8)
        
        # Visualize the disease with the highest probability
        target_idx = np.argmax(pred_probs)
        disease_name = DISEASE_LABELS[target_idx]
        is_correct = (true_labels[target_idx] == pred_labels[target_idx])
        
        # 5. Generate Grad-CAM
        vis_image = generate_gradcam(model, img_tensor, target_idx, orig_img_np)
        
        # Plot and save
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(orig_img_np)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        
        ax[1].imshow(vis_image)
        status = "Correct" if is_correct else "Incorrect"
        title = f"Grad-CAM: {disease_name}\n({status} Prediction - Prob: {pred_probs[target_idx]:.2f})"
        ax[1].set_title(title)
        ax[1].axis('off')
        
        save_path = os.path.join(output_dir, f"sample_{i+1}_{disease_name}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved visualization to {save_path}")
        
        if log_wandb:
            wandb_images.append(wandb.Image(save_path, caption=title))
            
    if log_wandb and wandb_images:
        wandb.log({"gradcam_visualizations": wandb_images})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 and run Grad-CAM")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log evaluation metrics to Weights & Biases")
    
    args = parser.parse_args()
    
    if args.wandb:
        wandb.init(project="nih-chest-xray-assistant", name="evaluation")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load trained model
    print("Loading model structure...")
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(DISEASE_LABELS))
    
    if os.path.exists(args.checkpoint_path):
        print(f"Loading weights from {args.checkpoint_path}...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        # Handle different saved dictionary structures
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Weights loaded successfully.")
    else:
        print(f"WARNING: Checkpoint not found at {args.checkpoint_path}. Evaluating with random weights.")
        
    model = model.to(device)
    
    # 2. Get DataLoaders
    print("Initializing Test DataLoader...")
    csv_file = os.path.join(args.data_dir, "Data_Entry_2017.csv")
    image_dir = os.path.join(args.data_dir, "images")
    
    if os.path.isdir(os.path.join(image_dir, "images")):
        image_dir = os.path.join(image_dir, "images")
    
    _, _, test_loader, _ = get_data_loaders(
        csv_file=csv_file,
        image_dir=image_dir,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        num_workers=args.num_workers
    )
    
    # 3 & 4. Evaluation
    evaluate_model(model, test_loader, device, log_wandb=args.wandb)
    
    # 5. Explainability
    run_explainability(model, test_loader, device, log_wandb=args.wandb)
    
    if args.wandb:
        wandb.finish()
        
    print("\nPhase 3 Evaluation and Explainability Complete.")
