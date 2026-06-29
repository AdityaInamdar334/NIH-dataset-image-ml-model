import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score
import wandb
from tqdm import tqdm
import ssl

# Fix SSL certificate verify failed error on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Import DataLoader and Labels from Phase 1
from data_loader import get_data_loaders, DISEASE_LABELS


def calculate_metrics(y_true, y_prob):
    """
    Computes AUC-ROC for each disease and the mean AUC across all diseases.
    Returns a dictionary of metrics.
    """
    metrics = {}
    auc_scores = []
    
    for i, label in enumerate(DISEASE_LABELS):
        try:
            # Calculate AUC only if there are both positive and negative samples
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                auc_scores.append(auc)
                metrics[f"val_auc_{label}"] = auc
            else:
                metrics[f"val_auc_{label}"] = float('nan')
        except ValueError:
            metrics[f"val_auc_{label}"] = float('nan')

    # Mean AUC ignoring NaNs
    valid_aucs = [score for score in auc_scores if not np.isnan(score)]
    metrics["val_mean_auc"] = np.mean(valid_aucs) if valid_aucs else 0.0
    
    return metrics


def train(args):
    # 1. Set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 2. Initialize Weights & Biases
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=args)

    # 3. Load Data
    print("Initializing DataLoaders...")
    csv_file = os.path.join(args.data_dir, "Data_Entry_2017.csv")
    image_dir = os.path.join(args.data_dir, "images")
    
    # Handle nested 'images/images' from tar extraction
    if os.path.isdir(os.path.join(image_dir, "images")):
        image_dir = os.path.join(image_dir, "images")
    
    train_loader, val_loader, _, _ = get_data_loaders(
        csv_file=csv_file,
        image_dir=image_dir,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        num_workers=args.num_workers
    )

    # 4. Initialize Model
    print("Loading Pre-trained ResNet50...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(DISEASE_LABELS))
    model = model.to(device)

    # 5. Loss, Optimizer, and Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # AMP setup for Mixed Precision Training (CUDA only)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Tracking variables
    best_mean_auc = 0.0
    epochs_no_improve = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_epoch = 1
    if hasattr(args, 'resume') and args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mean_auc = checkpoint.get('val_mean_auc', 0.0)

    # 6. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for images, targets in train_bar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Autocast for mixed precision
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            # Backward pass and gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_probs = []
        
        print("Running Validation...")
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation", leave=False):
                images, targets = images.to(device), targets.to(device)
                
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Compute Validation Metrics
        all_targets = np.vstack(all_targets)
        all_probs = np.vstack(all_probs)
        
        metrics = calculate_metrics(all_targets, all_probs)
        metrics["train_loss"] = avg_train_loss
        metrics["val_loss"] = avg_val_loss
        metrics["epoch"] = epoch
        metrics["learning_rate"] = optimizer.param_groups[0]['lr']
        
        current_mean_auc = metrics.get("val_mean_auc", 0.0)
        
        # Log to Weights & Biases
        wandb.log(metrics)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Mean Val AUC-ROC: {current_mean_auc:.4f}")
        
        # Step the learning rate scheduler
        scheduler.step(avg_val_loss)

        # --- CHECKPOINTING ---
        # 1. Save epoch checkpoint
        epoch_checkpoint = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_mean_auc': current_mean_auc
        }, epoch_checkpoint)
        
        # 2. Save best model (Based on Mean AUC)
        if current_mean_auc > best_mean_auc:
            print(f"New Best Mean AUC! ({best_mean_auc:.4f} --> {current_mean_auc:.4f}). Saving best_model.pth...")
            best_mean_auc = current_mean_auc
            epochs_no_improve = 0
            
            best_checkpoint = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_checkpoint)
            
            # Optional: Log best model to wandb
            # wandb.save(best_checkpoint)
        else:
            epochs_no_improve += 1
            print(f"No improvement in Mean AUC for {epochs_no_improve} epoch(s).")
            
            # 3. Early Stopping
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                break

    wandb.finish()
    print("\nTraining Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Train ResNet50 on NIH ChestX-ray14")
    
    # Required Paths & Configs
    parser.add_argument("--data_dir", type=str, default="./data", help="Base directory containing the CSV and images/ folder")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Where to save model weights")
    
    # W&B Configs
    parser.add_argument("--wandb_project", type=str, default="nih-chest-xray-assistant", help="Weights & Biases project name")
    parser.add_argument("--wandb_run", type=str, default=None, help="Weights & Biases run name")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size for rapid prototyping")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
                        
    args = parser.parse_args()
    train(args)
