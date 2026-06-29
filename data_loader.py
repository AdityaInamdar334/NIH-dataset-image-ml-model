import os
import tarfile
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional

# The 14 disease classes defined in the NIH ChestX-ray14 dataset
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]


def extract_archives(archive_dir: str, extract_to: str) -> None:
    """
    Extracts all .tar.gz files found in `archive_dir` into the `extract_to` folder.
    """
    os.makedirs(extract_to, exist_ok=True)
    tar_files = glob.glob(os.path.join(archive_dir, '*.tar.gz'))
    
    if not tar_files:
        print(f"No .tar.gz files found in {archive_dir}.")
        return

    print(f"Found {len(tar_files)} archive(s). Starting extraction...")
    for tar_path in sorted(tar_files):
        print(f"Extracting {os.path.basename(tar_path)}...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Using a safe extraction method or simply extractall since it's a trusted local dataset
                tar.extractall(path=extract_to)
        except Exception as e:
            print(f"Failed to extract {tar_path}: {e}")
            
    print(f"Extraction complete. Images are saved to {extract_to}")


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for the NIH ChestX-ray14 dataset.
    Handles loading images, applying transformations, and returning multi-label tensors.
    """
    def __init__(self, df: pd.DataFrame, image_dir: str, transform: Optional[transforms.Compose] = None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            # Convert to RGB as ResNet expects 3 channels
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Fallback: create a blank image to avoid crashing the training loop
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        # Extract the binary labels for the 14 diseases
        labels = row[DISEASE_LABELS].values.astype(np.float32)
        labels = torch.tensor(labels)
        
        return image, labels


def get_data_loaders(csv_file: str, image_dir: str, batch_size: int = 32, 
                     subset_size: Optional[int] = None, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    Parses the dataset CSV, creates train/val/test splits, applies transforms, 
    and returns PyTorch DataLoaders.
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Process 'Finding Labels' into individual binary columns (Multi-label setup)
    # The original CSV contains a column 'Finding Labels' with labels separated by '|'
    for label in DISEASE_LABELS:
        df[label] = df['Finding Labels'].apply(lambda x: 1.0 if label in x else 0.0)
        
    # Scale down for rapid prototyping if subset_size is provided
    if subset_size is not None and subset_size < len(df):
        print(f"Subsampling {subset_size} images for rapid prototyping...")
        df = df.sample(n=subset_size, random_state=42).reset_index(drop=True)
        
    # Verify data integrity: only keep rows where the image file actually exists
    existing_images = []
    for img_name in df['Image Index']:
        existing_images.append(os.path.exists(os.path.join(image_dir, img_name)))
    
    missing_count = len(df) - sum(existing_images)
    if missing_count > 0:
        print(f"Warning: {missing_count} images specified in the CSV were not found in {image_dir}.")
    
    df = df[existing_images].reset_index(drop=True)
    print(f"Total valid images ready for split: {len(df)}")
    
    # Split data: 70% Train, 15% Val, 15% Test
    # First split off the test set (15%)
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    # Then split train and val. To get 15% of total from the remaining 85%, val_size = 15/85 ≈ 0.17647
    train_df, val_df = train_test_split(train_val_df, test_size=(0.15/0.85), random_state=42)
    
    print(f"Data Split - Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Define Image Transforms
    # ResNet models expect images resized to 224x224 and normalized with ImageNet mean/std
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15), # Slight rotation for data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create Dataset Instances
    train_dataset = ChestXrayDataset(train_df, image_dir, transform=train_transform)
    val_dataset = ChestXrayDataset(val_df, image_dir, transform=val_test_transform)
    test_dataset = ChestXrayDataset(test_df, image_dir, transform=val_test_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
                             
    return train_loader, val_loader, test_loader, df


def verify_data_integrity(csv_file: str, image_dir: str) -> None:
    """
    Checks the dataset for missing image paths and prints the class distribution.
    """
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    print(f"Total entries in CSV: {len(df)}")
    
    # Check for missing images
    missing_images = sum(not os.path.exists(os.path.join(image_dir, img_name)) for img_name in df['Image Index'])
    print(f"Missing images in directory ({image_dir}): {missing_images}")
    
    # Check class distribution
    print("\nClass Distribution:")
    for label in DISEASE_LABELS:
        # Summing occurrences where the disease label is present in 'Finding Labels'
        count = df['Finding Labels'].apply(lambda x: 1 if label in x else 0).sum()
        percentage = (count / len(df)) * 100
        print(f"  {label:<18}: {count:<6} ({percentage:.2f}%)")


if __name__ == '__main__':
    # Configuration paths for local testing
    DATA_DIR = './data'
    IMAGE_DIR = './data/images'
    CSV_FILE = './data/Data_Entry_2017.csv'
    ARCHIVE_DIR = './data/archives' # Where the user has stored the 12 tar.gz files
    
    # 1. Extract archives if images folder is empty or doesn't exist
    if not os.path.exists(IMAGE_DIR) or len(os.listdir(IMAGE_DIR)) == 0:
        print("Images directory is empty or missing. Extracting archives...")
        extract_archives(ARCHIVE_DIR, IMAGE_DIR)
    else:
        print(f"Images directory {IMAGE_DIR} already populated.")
    
    # 2. Verify data integrity
    if os.path.exists(CSV_FILE):
        print("\n--- Verifying Data Integrity ---")
        verify_data_integrity(CSV_FILE, IMAGE_DIR)
        
        # 3. Test DataLoaders (using a subset for speed)
        print("\n--- Testing DataLoaders (Subset: 1000 images) ---")
        train_loader, val_loader, test_loader, processed_df = get_data_loaders(
            csv_file=CSV_FILE, 
            image_dir=IMAGE_DIR, 
            batch_size=32, 
            subset_size=1000, 
            num_workers=2
        )
        
        # Fetch a single batch to verify shapes
        for images, labels in train_loader:
            print(f"\nBatch shape successfully loaded:")
            print(f"  Images tensor shape: {images.shape}")
            print(f"  Labels tensor shape: {labels.shape}")
            print(f"\nSample labels for first image in batch:\n  {labels[0].numpy()}")
            break
    else:
        print(f"\nSkipping loader test: CSV file {CSV_FILE} not found. Please ensure it's in the data folder.")
