import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
import os
from pathlib import Path
import numpy as np
from .transforms import get_image_transform, get_text_transform


class MultimodalDataset(Dataset):
    """Dataset for multimodal fake news detection with images and text"""
    
    def __init__(self, tsv_path, image_cache_dir='./image_cache', transform=None, 
                 text_transform=None, max_samples=None, download_images=True):
        """
        Args:
            tsv_path: Path to TSV file
            image_cache_dir: Directory to cache downloaded images
            transform: Image transformations
            text_transform: Text preprocessing function
            max_samples: Limit number of samples (for testing)
            download_images: Whether to download images or use cached
        """
        self.df = pd.read_csv(tsv_path, sep='\t')
        
        # Filter only samples with images
        self.df = self.df[self.df['hasImage'] == True].reset_index(drop=True)
        
        if max_samples:
            self.df = self.df.head(max_samples)
        
        self.image_cache_dir = Path(image_cache_dir)
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.transform = transform
        self.text_transform = text_transform
        self.download_images = download_images
        
        # Cache for failed downloads
        self.failed_downloads = set()
        
    def __len__(self):
        return len(self.df)
    
    def download_image(self, url, image_id):
        """Download and cache image from URL"""
        cache_path = self.image_cache_dir / f"{image_id}.jpg"
        
        # Return cached image if exists
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert('RGB')
            except:
                cache_path.unlink()  # Delete corrupted file
        
        # Download image
        if url in self.failed_downloads:
            return None
            
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0'
            })
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Cache the image
            img.save(cache_path)
            return img
        except Exception as e:
            self.failed_downloads.add(url)
            return None
    
    def __getitem__(self, idx):
        """Get a single sample"""
        row = self.df.iloc[idx]
        
        # Get image
        image_url = row['image_url']
        image_id = row['id']
        
        image = self.download_image(image_url, image_id)
        
        # If image download failed, create a black placeholder
        if image is None:
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply image transformations
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor if no transform provided
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Get text
        text = str(row['title']) if pd.notna(row['title']) else ""
        
        # Apply text transformations
        if self.text_transform:
            text = self.text_transform(text)
        
        # Get label (using 2-way classification: 0=real, 1=fake)
        label = int(row['2_way_label'])
        
        return {
            'image': image,
            'text': text,
            'label': torch.tensor(label, dtype=torch.long),
            'id': image_id
        }


def get_dataloaders(train_path, val_path, test_path=None, batch_size=32, 
                   num_workers=0, image_cache_dir='./image_cache', max_samples=None):
    """
    Create DataLoaders for training, validation, and test sets
    
    Args:
        train_path: Path to training TSV
        val_path: Path to validation TSV
        test_path: Path to test TSV (optional)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        image_cache_dir: Directory for caching images
        max_samples: Limit samples per dataset (for testing)
    
    Returns:
        train_loader, val_loader, (test_loader if test_path provided)
    """
    train_transform = get_image_transform(train=True)
    val_transform = get_image_transform(train=False)
    text_transform = get_text_transform()
    
    train_dataset = MultimodalDataset(
        train_path, 
        image_cache_dir=image_cache_dir,
        transform=train_transform,
        text_transform=text_transform,
        max_samples=max_samples
    )
    
    val_dataset = MultimodalDataset(
        val_path,
        image_cache_dir=image_cache_dir,
        transform=val_transform,
        text_transform=text_transform,
        max_samples=max_samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if test_path:
        test_dataset = MultimodalDataset(
            test_path,
            image_cache_dir=image_cache_dir,
            transform=val_transform,
            text_transform=text_transform,
            max_samples=max_samples
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function to handle variable length text"""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    ids = [item['id'] for item in batch]
    
    return {
        'image': images,
        'text': texts,
        'label': labels,
        'id': ids
    }
