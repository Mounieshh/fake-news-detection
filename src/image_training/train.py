import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from .loader import get_dataloaders, collate_fn
from .transforms import TextVectorizer


class MultimodalFakeNewsDetector(nn.Module):
    """Multimodal model combining image and text features"""
    
    def __init__(self, vocab_size, embedding_dim=256, text_hidden_dim=128, 
                 num_classes=2, pretrained=True):
        """
        Args:
            vocab_size: Size of text vocabulary
            embedding_dim: Dimension of text embeddings
            text_hidden_dim: Hidden dimension for text LSTM
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Use pretrained image encoder
        """
        super(MultimodalFakeNewsDetector, self).__init__()
        
        # Image encoder (ResNet50) - More powerful than ResNet18
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.image_encoder = models.resnet50(weights=weights)
        image_feature_dim = self.image_encoder.fc.in_features  # 2048 for ResNet50
        self.image_encoder.fc = nn.Identity()  # Remove final FC layer
        
        # Text encoder (Embedding + LSTM)
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.text_lstm = nn.LSTM(
            embedding_dim, 
            text_hidden_dim, 
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        text_feature_dim = text_hidden_dim * 2  # Bidirectional
        
        # Fusion layer
        combined_dim = image_feature_dim + text_feature_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image, text):
        """
        Forward pass
        
        Args:
            image: Image tensor (batch_size, 3, 224, 224)
            text: Text sequence tensor (batch_size, seq_length)
        
        Returns:
            Logits (batch_size, num_classes)
        """
        # Extract image features
        image_features = self.image_encoder(image)  # (batch_size, 512)
        
        # Extract text features
        text_embedded = self.text_embedding(text)  # (batch_size, seq_length, embedding_dim)
        _, (hidden, _) = self.text_lstm(text_embedded)
        # Concatenate forward and backward hidden states from last layer
        text_features = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch_size, text_hidden_dim*2)
        
        # Combine features
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # Classification
        logits = self.fusion(combined_features)
        
        return logits


def train_epoch(model, train_loader, criterion, optimizer, device, text_vectorizer):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        texts = batch['text']
        labels = batch['label'].to(device)
        
        # Convert texts to sequences
        text_sequences = text_vectorizer.texts_to_sequences(texts).to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, text_sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, text_vectorizer):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            texts = batch['text']
            labels = batch['label'].to(device)
            
            # Convert texts to sequences
            text_sequences = text_vectorizer.texts_to_sequences(texts).to(device)
            
            # Forward pass
            outputs = model(images, text_sequences)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        train_path=args.train_data,
        val_path=args.val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_cache_dir=args.image_cache,
        max_samples=args.max_samples
    )
    
    # Build text vocabulary
    print("Building text vocabulary...")
    text_vectorizer = TextVectorizer(max_features=args.vocab_size, max_length=args.max_length)
    
    # Collect all texts from training data
    all_texts = []
    for batch in tqdm(train_loader, desc="Collecting texts"):
        all_texts.extend(batch['text'])
    
    text_vectorizer.build_vocab(all_texts)
    
    # Save text vectorizer
    vectorizer_path = output_dir / 'text_vectorizer.pt'
    torch.save({
        'word_to_idx': text_vectorizer.word_to_idx,
        'idx_to_word': text_vectorizer.idx_to_word,
        'vocab_size': text_vectorizer.vocab_size,
        'max_length': text_vectorizer.max_length
    }, vectorizer_path)
    print(f"Text vectorizer saved to {vectorizer_path}")
    
    # Initialize model
    print("Initializing model...")
    model = MultimodalFakeNewsDetector(
        vocab_size=text_vectorizer.vocab_size,
        embedding_dim=args.embedding_dim,
        text_hidden_dim=args.text_hidden_dim,
        num_classes=2,
        pretrained=args.pretrained
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, text_vectorizer
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, text_vectorizer
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args)
            }, best_model_path)
            print(f"✓ Best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, checkpoint_path)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*50)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multimodal Fake News Detector')
    
    # Data arguments
    parser.add_argument('--train_data', type=str, 
                        default='multimodal_only_samples/multimodal_train.tsv',
                        help='Path to training TSV file')
    parser.add_argument('--val_data', type=str,
                        default='multimodal_only_samples/multimodal_validate.tsv',
                        help='Path to validation TSV file')
    parser.add_argument('--image_cache', type=str, default='./image_cache',
                        help='Directory to cache downloaded images')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=5000,
                        help='Vocabulary size for text')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum text sequence length')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Text embedding dimension')
    parser.add_argument('--text_hidden_dim', type=int, default=128,
                        help='Text LSTM hidden dimension')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='Use pretrained image encoder')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./model',
                        help='Directory to save model and checkpoints')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples (for testing)')
    
    args = parser.parse_args()
    train(args)
