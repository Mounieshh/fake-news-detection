import torch
import torchvision.transforms as transforms
from PIL import Image
import re
import string


def get_image_transform(train=True, image_size=224):
    """
    Get image transformations for training or validation
    
    Args:
        train: If True, apply data augmentation
        image_size: Target image size (default: 224 for ResNet/EfficientNet)
    
    Returns:
        torchvision.transforms.Compose object
    """
    if train:
        # Training transforms with data augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet standards
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Validation/Test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def clean_text(text):
    """
    Clean and preprocess text
    
    Args:
        text: Input text string
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags (Reddit style)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def get_text_transform():
    """
    Get text transformation/cleaning function
    
    Returns:
        Text cleaning function
    """
    return clean_text


class TextVectorizer:
    """Simple text vectorizer using TF-IDF approach"""
    
    def __init__(self, max_features=5000, max_length=100):
        """
        Args:
            max_features: Maximum number of features/vocabulary size
            max_length: Maximum sequence length
        """
        self.max_features = max_features
        self.max_length = max_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        """
        Build vocabulary from list of texts
        
        Args:
            texts: List of text strings
        """
        from collections import Counter
        
        word_freq = Counter()
        
        for text in texts:
            words = clean_text(text).split()
            word_freq.update(words)
        
        # Get most common words
        most_common = word_freq.most_common(self.max_features - 2)  # -2 for PAD and UNK
        
        # Build vocabulary (0: PAD, 1: UNK)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_size = len(self.word_to_idx)
        print(f"Vocabulary built with {self.vocab_size} words")
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of indices
        
        Args:
            text: Input text string
        
        Returns:
            List of word indices
        """
        words = clean_text(text).split()
        sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1 is UNK
        
        # Pad or truncate to max_length
        if len(sequence) < self.max_length:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return sequence
    
    def texts_to_sequences(self, texts):
        """
        Convert multiple texts to sequences
        
        Args:
            texts: List of text strings
        
        Returns:
            Tensor of shape (batch_size, max_length)
        """
        sequences = [self.text_to_sequence(text) for text in texts]
        return torch.tensor(sequences, dtype=torch.long)


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized image tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor
