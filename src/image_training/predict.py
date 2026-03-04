import torch
import torch.nn as nn
from PIL import Image
import argparse
from pathlib import Path
import requests
from io import BytesIO

from .train import MultimodalFakeNewsDetector
from .transforms import get_image_transform, clean_text, TextVectorizer


class FakeNewsPredictor:
    """Predictor for multimodal fake news detection"""
    
    def __init__(self, model_path, vectorizer_path, device=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model checkpoint
            vectorizer_path: Path to saved text vectorizer
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load text vectorizer
        print(f"Loading text vectorizer from {vectorizer_path}...")
        vectorizer_data = torch.load(vectorizer_path, map_location=self.device)
        
        self.text_vectorizer = TextVectorizer()
        self.text_vectorizer.word_to_idx = vectorizer_data['word_to_idx']
        self.text_vectorizer.idx_to_word = vectorizer_data['idx_to_word']
        self.text_vectorizer.vocab_size = vectorizer_data['vocab_size']
        self.text_vectorizer.max_length = vectorizer_data['max_length']
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model arguments from checkpoint
        model_args = checkpoint.get('args', {})
        
        self.model = MultimodalFakeNewsDetector(
            vocab_size=self.text_vectorizer.vocab_size,
            embedding_dim=model_args.get('embedding_dim', 256),
            text_hidden_dim=model_args.get('text_hidden_dim', 128),
            num_classes=2,
            pretrained=False  # Don't need pretrained weights when loading checkpoint
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image transform
        self.image_transform = get_image_transform(train=False)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    def load_image_from_url(self, url):
        """
        Load image from URL
        
        Args:
            url: Image URL
        
        Returns:
            PIL Image
        """
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0'
            })
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image from URL: {e}")
            return None
    
    def load_image_from_path(self, path):
        """
        Load image from file path
        
        Args:
            path: Path to image file
        
        Returns:
            PIL Image
        """
        try:
            image = Image.open(path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image from path: {e}")
            return None
    
    def predict(self, image, text):
        """
        Make prediction on image and text
        
        Args:
            image: PIL Image or path to image or image URL
            text: Text string (title/headline)
        
        Returns:
            Dictionary with prediction results
        """
        # Load image if needed
        if isinstance(image, str):
            if image.startswith('http'):
                image = self.load_image_from_url(image)
            else:
                image = self.load_image_from_path(image)
        
        if image is None:
            return {
                'error': 'Failed to load image',
                'prediction': None,
                'confidence': None
            }
        
        # Preprocess image
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        
        # Preprocess text
        text_sequence = self.text_vectorizer.texts_to_sequences([text]).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor, text_sequence)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        # Convert to labels
        label_map = {0: 'REAL', 1: 'FAKE'}
        prediction = label_map[predicted.item()]
        confidence_score = confidence.item()
        
        return {
            'prediction': prediction,
            'confidence': confidence_score,
            'probabilities': {
                'REAL': probabilities[0][0].item(),
                'FAKE': probabilities[0][1].item()
            }
        }
    
    def predict_batch(self, images, texts):
        """
        Make predictions on batch of images and texts
        
        Args:
            images: List of PIL Images or paths or URLs
            texts: List of text strings
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image, text in zip(images, texts):
            result = self.predict(image, text)
            results.append(result)
        
        return results


def predict_from_tsv(predictor, tsv_path, output_path=None, max_samples=None):
    """
    Make predictions on data from TSV file
    
    Args:
        predictor: FakeNewsPredictor instance
        tsv_path: Path to TSV file
        output_path: Path to save predictions (optional)
        max_samples: Limit number of samples
    
    Returns:
        DataFrame with predictions
    """
    import pandas as pd
    from tqdm import tqdm
    
    # Load TSV
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Filter only samples with images
    df = df[df['hasImage'] == True].reset_index(drop=True)
    
    if max_samples:
        df = df.head(max_samples)
    
    # Make predictions
    predictions = []
    confidences = []
    
    print(f"Making predictions on {len(df)} samples...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_url = row['image_url']
        text = str(row['title']) if pd.notna(row['title']) else ""
        
        result = predictor.predict(image_url, text)
        
        predictions.append(result.get('prediction', 'ERROR'))
        confidences.append(result.get('confidence', 0.0))
    
    # Add predictions to dataframe
    df['predicted_label'] = predictions
    df['confidence'] = confidences
    
    # Calculate accuracy if labels are available
    if '2_way_label' in df.columns:
        label_map = {0: 'REAL', 1: 'FAKE'}
        df['true_label'] = df['2_way_label'].map(label_map)
        df['correct'] = df['predicted_label'] == df['true_label']
        
        accuracy = df['correct'].mean() * 100
        print(f"\nAccuracy: {accuracy:.2f}%")
        
        # Confusion matrix
        from collections import Counter
        correct_counts = Counter(zip(df['true_label'], df['predicted_label']))
        print("\nConfusion Matrix:")
        print(f"True REAL, Predicted REAL: {correct_counts[('REAL', 'REAL')]}")
        print(f"True REAL, Predicted FAKE: {correct_counts[('REAL', 'FAKE')]}")
        print(f"True FAKE, Predicted REAL: {correct_counts[('FAKE', 'REAL')]}")
        print(f"True FAKE, Predicted FAKE: {correct_counts[('FAKE', 'FAKE')]}")
    
    # Save predictions
    if output_path:
        df.to_csv(output_path, sep='\t', index=False)
        print(f"\nPredictions saved to {output_path}")
    
    return df


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Predict with Multimodal Fake News Detector')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='./model/best_model.pt',
                        help='Path to saved model checkpoint')
    parser.add_argument('--vectorizer_path', type=str, default='./model/text_vectorizer.pt',
                        help='Path to saved text vectorizer')
    
    # Prediction mode
    parser.add_argument('--mode', type=str, choices=['single', 'tsv'], default='single',
                        help='Prediction mode: single or tsv')
    
    # Single prediction arguments
    parser.add_argument('--image', type=str, help='Image path or URL')
    parser.add_argument('--text', type=str, help='Text/headline')
    
    # TSV prediction arguments
    parser.add_argument('--tsv_path', type=str,
                        help='Path to TSV file for batch prediction')
    parser.add_argument('--output_path', type=str,
                        help='Path to save predictions')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples for TSV prediction')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = FakeNewsPredictor(
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path
    )
    
    if args.mode == 'single':
        # Single prediction
        if not args.image or not args.text:
            print("Error: --image and --text are required for single prediction mode")
            return
        
        print("\nMaking prediction...")
        print(f"Image: {args.image}")
        print(f"Text: {args.text}")
        print("-" * 50)
        
        result = predictor.predict(args.image, args.text)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nPrediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"\nProbabilities:")
            print(f"  REAL: {result['probabilities']['REAL']:.2%}")
            print(f"  FAKE: {result['probabilities']['FAKE']:.2%}")
    
    elif args.mode == 'tsv':
        # TSV batch prediction
        if not args.tsv_path:
            print("Error: --tsv_path is required for tsv prediction mode")
            return
        
        predict_from_tsv(
            predictor,
            tsv_path=args.tsv_path,
            output_path=args.output_path,
            max_samples=args.max_samples
        )


if __name__ == '__main__':
    main()
