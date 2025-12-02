#!/usr/bin/env python3
"""
Fake News Detection System - Main Application
Handles input processing for text, image, and URL inputs
"""

import sys
import os
from typing import Dict, Any, Union, List
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from input_handlers.text_handler import TextHandler
from input_handlers.image_handler import ImageHandler
from input_handlers.url_handler import URLHandler
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from data_manager import DataManager
from huggingface_predictor import HuggingFacePredictor
from gemini_predictor import GeminiPredictor
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup


class FakeNewsDetector:
    """Main class for fake news detection system"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the fake news detection system"""
        from utils.logger import get_logger
        self.logger = get_logger(__name__)
        self.text_handler = TextHandler()
        self.image_handler = ImageHandler()
        self.url_handler = URLHandler()
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.data_manager = DataManager(data_dir)
        self.hf_predictor = HuggingFacePredictor()
        self.gemini_predictor = GeminiPredictor()

        self.BeautifulSoup = BeautifulSoup()
        
        # Flask app initialization
        self.app = Flask(__name__)
        self.setup_routes()
        
        self.logger.info("Fake News Detection System Initialized!")
        self.logger.info("=" * 50)
    
    def setup_routes(self):
        """Setup Flask routes for web API"""
        
        @self.app.route('/')
        def index():
            """Home page"""
            return render_template('index.html')
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze():
            """Handle form submissions from the web interface"""
            try:
                if 'text' in request.form and request.form['text'].strip():
                    text = request.form['text']
                    result = self.process_input(text, 'text', save_result=True)
                elif 'url' in request.form and request.form['url'].strip():
                    url = request.form['url']
                    result = self.process_input(url, 'url', save_result=True)
                elif 'image' in request.files:
                    image = request.files['image']
                    if image.filename:
                        result = self.process_input(image, 'image', save_result=True)
                    else:
                        return jsonify({'error': 'No image file provided'}), 400
                else:
                    return jsonify({'error': 'No input provided'}), 400
                
                return jsonify(result)
            except Exception as e:
                self.logger.exception("Analysis failed")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/status')
        def status():
            """Get system status"""
            return jsonify(self.get_system_status())
        
        @self.app.route('/predict/text', methods=['POST'])
        def predict_text():
            """Analyze text input"""
            try:
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({'error': 'No text provided'}), 400
                
                text = data['text']
                save_result = data.get('save_result', True)
                
                result = self.process_input(text, 'text', save_result)
                return jsonify(result)
            
            except Exception as e:
                self.logger.exception("Text prediction failed")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict/url', methods=['POST'])
        def predict_url():
            """Analyze URL input"""
            try:
                data = request.get_json()
                if not data or 'url' not in data:
                    return jsonify({'error': 'No URL provided'}), 400
                
                url = data['url']
                save_result = data.get('save_result', True)
                
                result = self.process_input(url, 'url', save_result)
                return jsonify(result)
            
            except Exception as e:
                self.logger.exception("URL prediction failed")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict/image', methods=['POST'])
        def predict_image():
            """Analyze image input"""
            try:
                if 'image' not in request.files:
                    return jsonify({'error': 'No image file provided'}), 400
                
                image_file = request.files['image']
                image_bytes = image_file.read()
                save_result = request.form.get('save_result', 'true').lower() == 'true'
                
                result = self.process_input(image_bytes, 'image', save_result)
                return jsonify(result)
            
            except Exception as e:
                self.logger.exception("Image prediction failed")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict/gemini/text', methods=['POST'])
        def predict_gemini_text():
            """Analyze text using Gemini AI"""
            try:
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({'error': 'No text provided'}), 400
                
                text = data['text']
                result = self.gemini_predictor.predict_text(text)
                return jsonify(result)
            
            except Exception as e:
                self.logger.exception("Gemini text prediction failed")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict/gemini/image', methods=['POST'])
        def predict_gemini_image():
            """Analyze image using Gemini AI"""
            try:
                if 'image' not in request.files:
                    return jsonify({'error': 'No image file provided'}), 400
                
                image_file = request.files['image']
                image_bytes = image_file.read()
                
                result = self.gemini_predictor.predict_image(image_bytes)
                return jsonify(result)
            
            except Exception as e:
                self.logger.exception("Gemini image prediction failed")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/results', methods=['GET'])
        def list_results():
            """List all saved results"""
            try:
                input_type = request.args.get('type', None)
                results = self.list_saved_results(input_type)
                return jsonify({'results': results, 'count': len(results)})
            
            except Exception as e:
                self.logger.exception("Failed to list results")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/results/<path:result_id>', methods=['GET'])
        def get_result(result_id):
            """Get a specific saved result"""
            try:
                result = self.load_saved_result(result_id)
                return jsonify(result)
            
            except Exception as e:
                self.logger.exception(f"Failed to load result: {result_id}")
                return jsonify({'error': str(e)}), 404
        
        @self.app.route('/results/<path:result_id>', methods=['DELETE'])
        def delete_result(result_id):
            """Delete a specific saved result"""
            try:
                success = self.delete_saved_result(result_id)
                if success:
                    return jsonify({'message': 'Result deleted successfully'})
                else:
                    return jsonify({'error': 'Failed to delete result'}), 500
            
            except Exception as e:
                self.logger.exception(f"Failed to delete result: {result_id}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/export/csv', methods=['GET'])
        def export_csv():
            """Export results to CSV"""
            try:
                csv_path = self.export_results_to_csv()
                return jsonify({
                    'message': 'Results exported successfully',
                    'file_path': csv_path
                })
            
            except Exception as e:
                self.logger.exception("Failed to export results")
                return jsonify({'error': str(e)}), 500
    
    def run_server(self, host='0.0.0.0', port=5000, debug=False):
        """Run Flask server"""
        self.logger.info(f"Starting Flask server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
    
    def process_input(self, input_data: Union[str, bytes], input_type: str = "auto", save_result: bool = True) -> Dict[str, Any]:
        """
        Process input based on type or auto-detect
        
        Args:
            input_data: The input data (text, image path, URL, or image bytes)
            input_type: Type of input ('text', 'image', 'url', 'auto')
            save_result: Whether to save the result to JSON file
            
        Returns:
            Dict containing processing results
        """
        try:
            # Auto-detect input type if not specified
            if input_type == "auto":
                input_type = self._detect_input_type(input_data)
            
            self.logger.info(f"Processing {input_type} input...")
            
            # Process based on type
            if input_type == "text":
                # Use Gemini predictor behind the scenes but format result like HuggingFace
                result = self.gemini_predictor.predict_text(input_data)
                # Format the result to match the expected structure
                processed_result = {
                    'prediction': result['prediction'],
                    'probability': result['probability'],
                    'explanation': result['explanation'],
                    'red_flags': result.get('red_flags', []),
                    'status': 'success'
                }
            elif input_type == "image":
                if isinstance(input_data, bytes):
                    result = self.gemini_predictor.predict_image(input_data)
                else:
                    # For file objects from Flask
                    image_bytes = input_data.read()
                    result = self.gemini_predictor.predict_image(image_bytes)
                processed_result = {
                    'prediction': result['prediction'],
                    'probability': result['probability'],
                    'explanation': result['explanation'],
                    'red_flags': result.get('red_flags', []),
                    'status': 'success'
                }
            elif input_type == "url":
                # Process URL using URL handler and then analyze resulting text
                url_result = self.url_handler.process_url_input(input_data)
                if url_result.get('status') != 'success':
                    return {
                        'status': 'error',
                        'error': url_result.get('error', 'Failed to process URL'),
                        'prediction': 'ERROR',
                        'probability': 0,
                        'explanation': f"URL processing failed: {url_result.get('error', 'Unknown error')}",
                        'red_flags': ['URL processing error']
                    }
                
                # Extract article content from the URL result
                if 'extracted_content' not in url_result:
                    return {
                        'status': 'error',
                        'prediction': 'ERROR',
                        'probability': 0,
                        'explanation': 'Content extraction failed from URL',
                        'red_flags': ['Content extraction error']
                    }
                
                article_content = url_result.get('extracted_content', {}).get('main_content', '')
                
                if not article_content:
                    return {
                        'status': 'error',
                        'prediction': 'ERROR',
                        'probability': 0,
                        'explanation': 'Could not extract article content from the URL',
                        'red_flags': ['Content extraction failed']
                    }
                
                # Add URL and title context to the analysis
                analysis_text = f"URL: {url_result['url']}\nTitle: {url_result.get('title', '')}\n\nContent:\n{article_content}"
                
                result = self.gemini_predictor.predict_text(analysis_text)
                processed_result = {
                    'prediction': result['prediction'],
                    'probability': result['probability'],
                    'explanation': result['explanation'],
                    'red_flags': result.get('red_flags', []),
                    'source_url': url_result['url'],
                    'title': url_result.get('title', ''),
                    'status': 'success'
                }
            else:
                processed_result = {
                    'status': 'error',
                    'error': f'Unsupported input type: {input_type}'
                }
            
            # Save result if requested and processing was successful
            if save_result and processed_result['status'] == 'success':
                try:
                    saved_path = self.data_manager.save_preprocessing_result(processed_result, input_type)
                    processed_result['saved_to'] = saved_path
                    self.logger.info(f"Result saved to: {saved_path}")
                except Exception as e:
                    self.logger.warning(f"Could not save result: {str(e)}")
                    processed_result['save_error'] = str(e)
            
            return result
                
        except Exception as e:
            self.logger.exception("Processing failed")
            return {
                'status': 'error',
                'error': f'Processing failed: {str(e)}'
            }
    
    def _detect_input_type(self, input_data: Union[str, bytes]) -> str:
        """Auto-detect input type"""
        if isinstance(input_data, bytes):
            return "image"
        
        if isinstance(input_data, str):
            # Check if it's a URL
            if input_data.startswith(('http://', 'https://', 'www.')):
                return "url"
            
            # Check if it's a file path
            if os.path.exists(input_data):
                ext = os.path.splitext(input_data)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    return "image"
                elif ext in ['.txt', '.csv', '.json']:
                    return "text"
            
            # Default to text
            return "text"
        
        return "text"
    
    def _process_text_input(self, input_data: str) -> Dict[str, Any]:
        """Process text input with preprocessing pipeline and HuggingFace prediction"""
        # Check if it's a file path
        if os.path.exists(input_data):
            # Use preprocessing pipeline for file processing
            result = self.preprocessing_pipeline.process_file(input_data)
        else:
            # Use preprocessing pipeline for direct text processing
            result = self.preprocessing_pipeline.process_text(input_data, "text")
        
        # Add processing metadata
        result['processed_at'] = datetime.now().isoformat()
        result['processor'] = 'PreprocessingPipeline'
        
        # Run HuggingFace prediction if preprocessing was successful
        if result.get('status') == 'success':
            # Extract text for prediction
            text_for_prediction = self._extract_text_for_prediction(result)
            
            if text_for_prediction:
                self.logger.info("Running HuggingFace fake news prediction...")
                prediction_result = self.hf_predictor.predict(text_for_prediction)
                result['huggingface_prediction'] = prediction_result
                
                # Log the prediction result
                if prediction_result.get('status') == 'success':
                    pred_data = prediction_result.get('prediction', {})
                    label = pred_data.get('label', 'UNKNOWN')
                    score = pred_data.get('score', 0.0)
                    confidence = pred_data.get('confidence', 0.0)
                    
                    self.logger.info(f"Fake News Prediction - Label: {label}, Score: {score:.4f}, Confidence: {confidence:.2f}%")
                else:
                    self.logger.warning(f"Prediction failed: {prediction_result.get('error', 'Unknown error')}")
            else:
                self.logger.warning("No text available for prediction")
                result['huggingface_prediction'] = {
                    'status': 'error',
                    'error': 'No text available for prediction'
                }
        
        return result
    
    def _extract_text_for_prediction(self, result: Dict[str, Any]) -> str:
        """
        Extract text from preprocessing result for HuggingFace prediction
        
        Args:
            result (Dict[str, Any]): Preprocessing result
            
        Returns:
            str: Text to use for prediction
        """
        # Try to get cleaned text first, then original text
        data = result.get('data', {})
        
        # Try cleaned text first (preferred)
        if 'cleaned_text' in data and data['cleaned_text']:
            return data['cleaned_text']
        
        # Fall back to original text
        if 'original_text' in data and data['original_text']:
            return data['original_text']
        
        # Try other text fields
        text_fields = ['extracted_text', 'text', 'content']
        for field in text_fields:
            if field in data and data[field]:
                text = data[field]
                if isinstance(text, str):
                    return text
                elif isinstance(text, dict) and 'text' in text:
                    return text['text']
        
        return ""
    
    def _process_image_input(self, input_data: Union[str, bytes]) -> Dict[str, Any]:
        """Process image input"""
        if isinstance(input_data, str):
            # File path
            result = self.image_handler.process_image_input(input_data)
        else:
            # Image bytes
            result = self.image_handler.process_image_data(input_data)
        
        # Add processing metadata
        result['processed_at'] = datetime.now().isoformat()
        result['processor'] = 'ImageHandler'
        
        return result
    
    def _process_url_input(self, input_data: str) -> Dict[str, Any]:
        """Process URL input"""
        result = self.url_handler.process_url_input(input_data)
        
        # Add processing metadata
        result['processed_at'] = datetime.now().isoformat()
        result['processor'] = 'URLHandler'
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and capabilities"""
        pipeline_status = self.preprocessing_pipeline.get_pipeline_status()
        data_summary = self.data_manager.get_result_summary()
        
        return {
            'status': 'operational',
            'text_processing': True,
            'image_processing': self.image_handler.ocr_available,
            'url_processing': self.url_handler.bs4_available,
            'preprocessing_pipeline': pipeline_status,
            'data_storage': {
                'enabled': True,
                'data_directory': str(self.data_manager.data_dir),
                'summary': data_summary
            },
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    
    def list_saved_results(self, input_type: str = None) -> List[Dict[str, Any]]:
        """
        List all saved preprocessing results
        
        Args:
            input_type: Filter by input type (text, image, url)
            
        Returns:
            List of result metadata
        """
        return self.data_manager.list_saved_results(input_type)
    
    def load_saved_result(self, file_path: str) -> Dict[str, Any]:
        """
        Load a saved preprocessing result
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Loaded preprocessing result
        """
        return self.data_manager.load_preprocessing_result(file_path)
    
    def delete_saved_result(self, file_path: str) -> bool:
        """
        Delete a saved preprocessing result
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        return self.data_manager.delete_result(file_path)
    
    def export_results_to_csv(self, output_file: str = None) -> str:
        """
        Export all results metadata to CSV
        
        Args:
            output_file: Output CSV file path
            
        Returns:
            Path to exported CSV file
        """
        return self.data_manager.export_results_to_csv(output_file)
    
    def interactive_mode(self):
        """Run interactive mode for testing"""
        self.logger.info("Interactive Mode - Test Your Inputs!")
        self.logger.info("=" * 50)
        
        while True:
            print("\nChoose input type:")
            print("1. Text input")
            print("2. Image file path")
            print("3. URL")
            print("4. System status")
            print("5. List saved results")
            print("6. Load saved result")
            print("7. Export results to CSV")
            print("8. Start Flask server")
            print("9. Exit")
            
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == "1":
                text = input("Enter text to analyze: ").strip()
                if text:
                    result = self.process_input(text, "text")
                    self._display_result(result)
                else:
                    self.logger.warning("No text entered")
            
            elif choice == "2":
                image_path = input("Enter image file path: ").strip()
                if image_path:
                    result = self.process_input(image_path, "image")
                    self._display_result(result)
                else:
                    self.logger.warning("No image path entered")
            
            elif choice == "3":
                url = input("Enter URL to analyze: ").strip()
                if url:
                    result = self.process_input(url, "url")
                    self._display_result(result)
                else:
                    self.logger.warning("No URL entered")
            
            elif choice == "4":
                status = self.get_system_status()
                self._display_result(status)
            
            elif choice == "5":
                self._list_saved_results_interactive()
            
            elif choice == "6":
                self._load_saved_result_interactive()
            
            elif choice == "7":
                self._export_results_interactive()
            
            elif choice == "8":
                print("\nStarting Flask server...")
                print("Press Ctrl+C to stop the server")
                try:
                    self.run_server(debug=True)
                except KeyboardInterrupt:
                    print("\nServer stopped")
            
            elif choice == "9":
                self.logger.info("Goodbye!")
                break
            
            else:
                self.logger.warning("Invalid choice. Please enter 1-9.")
    
    def _display_result(self, result: Dict[str, Any]):
        """Display processing results in a formatted way"""
        print("\n" + "=" * 50)
        print("PROCESSING RESULT")
        print("=" * 50)
        
        if result.get('status') == 'success':
            print("Status: Success")
            
            if 'input_type' in result:
                print(f"Input Type: {result['input_type']}")
            
            if 'cleaned_text' in result:
                print(f"Cleaned Text: {result['cleaned_text'][:200]}...")
            
            if 'extracted_text' in result:
                text_data = result['extracted_text']
                if isinstance(text_data, dict) and text_data.get('text'):
                    print(f"Extracted Text: {text_data['text'][:200]}...")
            
            if 'extracted_content' in result:
                content = result['extracted_content']
                if isinstance(content, dict):
                    if 'title' in content:
                        print(f"Title: {content['title']}")
                    if 'main_content' in content:
                        print(f"Content: {content['main_content'][:200]}...")
            
            # Display features if available
            if 'data' in result and 'features' in result['data']:
                features = result['data']['features']
                print(f"\nExtracted Features ({len(features)} total):")
                for key, value in list(features.items())[:10]:  # Show first 10 features
                    print(f"   {key}: {value}")
                if len(features) > 10:
                    print(f"   ... and {len(features) - 10} more features")
            
            # Display preprocessing summary if available
            if 'summary' in result:
                summary = result['summary']
                print(f"\nPreprocessing Summary:")
                print(f"   Features extracted: {summary.get('total_features_extracted', 0)}")
                print(f"   Cleaning steps: {summary.get('cleaning_steps_applied', 0)}")
                print(f"   Final word count: {summary.get('final_word_count', 0)}")
                print(f"   Processing successful: {summary.get('processing_successful', False)}")
            
            # Display HuggingFace prediction results if available
            if 'huggingface_prediction' in result:
                hf_pred = result['huggingface_prediction']
                print(f"\nFAKE NEWS PREDICTION:")
                print("=" * 30)
                
                if hf_pred.get('status') == 'success':
                    pred_data = hf_pred.get('prediction', {})
                    label = pred_data.get('label', 'UNKNOWN')
                    score = pred_data.get('score', 0.0)
                    confidence = pred_data.get('confidence', 0.0)
                    
                    print(f"   Label: {label}")
                    print(f"   Score: {score:.4f}")
                    print(f"   Confidence: {confidence:.2f}%")
                    print(f"   Model: {hf_pred.get('model_name', 'Unknown')}")
                    
                    # Add interpretation
                    if 'FAKE' in label.upper() or 'FALSE' in label.upper():
                        print(f"   ⚠️  This content appears to be FAKE NEWS")
                    elif 'REAL' in label.upper() or 'TRUE' in label.upper():
                        print(f"   ✓  This content appears to be REAL NEWS")
                    else:
                        print(f"   ?  Prediction result unclear")
                else:
                    print(f"   ✗  Prediction failed: {hf_pred.get('error', 'Unknown error')}")
                print("=" * 30)
            
        else:
            print("Status: Error")
            if 'error' in result:
                print(f"Error: {result['error']}")
        
        print("=" * 50)
    
    def _list_saved_results_interactive(self):
        """Interactive method to list saved results"""
        print("\nSaved Results:")
        print("=" * 30)
        
        # Ask for filter
        filter_type = input("Filter by type (text/image/url) or press Enter for all: ").strip().lower()
        if filter_type not in ['text', 'image', 'url']:
            filter_type = None
        
        results = self.list_saved_results(filter_type)
        
        if not results:
            print("No saved results found.")
            return
        
        print(f"\nFound {len(results)} saved results:")
        for i, result in enumerate(results[:10], 1):  # Show first 10
            print(f"{i}. {result['filename']}")
            print(f"   Type: {result['input_type']} | Status: {result['status']}")
            print(f"   Features: {result['feature_count']} | Created: {result['created_at'][:19]}")
            print()
        
        if len(results) > 10:
            print(f"... and {len(results) - 10} more results")
    
    def _load_saved_result_interactive(self):
        """Interactive method to load a saved result"""
        print("\nLoad Saved Result:")
        print("=" * 30)
        
        file_path = input("Enter the full path to the JSON file: ").strip()
        
        if not file_path:
            self.logger.warning("No file path provided")
            return
        
        try:
            result = self.load_saved_result(file_path)
            self.logger.info("Result loaded successfully")
            self._display_result(result)
        except Exception as e:
            self.logger.exception(f"Error loading result: {str(e)}")
    
    def _export_results_interactive(self):
        """Interactive method to export results to CSV"""
        print("\nExport Results to CSV:")
        print("=" * 30)
        
        output_file = input("Enter output CSV file path (or press Enter for default): ").strip()
        
        if not output_file:
            output_file = None
        
        try:
            csv_path = self.export_results_to_csv(output_file)
            self.logger.info(f"Results exported successfully to: {csv_path}")
        except Exception as e:
            self.logger.exception(f"Error exporting results: {str(e)}")


def main():
    """Main function to run the fake news detection system"""
    try:
        # Initialize the system
        detector = FakeNewsDetector()
        
        # Check if command line arguments are provided
        if len(sys.argv) > 1:
            if sys.argv[1] == '--server':
                # Start Flask server
                host = sys.argv[2] if len(sys.argv) > 2 else '0.0.0.0'
                port = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
                detector.run_server(host=host, port=port, debug=False)
            else:
                # Command line mode
                input_data = sys.argv[1]
                input_type = sys.argv[2] if len(sys.argv) > 2 else "auto"
                
                detector.logger.info(f"Processing: {input_data}")
                result = detector.process_input(input_data, input_type)
                detector._display_result(result)
        else:
            # Interactive mode
            detector.interactive_mode()
            
    except KeyboardInterrupt:
        from utils.logger import get_logger
        get_logger(__name__).info("\nSystem interrupted by user. Goodbye!")
    except Exception as e:
        from utils.logger import get_logger
        get_logger(__name__).exception(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()