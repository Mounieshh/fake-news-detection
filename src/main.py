from flask import Flask, render_template, request, jsonify
from gemini_predictor import GeminiPredictor
from input_handlers.url_handler import URLHandler
from image_predict.image_model import load_model as load_image_model, predict as predict_image
import os

app = Flask(__name__)
predictor = GeminiPredictor()
url_handler = URLHandler()
image_model = load_image_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check JSON or Form data
    data = request.get_json(silent=True) or request.form
    
    text = data.get('text')
    url = data.get('url')
    
    if not text and not url:
        return jsonify({"prediction": "ERROR", "explanation": "Please provide text or a URL."}), 400
        
    content_to_analyze = text
    
    if url:
        # Fetch URL content
        fetched_content = url_handler.get_content(url)
        if fetched_content.startswith("Error"):
            return jsonify({"prediction": "ERROR", "explanation": fetched_content}), 400
        content_to_analyze = f"URL: {url}\nContent: {fetched_content}"
        
    # Predict
    result = predictor.predict(content_to_analyze)
    return jsonify(result)


@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    image_file = request.files.get('image')
    context_text = request.form.get('text', '')

    if not image_file or not image_file.filename:
        return jsonify({"prediction": "ERROR", "explanation": "Please upload an image file."}), 400

    image_bytes = image_file.read()
    mime_type = image_file.mimetype or 'image/jpeg'

    result = predict_image(
        model=image_model,
        image_bytes=image_bytes,
        mime_type=mime_type,
        context=context_text
    )

    if result.get('prediction') == 'ERROR':
        return jsonify(result), 400

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)