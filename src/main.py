from flask import Flask, render_template, request, jsonify
from gemini_predictor import GeminiPredictor
from input_handlers.url_handler import URLHandler
import os

app = Flask(__name__)
predictor = GeminiPredictor()
url_handler = URLHandler()

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)