# Fake News Detection System with BERT

A powerful, AI-driven application designed to detect fake news using a fine-tuned BERT model. This system analyzes text and URLs to provide accurate credibility assessments, helping users verify information in real-time.

## 🚀 Features

- **BERT-Powered Analysis**: Utilizes a state-of-the-art BERT model for high-accuracy fake news classification.
- **Dual Input Modes**:
  - **Text Analysis**: Directly paste news content for instant verification.
  - **URL Analysis**: Analyze online articles by simply providing the link.
- **Modern User Interface**: 
  - Clean, minimal design with a responsive layout.
  - **Dark/Light Mode** toggle for comfortable viewing.
- **Transparent Results**:
  - Clear "REAL" or "FAKE" prediction.
  - Confidence scores.
  - Detailed explanations for every analysis.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **AI Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Frontend**: HTML5, CSS3, JavaScript
- **Web Scraping**: BeautifulSoup4 (for URL content extraction)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fake-news-detection-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory and add any necessary API keys (if applicable for extended features).

## 🏃 usage

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Access the interface**
   Open your browser and navigate to `http://localhost:5000`.

3. **Analyze News**
   - **Text**: Paste the article text into the input box and click "Analyze".
   - **URL**: Switch to the URL tab, paste the link, and click "Analyze".

## 🛡️ Best Practices

- Use this tool as a supplementary verification step.
- Always cross-reference critical information with multiple reputable sources.
- The BERT model provides probability-based assessments, not absolute truths.

## 👥 Contributing

Contributions are welcome!
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
