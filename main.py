
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import app

if __name__ == "__main__":
    print("Starting Fake News Detector...")
    app.run(debug=True, port=5000)