"""
Flask API of the SMS Spam detection model.
Downloads model from GitHub release on startup if not present locally.
"""
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd
import os 
import requests
import logging

from text_preprocessing import prepare, _extract_message_len, _text_process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
swagger = Swagger(app)

# Configuration
DEFAULT_MODEL_URL = "https://api.github.com/repos/doda25-team21/model-service/releases/latest"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
PORT = int(os.getenv('MODEL_PORT', 8081))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model variable
model = None

def download_model_from_release():
    """Download model from latest GitHub release if not present locally."""
    logger.info(f"Checking for model at {MODEL_PATH}...")
    
    if os.path.exists(MODEL_PATH):
        logger.info("Model already exists locally.")
        return True
    
    logger.info("Model not found. Downloading from GitHub...")
    
    try:
        response = requests.get(DEFAULT_MODEL_URL, timeout=10)
        response.raise_for_status()
        release_data = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch release: {e}")
        return False
    
    # Download all .joblib files from the release
    for asset in release_data.get("assets", []):
        if asset["name"].endswith(".joblib"):
            download_url = asset["browser_download_url"]
            local_path = os.path.join(OUTPUT_DIR, asset["name"])
            
            logger.info(f"Downloading {asset['name']}...")
            try:
                r = requests.get(download_url, timeout=30)
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(r.content)
                logger.info(f"Successfully downloaded {asset['name']}")
            except requests.RequestException as e:
                logger.error(f"Failed to download {asset['name']}: {e}")
                return False
    
    if os.path.exists(MODEL_PATH):
        logger.info("Model downloaded successfully.")
        return True
    
    logger.error("Model file not found after download.")
    return False

def load_model():
    """Load model from disk. Downloads if not present."""
    global model
    
    # Try to download if missing
    download_model_from_release()
    
    # Try to load the model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
        return None

# Load model on startup
logger.info("Initializing model service...")
model = load_model()

if model is None:
    logger.warning("Starting service without a model. Predictions will fail.")

@app.route('/predict', methods=['POST'])
def predict():
    
    if model is None:
        return jsonify({"error": "Model not available"}), 503
    
    try:
        input_data = request.get_json()
        sms = input_data.get('sms')
        
        if not sms:
            return jsonify({"error": "No SMS provided"}), 400
        
        processed_sms = prepare(sms)
        prediction = model.predict(processed_sms)[0]
        
        res = {
            "result": prediction,
            "classifier": "decision tree",
            "sms": sms
        }
        print(res)
        return jsonify(res)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model": "ready" if model is not None else "not_ready"
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=False)