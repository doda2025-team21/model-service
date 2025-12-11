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
import time
from threading import Thread

from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY
from text_preprocessing import prepare, _extract_message_len, _text_process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
METRICS_PORT = int(os.getenv('METRICS_PORT', 9091))

model_predictions_total = Counter(
    'model_predictions_total',
    'Total predictions made by ML model',
    ['model_name', 'prediction', 'confidence_bucket']
)

model_loaded = Gauge(
    'model_loaded',
    'Whether model is loaded and ready (1=yes, 0=no)',
    ['model_name', 'version']
)

model_inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Model inference time',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

app = Flask(__name__)
swagger = Swagger(app)

# Configuration
DEFAULT_MODEL_URL = "https://api.github.com/repos/doda2025-team21/model-service/releases/latest"
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
    
    # Download model files from the release (.joblib or .bin)
    for asset in release_data.get("assets", []):
        if asset["name"].endswith(".joblib") or asset["name"].endswith(".bin"):
            download_url = asset["browser_download_url"]
            # Always save as model.joblib so load_model can find it
            local_path = MODEL_PATH
            
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
    model_loaded.labels(model_name='spam_classifier', version='v1').set(0)
else:
    model_loaded.labels(model_name='spam_classifier', version='v1').set(1)

@app.route('/predict', methods=['POST'])
def predict():
    
    if model is None:
        return jsonify({"error": "Model not available"}), 503
    
    try:
        input_data = request.get_json()
        sms = input_data.get('sms')
        
        if not sms:
            return jsonify({"error": "No SMS provided"}), 400
        
        # Time the inference
        start_time = time.time()
        processed_sms = prepare(sms)
        prediction = model.predict(processed_sms)[0]
        duration = time.time() - start_time
        
        # Record metrics
        model_inference_duration.labels(model_name='spam_classifier').observe(duration)
        confidence = 'high'  # simplified - real impl would use predict_proba
        model_predictions_total.labels(
            model_name='spam_classifier',
            prediction=prediction,
            confidence_bucket=confidence
        ).inc()
        
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
    # Start Prometheus metrics server on separate port
    logger.info(f"Starting metrics server on port {METRICS_PORT}")
    start_http_server(METRICS_PORT)
    
    app.run(host="0.0.0.0", port=PORT, debug=False)