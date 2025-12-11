"""
Flask API of the SMS Spam detection model.
Exposes Prometheus metrics on a separate port for monitoring.
"""
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger
import os
import time
import threading
import logging

from prometheus_client import Counter, Histogram, Gauge, start_http_server
from text_preprocessing import prepare, _text_process, _extract_message_len

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
swagger = Swagger(app)

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "output/model.joblib")
PORT = int(os.getenv('MODEL_PORT', 8081))
METRICS_PORT = int(os.getenv('METRICS_PORT', 9091))

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

# Counter: Total predictions made (with labels for result and confidence)
PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total number of predictions made by the model',
    ['model_name', 'prediction', 'confidence_bucket']
)

# Gauge: Whether model is loaded and ready
MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether the model is loaded and ready (1=yes, 0=no)',
    ['model_name', 'version']
)

# Histogram: Model inference latency
INFERENCE_DURATION = Histogram(
    'model_inference_duration_seconds',
    'Time taken for model inference in seconds',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Counter: Prediction errors
PREDICTION_ERRORS = Counter(
    'model_prediction_errors_total',
    'Total number of prediction errors',
    ['model_name', 'error_type']
)

# =============================================================================
# MODEL LOADING
# =============================================================================

# Global model variable
model = None
MODEL_NAME = "spam_classifier"
MODEL_VERSION = "v1"

def load_model():
    """Load the ML model from disk."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            MODEL_LOADED.labels(model_name=MODEL_NAME, version=MODEL_VERSION).set(1)
            return model
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            MODEL_LOADED.labels(model_name=MODEL_NAME, version=MODEL_VERSION).set(0)
            return None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL_LOADED.labels(model_name=MODEL_NAME, version=MODEL_VERSION).set(0)
        return None

# Load model on startup
logger.info("Initializing model service...")
model = load_model()

# =============================================================================
# METRICS SERVER
# =============================================================================

def start_metrics_server():
    """Start the Prometheus metrics HTTP server on a separate port."""
    logger.info(f"Starting Prometheus metrics server on port {METRICS_PORT}")
    start_http_server(METRICS_PORT)

# Start metrics server in background thread
metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
metrics_thread.start()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether an SMS is Spam.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    if model is None:
        PREDICTION_ERRORS.labels(model_name=MODEL_NAME, error_type="model_not_loaded").inc()
        return jsonify({"error": "Model not available"}), 503
    
    try:
        input_data = request.get_json()
        sms = input_data.get('sms')
        
        if not sms:
            PREDICTION_ERRORS.labels(model_name=MODEL_NAME, error_type="invalid_input").inc()
            return jsonify({"error": "No SMS provided"}), 400
        
        # Time the inference
        start_time = time.time()
        processed_sms = prepare(sms)
        prediction = model.predict(processed_sms)[0]
        inference_time = time.time() - start_time
        
        # Record metrics
        INFERENCE_DURATION.labels(model_name=MODEL_NAME).observe(inference_time)
        
        # Determine confidence bucket (simulated based on message length for demo)
        msg_len = len(sms)
        if msg_len < 50:
            confidence_bucket = "high"
        elif msg_len < 150:
            confidence_bucket = "medium"
        else:
            confidence_bucket = "low"
        
        PREDICTIONS_TOTAL.labels(
            model_name=MODEL_NAME,
            prediction=prediction,
            confidence_bucket=confidence_bucket
        ).inc()
        
        res = {
            "result": prediction,
            "classifier": "decision tree",
            "sms": sms
        }
        print(res)
        return jsonify(res)
        
    except Exception as e:
        PREDICTION_ERRORS.labels(model_name=MODEL_NAME, error_type="prediction_failed").inc()
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model": "ready" if model is not None else "not_ready"
    })

@app.route('/metrics-info', methods=['GET'])
def metrics_info():
    """Information about available metrics."""
    return jsonify({
        "metrics_port": METRICS_PORT,
        "metrics_endpoint": f"http://localhost:{METRICS_PORT}/metrics",
        "available_metrics": [
            "model_predictions_total - Counter with labels: model_name, prediction, confidence_bucket",
            "model_loaded - Gauge with labels: model_name, version",
            "model_inference_duration_seconds - Histogram with labels: model_name",
            "model_prediction_errors_total - Counter with labels: model_name, error_type"
        ]
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=False)
