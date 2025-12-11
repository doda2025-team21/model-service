# using some official python image as base
FROM python:3.12.9-slim AS builder

# navigate to the working directory
WORKDIR /app

# check if the requirements change
COPY requirements.txt .

# install system dependencies (if any python libs are out of date)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt


FROM python:3.12.9-slim 

COPY --from=builder /usr/local /usr/local

WORKDIR /app

# copy src directory (not entire directory to avoid copying large files)
COPY src/ ./src/

# create output folder for models
RUN mkdir -p /app/output

ENV OUTPUT_DIR=/app/output \
    MODEL_PATH=/app/output/model.joblib \
    MODEL_PORT=8081 \
    METRICS_PORT=9091

# expose the ports (app + metrics)
EXPOSE 8081 9091

# run the Flask app
CMD ["python", "src/serve_model.py"]