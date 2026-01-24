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

# Download NLTK data required for text processing
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data'); nltk.download('punkt', download_dir='/usr/local/nltk_data')"
ENV NLTK_DATA=/usr/local/nltk_data

# copy rest of the directory into your container and make output folder
COPY . .

ENV MODEL_PORT=8081
ENV METRICS_PORT=9091

# expose the ports (API and Prometheus metrics)
EXPOSE 8081 9091

# run python command to start the microservice
ENTRYPOINT [ "bash", "train_model.sh" ]