# using some official python image as base
FROM python:3.12.9-slim

# navigate to the working directory
WORKDIR /app

# install system dependencies (if any python libs are out of date)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# check if the requirements change
COPY requirements.txt .

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of the directory into your container and make output folder
COPY . .
RUN mkdir -p output

# expose the port
EXPOSE 8081

# run python command to start the microservice
CMD ["python", "src/serve_model.py"]
