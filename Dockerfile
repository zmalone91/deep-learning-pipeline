# Use a lightweight python image
FROM python:3.10-slim

# Update package list & install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    python3-dev

# (Optional) install other dependencies for your project
# e.g. libssl-dev, libffi-dev if needed

# Copy requirements
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY . /app

# Set environment variables (optional)
# if you have an MLflow server running
# ENV MLFLOW_TRACKING_URI=http://my-mlflow-server:5000

# alternatively refer to the host machine  
# ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000  
# ENV MLFLOW_TRACKING_URI=http://localhost:5000


# Run the pipeline
CMD ["python", "run_pipeline.py"]
