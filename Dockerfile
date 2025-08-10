# Use base image with Python & TensorFlow
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Default command (can override in `docker run`)
CMD ["python", "train.py"]