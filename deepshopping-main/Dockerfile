# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for dlib and opencv-python
# This is crucial to avoid compilation errors inside the container
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
# We copy this first to leverage Docker's layer caching
COPY ./requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Render will automatically use the PORT environment variable, which our app will read
# We tell Docker that the container listens on this port
EXPOSE 8000

# Run main.py when the container launches.
# Our main.py will be modified to contain the logic to start the uvicorn server.
CMD ["python", "-m", "api.main"] 