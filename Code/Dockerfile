FROM python:3.10.0


# Set the working directory in the container.
WORKDIR /app


# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0


# Copy the requirements.txt file into the container.
COPY requirements.txt .


# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt


# Copy the current directory contents into the container at /app.
COPY . /app


# Expose the port Flask app runs on
EXPOSE 8080


# Run the application.
CMD ["python", "app.py"]
