# YOLO Detection Application

## System Requirements to Run Locally

### Hardware:
- **Minimum CPU**: Intel i5 or equivalent
- **Recommended GPU**: NVIDIA GTX 1060 or higher
- **RAM**: 8GB minimum (16GB recommended)

### Operating Systems:
- **Supported**: 
  - Windows 10 or later
  - macOS 10.15 or later
  - Ubuntu 18.04 or later

### Software Dependencies:
- **Python**: 3.8 or higher

## Getting Started Without Docker

### 1. Open Command Prompt
Navigate to the folder where you have unzipped `Yolo_code.zip`.

### 2. Install Required Python Libraries
Use the following command to install the required Python libraries:

```bash
pip install -r requirements.txt
```
### 3. Run the Flask Application
Once the necessary libraries are installed, stay in the same directory and execute the following command:

```bash
python app.py
```
Wait for a moment as Flask initializes. Once it's ready, you will see two ports displayed in the terminal. You can use these ports to access the final application through your preferred web browser.

## Getting Started With Docker

### 1. Ensure Docker is Installed
Make sure Docker Desktop is installed and running on your system before proceeding.

### 2. Open Command Prompt
Navigate to the folder where you have unzipped `Yolo_code.zip`.

### 3. Build the Docker Image
Use the following command to build the Docker image:

```bash
docker build -t yolotest .
```
The build process may take some time as it installs the required dependencies and combines files to create the Docker image.

### 4. Run the Docker Container
Once the build is complete, remain in the same directory and execute the following command:

```bash
docker run -p 5000:5000 yolotest
```
Wait a few moments as Docker initializes. Once ready, you will see two ports displayed. You can use these ports to access the final application through your web browser of choice.

## Overview of the User Interface

- **Model Selection**: Choose between YOLOv8n and the fine-tuned model
![Home Page](https://github.com/user-attachments/assets/3aac58bb-2ff5-4dde-9136-b5e997e8360e](https://github.com/mahimayadav97/YOLO-object-detection-deployment/blob/main/images/Home%20Page.png)
- **File Upload**: Button to browse and upload images/videos.
- 
- **Results Display**: Area where processed media with annotations will be displayed.
- 

## File Uploading and Processing

### Supported File Formats:
- **Images**: .jpg, .png
- **Videos**: .mp4, .avi

### Size Limitations:
- **Images**: Maximum 5MB
- **Videos**: Maximum 5MB
