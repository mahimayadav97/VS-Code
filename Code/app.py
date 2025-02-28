from flask import Flask, request, render_template, redirect, url_for, send_from_directory, session, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from sort import Sort  # Import the Sort class from sort.py
from ultralytics import YOLO
import math
import cvzone
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Define class names for both models
classNames_leaf = ["leaf"]
classNames_general = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Path to the fine-tuned YOLOv8 model (inside Docker)
model_path_leaf = os.getenv('MODEL_PATH', './model/best.pt')

@app.route('/')
def select_model():
    return render_template('select_model.html')

@app.route('/upload/<model_type>')
def upload_file(model_type):
    return render_template('upload.html', model_type=model_type)

@app.route('/uploader/<model_type>', methods=['GET', 'POST'])
def upload_file_route(model_type):
    try:
        if request.method == 'POST':
            file = request.files['file']
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                if is_video(file_path):
                    if model_type == 'leaf':
                        processed_path = process_video_leaf(file_path)
                    else:
                        processed_path = process_video_general(file_path)
                    return redirect(url_for('uploaded_file', filename=os.path.basename(processed_path), model_type=model_type))
                else:
                    processed_path, object_counts = process_image(file_path, model_type)
                    session['object_counts'] = json.dumps(object_counts)
                    return redirect(url_for('uploaded_file', filename=os.path.basename(processed_path), model_type=model_type))
        return render_template('upload.html')
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('error_page'))

def is_video(file_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

# Process video using the model 'model.pt'
def process_video_leaf(file_path):
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    cap = cv2.VideoCapture(file_path)

    model = YOLO(model_path_leaf)  # Load fine-tuned model for leaves
    classNames = classNames_leaf

    output_file_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_' + os.path.basename(file_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    object_counts = {}

    while True:
        ret, img_bgr = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for YOLO processing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Perform YOLO inference in RGB
        results = model(img_rgb, stream=False)

        # Convert back to BGR for display/output
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Manually draw bounding boxes and labels
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Draw the bounding box on the image
                color = (0, 255, 0)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

                # Put the class label and confidence
                label = f"{classNames[cls]} {conf:.2f}"
                cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Object counting
                if classNames[cls] not in object_counts:
                    object_counts[classNames[cls]] = 0
                object_counts[classNames[cls]] += 1

        out.write(img_bgr)

    cap.release()
    out.release()
    return output_file_path

# Process video using the general YOLOv8n model 'yolov8n.pt'
def process_video_general(file_path):
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    cap = cv2.VideoCapture(file_path)

    model = YOLO("yolov8n.pt")
    classNames = classNames_general

    output_file_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_' + os.path.basename(file_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    object_counts = {}
    alert_threshold = 5

    while True:
        ret, img = cap.read()
        if not ret:
            break

        results = model(img, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
               
                cls = int(box.cls[0])
                conf = math.ceil(box.conf[0] * 100) / 100
                
                if conf > 0.5:
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (x2, y2), scale=1, thickness=1, colorR=(0, 0, 255))
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
                   
                    # Count objects
                    if classNames[cls] not in object_counts:
                        object_counts[classNames[cls]] = 0
                    object_counts[classNames[cls]] += 1

        resultTracker = tracker.update(detections)

        for res in resultTracker:
            x1, y1, x2, y2, id = res
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            w, h = x2 - x1, y2 - y1
            cvzone.putTextRect(img, f'ID: {id}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))

        # Alert if any object count exceeds threshold
        for obj, count in object_counts.items():
            if count > alert_threshold:
                print(f"Alert: High number of {obj}s detected ({count})")

        out.write(img)

    cap.release()
    out.release()
    return output_file_path

def process_image(file_path, model_type):
    img_bgr = cv2.imread(file_path)
    if model_type == 'leaf':
        # Convert BGR to RGB for YOLO processing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Load YOLO model
        model = YOLO(model_path_leaf)
        classNames = classNames_leaf
        
        # Perform YOLO inference in RGB
        results = model(img_rgb, stream=False)
        
        # Convert the result image back to BGR for saving
        result_img_bgr = results[0].plot()
    else:
        model = YOLO("yolov8n.pt")
        classNames = classNames_general
        results = model(img_bgr, stream=False)
        result_img_bgr = results[0].plot()

    output_file_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_' + os.path.basename(file_path))
    cv2.imwrite(output_file_path, result_img_bgr)

    object_counts = {}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if classNames[cls] not in object_counts:
                object_counts[classNames[cls]] = 0
            object_counts[classNames[cls]] += 1

    return output_file_path, object_counts

@app.route('/processed/<filename>')
def uploaded_file(filename):
    object_counts = json.loads(session.get('object_counts', '{}'))
    return render_template('uploaded.html', filename=filename, object_counts=object_counts)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="404 - Page Not Found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', message="500 - Internal Server Error"), 500

@app.route('/error')
def error_page():
    return render_template('error.html', message="An unexpected error occurred. Please try again.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
