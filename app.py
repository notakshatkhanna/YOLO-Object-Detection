import cv2
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, render_template, request, jsonify, make_response
import time
import torch
import torchvision.transforms as transforms
import io
import base64


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))

    # Perform object detection using YOLOv5
    results = model(image)
    boxes = results.xyxy[0].tolist()
    labels = results.names[0]

    # Convert the image to base64 for display in HTML
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Render the result in HTML
    return render_template('result.html', img_str=img_str, boxes=boxes, labels=labels, zip=zip)


# Define a function to draw bounding boxes and labels on an image
def draw_boxes(image, boxes, labels):
    # Convert the image to a NumPy array
    img = np.array(image)

    # Loop over the bounding boxes and labels
    for box, label in zip(boxes, labels):
        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = map(int, box)

        # Draw the bounding box on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the label on the image
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the NumPy array back to an image
    img = Image.fromarray(img)

    # Return the annotated image
    return img


# Define a function to run the object detection model on an image
def detect(image):
    # Convert the image to a NumPy array
    img = np.array(image)

    # Run the object detection model
    results = model(img)

    # Extract the bounding boxes and labels from the results
    boxes = results.xyxy[0].tolist()
    labels = results.names[results.xyxy[0][:, -1].long().tolist()]

    # Draw the bounding boxes and labels on the image
    img = draw_boxes(img, boxes, labels)

    # Convert the NumPy array back to an image
    img = Image.fromarray(img)

    # Return the image
    return img
