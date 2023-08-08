import torch
import numpy as np
import cv2
import time
from mss import mss
from PIL import Image

# Load the YOLOv7 model from Torch Hub
model = torch.hub.load('WongKinYiu/yolov7', 'custom',  path_or_model='C:/Users/lolil/.cache/torch/hub/WongKinYiu_yolov7_main/runs/train/yolov7-custom32/weights/last.pt', force_reload=False)

# Set up the screen capture
sct = mss()
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Adjust the monitor values to capture the desired region

while True:
    # Capture the screen frame
    sct_img = sct.grab(monitor)
    frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2RGB)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make detections
    results = model(frame)

    # Render the detections on the frame
    rendered_frame = np.squeeze(results.render())

    # Display the frame with detections
    cv2.imshow('YOLO', rendered_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
