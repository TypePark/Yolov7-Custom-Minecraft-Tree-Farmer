import torch
import numpy as np
import cv2
import time
from mss import mss
from PIL import Image
import pydirectinput
import math
import pyautogui

# Load the YOLOv7 model from Torch Hub
model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='C:/Users/lolil/.cache/torch/hub/WongKinYiu_yolov7_main/runs/train/yolov7-custom32/weights/last.pt', force_reload=False)

# Set up the screen capture
sct = mss()
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Adjust the monitor values to capture the desired region

# Initialize variables for left-click holding and rotation
holding_click = False
click_start_time = 0
rotation_angle = 0
rotate_enabled = True
last_detection_time = time.time()
stop_looking = False  # Flag to indicate whether the AI should stop looking at the oak logs

while True:
    # Capture the screen frame
    sct_img = sct.grab(monitor)
    frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2RGB)

    # Make detections
    results = model(frame)

    try:
        # Render the detections on the frame
        rendered_frame = np.squeeze(results.render())

        # Display the frame with detections
        cv2.imshow('YOLO', rendered_frame)

        if len(results.pred) > 0:
            # Find the closest bounding box
            closest_distance = float('inf')  # Initialize with a very large value
            closest_bbox_center_x = 0  # Initialize with a default value
            closest_bbox_center_y = 0  # Initialize with a default value

            ai_x = monitor["width"] // 2
            ai_y = monitor["height"] // 2

            for detection in results.pred[0]:
                # Calculate the center of the bounding box
                bbox_center_x = int((detection[0] + detection[2]) / 2)
                bbox_center_y = int((detection[1] + detection[3]) / 2)

                # Calculate the distance between AI and the bounding box center
                distance = math.sqrt((ai_x - bbox_center_x) ** 2 + (ai_y - bbox_center_y) ** 2)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_bbox_center_x = bbox_center_x
                    closest_bbox_center_y = bbox_center_y

            # Rotate the AI to face the closest bounding box if stop_looking is False
            if not stop_looking:
                dx = closest_bbox_center_x - ai_x
                dy = closest_bbox_center_y - ai_y
                angle = math.degrees(math.atan2(dy, dx))
                pydirectinput.moveTo(ai_x, ai_y)  # Move the mouse to AI's position
                pydirectinput.moveRel(dx, dy, duration=0.2)  # Move the mouse relative to AI's position to rotate

            # Check if the closest bounding box covers 25% of the screen
            bbox_width = closest_bbox_center_x - results.pred[0][0][0]
            bbox_height = closest_bbox_center_y - results.pred[0][0][1]
            bbox_area = bbox_width * bbox_height
            screen_area = monitor["width"] * monitor["height"]
            bbox_coverage = bbox_area / screen_area

            min_threshold = 0.08  # Minimum threshold value
            max_threshold = 0.15  # Maximum threshold value
            if min_threshold <= bbox_coverage <= max_threshold:
                # Perform actions when the bounding box coverage is within the specified range

                # Stop looking at the oak logs
                stop_looking = True

                # Reset rotation angle
                rotation_angle = 0

                # Stop walking if the oak log covers 25% of the screen
                pydirectinput.keyUp('w')
                pydirectinput.keyUp('a')
                pydirectinput.keyUp('d')
                if not holding_click:
                    # Start holding the left click for 3 seconds
                    pydirectinput.mouseDown()
                    holding_click = True
                    click_start_time = time.time()
                else:
                    # Release the left click after 3 seconds and reset variables
                    current_time = time.time()
                    if current_time - click_start_time >= 2:
                        pydirectinput.mouseUp()
                        holding_click = False
                        last_detection_time = time.time()
                        rotate_enabled = True
            else:
                # Continue walking and rotating if the oak log does not cover 25% of the screen
                if not holding_click:
                    pydirectinput.keyDown('w')
                if holding_click:
                    # Release the left click after 3 seconds and reset variables
                    current_time = time.time()
                    if current_time - click_start_time >= 2:
                        pydirectinput.mouseUp()
                        holding_click = False
                        last_detection_time = time.time()
                        stop_looking = False

    except IndexError:
        print("IndexError: No detections found.")
        pydirectinput.keyUp('w')
        x_offset = 16  # Number of pixels to move along the X-axis
        current_x, current_y = pyautogui.position()
        new_x = current_x + x_offset
        pyautogui.moveTo(new_x, current_y, duration=0.5)
        time.sleep(1)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
