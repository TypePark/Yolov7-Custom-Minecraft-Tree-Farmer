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
model = torch.hub.load('WongKinYiu/yolov7', 'custom',  path_or_model='', force_reload=False)

# Set up the screen capture
sct = mss()
monitor = {"top": 0, "left": 0, "width": 1920,
           "height": 1080}  # Adjust the monitor values to capture the desired region

# Define the threshold distance
threshold_distance = 1

# Define the confidence score threshold
confscore = 0.1

boundbox = 100

# Define the duration for mouse button press (in seconds)
button_press_duration = 0.3

# Initialize variables for spacebar pressing
spacebar_pressed = False
spacebar_start_time = 0
spacebar_duration = 1.0  

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
            closest_distance = float('inf')
            closest_bbox_center_x = 0
            closest_bbox_center_y = 0
            closest_confidence = 0

            ai_x = monitor["width"] // 2
            ai_y = monitor["height"] // 2

            for detection in results.pred[0]:
                if detection[4] > confscore:
                    bbox_center_x = int((detection[0] + detection[2]) / 2)
                    bbox_center_y = int((detection[1] + detection[3]) / 2)
                    distance = math.sqrt((ai_x - bbox_center_x) ** 2 + (ai_y - bbox_center_y) ** 2)

                    if (detection[2] - detection[0]) > boundbox and (detection[3] - detection[1]) > boundbox:
                        if distance < threshold_distance:
                            continue

                        if distance < closest_distance:
                            closest_distance = distance
                            closest_bbox_center_x = bbox_center_x
                            closest_bbox_center_y = bbox_center_y
                            closest_confidence = detection[4]

            if closest_confidence > 0:
                dx = closest_bbox_center_x - ai_x
                dy = closest_bbox_center_y - ai_y
                angle = math.degrees(math.atan2(dy, dx))
                pydirectinput.moveTo(ai_x, ai_y)
                pydirectinput.moveRel(dx, dy, duration=0.2)

                # Press the left mouse button for {button_press_duration} second
                pyautogui.mouseDown(button='left')
                time.sleep(button_press_duration)
                pyautogui.mouseUp(button='left')

                if (detection[2] - detection[0]) <= boundbox or (detection[3] - detection[1]) <= boundbox:
                    print(f"Bounding box is smaller than {boundbox} x {boundbox} pixels. Pressing 'w'...")

                    # Press the 'w' key until the bounding box becomes larger or equal to boundbox
                    while (detection[2] - detection[0]) <= boundbox or (detection[3] - detection[1]) <= boundbox:
                        pydirectinput.keyDown('w')
                        time.sleep(0.1)  # Adjust the delay as needed
                        pydirectinput.keyUp('w')
                        sct_img = sct.grab(monitor)
                        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2RGB)
                        results = model(frame)
                        detection = results.pred[0][0]  # Update the detection

                    print("Bounding box size is now greater than or equal to", boundbox)

                    # Press the spacebar for 1 seconds if the bounding box dimensions are not increasing
                    if not spacebar_pressed:
                        spacebar_start_time = time.time()
                        spacebar_pressed = True
                    elif spacebar_pressed and (time.time() - spacebar_start_time) >= spacebar_duration:
                        pydirectinput.keyDown('w')
                        pydirectinput.keyDown('space')
                        print("Ai stuck, pressing spacebar...")
                        time.sleep(spacebar_duration)
                        pydirectinput.keyUp('space')
                        pydirectinput.keyUp('w')
                        spacebar_pressed = False

            else:
                print(f"There is no Oak_log present on the screen with a confidence score greater than {confscore}")

    except IndexError:
        print("IndexError: No detections found.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
