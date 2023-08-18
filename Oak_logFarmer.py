import torch
import pydirectinput
import math
import pyautogui
import cv2
import time
import numpy as np
from mss import mss
from PIL import Image

weights_path = ''

# Loads the YOLOv7 model from Torch Hub
model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model=weights_path, force_reload=False)

sct = mss() # Sets up the screen capture
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Adjusts the monitor values to capture the desired region

# Defines the confidence score threshold
confscore = 0.15

# Defines the minimun bounding box value to break Oak_log (in pixels)
minboundbox = 100

# Defines the duration for mouse button press (in seconds)
button_press_duration = 1

# Initializes variables for spacebar pressing
spacebar_pressed = False
spacebar_start_time = 0
spacebar_duration = 0.1

while True:
    # Captures the screen frame
    sct_img = sct.grab(monitor)
    frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2RGB)

    # Makes detections
    results = model(frame)

    try:
        # Renders the detections on the frame
        rendered_frame = np.squeeze(results.render())

        # Displays the frame with detections
        cv2.imshow('TreeDetection', rendered_frame)

        if len(results.pred) > 0:
            for detection in results.pred[0]:
                if detection[4] > confscore:
                    bbox_center_x = int((detection[0] + detection[2]) / 2)
                    bbox_center_y = int((detection[1] + detection[3]) / 2)

                    if (detection[2] - detection[0]) > minboundbox and (detection[3] - detection[1]) > minboundbox:
                        ai_x = monitor["width"] // 2
                        ai_y = monitor["height"] // 2
                        dx = bbox_center_x - ai_x
                        dy = bbox_center_y - ai_y
                        angle = math.degrees(math.atan2(dy, dx))
                        pydirectinput.moveTo(ai_x, ai_y)
                        pydirectinput.moveRel(dx, dy, duration=0.2)

                        # Presses the left mouse button for {button_press_duration} second
                        pyautogui.mouseDown(button='left')
                        time.sleep(button_press_duration)
                        pyautogui.mouseUp(button='left')

                    if (detection[2] - detection[0]) <= minboundbox or (detection[3] - detection[1]) <= minboundbox:
                        print(f"Bounding box is smaller than {minboundbox} x {minboundbox} pixels. Pressing 'w'")

                        # Presses the 'w' key until the bounding box becomes larger or equal to minboundbox
                        while (detection[2] - detection[0]) <= minboundbox or (detection[3] - detection[1]) <= minboundbox:
                            pydirectinput.keyDown('w')
                            time.sleep(0.05)
                            pydirectinput.keyUp('w')
                            sct_img = sct.grab(monitor)
                            frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2RGB)
                            results = model(frame)
                            detection = results.pred[0][0]  # Updates the detection

                            print(f"Bounding box size is now greater than or equal to: {minboundbox}")

                            # Presses the spacebar if the bounding box dimensions are not increasing
                            if not spacebar_pressed:
                                spacebar_start_time = time.time()
                                spacebar_pressed = True
                            elif spacebar_pressed and (time.time() - spacebar_start_time) >= spacebar_duration:
                                pydirectinput.keyDown('w')
                                pydirectinput.keyDown('space')
                                print("Ai stuck, pressing spacebar")
                                time.sleep(spacebar_duration)
                                pydirectinput.keyUp('space')
                                pydirectinput.keyUp('w')
                                spacebar_pressed = False

        else:
            print(f"There is no Oak_log present on the screen with a confidence score greater than {confscore}")

    except IndexError:
        print("IndexError: There is no Oak_log present on the screen")
        pydirectinput.moveRel(10, 0, duration=0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
