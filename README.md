# Yolov7-Custom-Minecraft-Tree-Farmer
YOLOv7-based Minecraft tree (oak) farmer with custom tree detection.

# Necessary libraries
-  torch
-  numpy 
-  cv2
-  time
-  mss
-  PIL 
-  pydirectinput
-  math
-  pyautogui
-  os
-  subprocess
-  wandb

# Steps
- Annotate your images (I used ImgLabel). Use only 1 label (Oak_log) or you'll have to make significant changes to the code.
  
- Indicate the paths of the YOLOv7 folder, then execute Oak_logTrainer.py. (You need a GPU with at least 16GB of VRAM for images with dimensions of 1920 x 1080;
otherwise, you will encounter an error. Train with smaller dimensions if you encounter this issue.)

- Once training is completed, run Oak_logFarmer.py.

