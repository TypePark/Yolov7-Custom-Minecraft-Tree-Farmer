import os
import subprocess
import wandb

# Change the working directory to yolov7-main
yolov7_main_dir = ""  # Replace this with the actual path to the yolov7-main directory
os.chdir(yolov7_main_dir)

# Specify your wandb API key
wandb_api_key = ""

try:
    # Log in to wandb with your API key
    wandb.login(key=wandb_api_key)

    # Specify the terminal command to be executed
    command = "python train_aux.py --workers 8 --device 0 --batch-size 2 --data data/custom.yaml --img 1920 1080 --cfg cfg/training/yolov7-w6.yaml --weights 'yolov7-w6_training.pt' --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml --epochs 20"

    # Run the command in a new shell
    subprocess.run(command, shell=True, check=True)

except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")

except wandb.errors.UsageError as e:
    print(f"Error occurred during wandb login: {e}")
