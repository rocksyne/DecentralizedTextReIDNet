import imagezmq
import cv2
import os, sys

# Application modules and configurations
sys.path.insert(0, os.path.abspath('../'))
from datasets.custom_dataloader import process_text_into_tokens
from config import sys_configuration
configs = sys_configuration()

# Initialize the ImageHub
address = f"tcp://{configs.command_station_address}:{configs.video_parsing_app_port}"
image_hub = imagezmq.ImageHub(open_port=address, REQ_REP=True)
image_hub = imagezmq.ImageHub()

while True:
    # Receive an image
    sender_name, image = image_hub.recv_image()
    
    # Display the image
    cv2.imshow(sender_name, image)
    
    # Send an acknowledgment back to the sender
    image_hub.send_reply(b'OK')
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
