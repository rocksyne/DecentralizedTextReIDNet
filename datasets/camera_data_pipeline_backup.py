"""
C. 2024. Rockson Agyeman and Bernhard Rinner

Doc.:   Data pipeline for reading data from camera
        This script handles webcam image stream reading and pre-processing
        and then passes these images to a neural network for inference in a 
        way that does not block the main execution thread.
"""
# 3rd party modules
import cv2
import torch
import numpy as np
from torchvision import transforms
from threading import Thread
from queue import Queue

# Global queues for frame processing and post-processing
processing_queue = Queue(maxsize=5)  # Adjust size as necessary
post_processing_queue = Queue(maxsize=5)  # Adjust size as necessary

# Define your transformations
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=config.MHPV2_means,std=config.MHPV2_stds)])

# Asynchronous webcam stream class
class CameraStream:
    def __init__(self, src:int=0):
        """
        Doc.:   Asynchronous ebcam Stream Reading
        Args.:  - src: source of the video stream. Default=0. First camera device.
        """
        self.cap = cv2.VideoCapture(src)
        self.q = Queue()
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

    def update(self):
        while True:
            if not self.q.full():
                ret, frame = self.cap.read()
                if ret:
                    self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.cap.release()

# Function to process and display frames
def process_frames(stream, model):
    while True:
        frame = stream.read()
        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        with torch.no_grad():  # Inference
            output = model(input_batch)

        # Post-processing: You can add your post-processing here
        # For example, drawing bounding boxes, etc.
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Main function to start the program
def main():
    stream = CameraStream()
    try:
        process_frames(stream, model)
    finally:
        stream.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
