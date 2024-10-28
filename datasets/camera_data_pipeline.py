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
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
from threading import Thread
from queue import Queue

from datasets.bases import place_image_on_canvas

# Global queues for frame processing and post-processing
processing_queue = Queue(maxsize=5)  # Adjust size as necessary
post_processing_queue = Queue(maxsize=5)  # Adjust size as necessary

# Define your transformations
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Asynchronous webcam stream class
class CameraStream:
    def __init__(self, 
                 config:dict=None, 
                 src:int=0):
        """
        Doc.:   Asynchronous ebcam Stream Reading
        Args.:  - src: source of the video stream. Default=0. First camera device.
                - configs: configurations
        """
        self.q = Queue()
        self.t = Thread(target=self.update, args=())
        self.configs = config
        video_resolutions = {'QVGA':[320,240], 'VGA':[640,480], 'SVGA':[800,600],'HD':[1280,720]} # {'res_key':[width, height]}
        width, height = video_resolutions[config.video_resolution]
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.t.daemon = True
        self.t.start()

        # pre_process
        self.transform = T.Compose([T.ToTensor(),T.Normalize(mean=config.MHPV2_means,std=config.MHPV2_stds)])

    def update(self):
        while True:
            if not self.q.full():
                ret, image_BGR = self.cap.read()
                if ret:
                    # perform pre_processing here
                    image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
                    image_RGB = Image.fromarray(image_RGB)
                    image_RGB = place_image_on_canvas(image_RGB,(512,512),"center")
                    preprocessed_img = self.transform(image_RGB).unsqueeze(0)
                    # print("The shape of aa is: ",preprocessed_img.shape)
                    self.q.put([image_RGB,preprocessed_img])

    def read(self):
        return self.q.get()

    def release(self):
        self.cap.release()

