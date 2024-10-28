"""
Credit: https://github.com/PyImageSearch/imutils/blob/master/imutils/video/webcamvideostream.py

Author: Rockson Ayeman (rockson.agyeman@aau.at, rocksyne@gmail.com)
        Bernhard Rinner (bernhard.rinner@aau.at)

For:    Pervasive Computing Group (https://nes.aau.at/?page_id=6065)
        Institute of Networked and Embedded Systems (NES)
        University of Klagenfurt, 9020 Klagenfurt, Austria.

Date:   Thursday 3rd Aug. 2023 (First authored date)

Documentation:
--------------------------------------------------------------------------------
Increasing webcam FPS with Python and OpenCV. 
See https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

TODO:   [x] Do proper documentation
        [x] Search where code base was gotten from and credit appropriately
"""

# import the necessary packages
from threading import Thread
import cv2
from queue import Queue
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
from utils.mhpv2_dataset_utils import CustomMHPv2ImageResizer



from datasets.bases import place_image_on_canvas


class WebcamVideoStream:
    def __init__(self, configs=dict, src=0, name="WebcamVideoStream", resolution="VGA"):

        video_resolutions = {'QVGA':[320,240], 'VGA':[640,480], 'SVGA':[800,600],'HD':[1280,720]} # {'res_key':[width, height]}
        
        width, height = video_resolutions[resolution]
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        (self.grabbed, self.frame) = self.stream.read()

        # pre-processing
        self.MHPv2_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=configs.MHPV2_means,std=configs.MHPV2_stds)])
        self.PIL_rgb_img_resizer = CustomMHPv2ImageResizer(dimensions=configs.MHPV2_image_size, image_type='RGB')

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        frame = cv2.flip(self.frame,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame )
        frame = self.PIL_rgb_img_resizer(frame)

        # split into partitions / original and pre-processed
        # original = np.array(frame)
        # original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        original = frame.copy()
        preprocessed = self.MHPv2_transform(frame).unsqueeze(0)

        return [original, preprocessed]

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def release(self):
        self.stream.release()