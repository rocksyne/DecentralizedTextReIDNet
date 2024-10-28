import cv2
import sys
import os
import subprocess
import threading
import time
from PIL import Image
import numpy as np
from torchvision import transforms

sys.path.insert(0, os.path.abspath('../'))
from utils.mhpv2_dataset_utils import CustomImageResizer

class WebcamVideoStream:
    def __init__(self, src=0, configs: dict = None, resolution="VGA", frame_rate: int = 30):
        self.src = src
        self.configs = configs
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.frame = None
        self.grabbed = False
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        
        video_resolutions = {'QVGA': [320, 240], 'VGA': [640, 480], 'SVGA': [800, 600], 'HD': [1280, 720]}  # {'res_key':[width, height]}
        
        # Camera set-up
        subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=focus_automatic_continuous=0"]) #Disable autofocus
        subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=focus_absolute=0"]) #Set focus manually (adjust the value as needed)
        self.cap = cv2.VideoCapture(self.src)

        try:
            # Check if camera opened successfully
            if not self.cap.isOpened():
                raise Exception("Could not open video device")

            # Video frame properties
            width, height = video_resolutions[resolution]
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 to disable auto-focus
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

            # Transformations and functions
            self.grabbed, self.frame = self.cap.read()  # initial read
            if not self.grabbed:
                raise Exception("Failed to read initial frame from video device")

            self.MHPv2_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=configs.get('MHPV2_means', [0.485, 0.456, 0.406]),
                                     std=configs.get('MHPV2_stds', [0.229, 0.224, 0.225]))
            ])
            self.PIL_rgb_img_resizer = CustomImageResizer(dimensions=configs.get('MHPV2_image_size', [640, 480]), image_type='RGB')

            self.thread.start()
        except Exception as e:
            self.cleanup()
            print(f"Error during initialization: {e}")
            sys.exit(1)

    def update(self):
        try:
            while self.running:
                if self.cap.isOpened():
                    start_time = time.time()
                    self.grabbed, self.frame = self.cap.read()
                    if not self.grabbed:
                        raise Exception("Failed to read frame from video device")
                    # Calculate the time to sleep to achieve the desired frame rate
                    elapsed_time = time.time() - start_time
                    sleep_time = max(1.0 / self.frame_rate - elapsed_time, 0)
                    time.sleep(sleep_time)
        except Exception as e:
            print(f"Error during frame update: {e}")
            self.stop()

    def read(self):
        try:
            if self.frame is None:
                raise Exception("No frame available")
            
            frame = cv2.flip(self.frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.PIL_rgb_img_resizer(frame)

            # split into partitions / original and pre-processed
            original = frame.copy()
            preprocessed = self.MHPv2_transform(frame).unsqueeze(0)
            original_img_as_tensor = transforms.ToTensor()(original)
            return self.grabbed, original, original_img_as_tensor, preprocessed
        
        except Exception as e:
            print(f"Error during frame read: {e}")
            return False, None, None, None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cleanup()

    def cleanup(self):
        if self.cap.isOpened():
            self.cap.release()

    def set_frame_rate(self, frame_rate):
        self.frame_rate = frame_rate

if __name__ == "__main__":
    # Example usage
    try:
        sys.path.insert(0, os.path.abspath('../'))
        from config import sys_configuration
        configs = sys_configuration()
        cam = WebcamVideoStream(configs=configs)

        while True:
            grabbed, frame, _, _ = cam.read()
            if grabbed:
                # Convert PIL image back to numpy array for display
                frame_np = np.array(frame)
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV display
                cv2.imshow('Webcam', frame_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.stop()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in main: {e}")
        if 'cam' in locals():
            cam.stop()
        cv2.destroyAllWindows()
        sys.exit(1)
