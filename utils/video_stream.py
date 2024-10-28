import time
from threading import Thread, Event
import cv2
from PIL import Image
from torchvision import transforms
from utils.mhpv2_dataset_utils import CustomImageResizer
from typing import Dict, List, Optional

class VideoFileStream:
    def __init__(self, 
                 configs: Dict, 
                 video_path: Optional[str] = None, 
                 name: str = "VideoFileStream", 
                 resolution: str = "VGA", 
                 fps: int = 30, 
                 sleep_time: float = 0.01,
                 loop: bool = False):
        """
        Initialize the video file stream with given configurations.

        Args:
            configs (dict): Configuration dictionary.
            video_path (str, optional): Path to the video file. Defaults to None.
            name (str): Thread name. Defaults to "VideoFileStream".
            resolution (str): Video resolution. Defaults to "VGA".
            fps (int): Frames per second. Defaults to 30.
            sleep_time (float): Sleep time between frame reads. Defaults to 0.01.
            loop (bool): Whether to loop the video endlessly. Defaults to False.
        """
        video_resolutions = {'QVGA': [320, 240], 'VGA': [640, 480], 'SVGA': [800, 600], 'HD': [1280, 720]}
        width, height = video_resolutions.get(resolution, video_resolutions['VGA'])
        self.stream = cv2.VideoCapture(video_path)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.sleep_time = sleep_time
        self.loop = loop

        self.grabbed, self.frame = self.stream.read()
        self.MHPv2_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=configs.get('MHPV2_means', [0.485, 0.456, 0.406]), 
                                 std=configs.get('MHPV2_stds', [0.229, 0.224, 0.225]))
        ])
        self.PIL_rgb_img_resizer = CustomImageResizer(
            dimensions=configs.get('MHPV2_image_size', [640, 480]), 
            image_type='RGB'
        )

        self.name = name
        self.stopped = Event()

    def start(self) -> 'VideoFileStream':
        """
        Start the thread to read frames from the video stream.

        Returns:
            VideoFileStream: The instance of the video file stream.
        """
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """
        Continuously read frames from the video stream.
        """
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        frame_duration = 1.0 / fps

        while not self.stopped.is_set():
            if not self.grabbed:
                if self.loop:
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
                    self.grabbed, self.frame = self.stream.read()
                    print("Restarting video stream")  # Debugging information
                else:
                    self.stop()
                    break
            else:
                start_time = time.time()
                self.grabbed, self.frame = self.stream.read()
                processing_time = time.time() - start_time
                sleep_time = max(frame_duration - processing_time, 0)
                time.sleep(sleep_time)

    def read(self) -> List:
        """
        Read the current frame and apply preprocessing.

        Returns:
            List: Original frame, original image as tensor, and preprocessed frame.
        """
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = self.PIL_rgb_img_resizer(frame)

        original = frame.copy()
        preprocessed = self.MHPv2_transform(frame).unsqueeze(0)
        original_img_as_tensor = transforms.ToTensor()(original)
        return [original, original_img_as_tensor, preprocessed]

    def stop(self):
        """
        Stop the video file stream.
        """
        self.stopped.set()

    def release(self):
        """
        Release the video stream resource.
        """
        self.stream.release()

    def __del__(self):
        """
        Destructor to ensure the video stream is released.
        """
        self.release()
