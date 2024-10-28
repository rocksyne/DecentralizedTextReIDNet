import cv2
import imagezmq
import sys, os, socket
import numpy as np
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath('../'))
from utils.video_stream import VideoFileStream
from config import sys_configuration

configs = sys_configuration()

def add_text_to_frame(frame, camera_id, timestamp):
    # Add camera name at the top left corner
    cv2.putText(frame, camera_id, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Add timestamp at the top right corner
    timestamp = "Sent: "+timestamp
    timestamp_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    timestamp_x = frame.shape[1] - timestamp_size[0] - 10
    cv2.putText(frame, timestamp, (timestamp_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

def send_video(server_ip, server_port, camera_name):
    sender = imagezmq.ImageSender(connect_to=f'tcp://{server_ip}:{server_port}')
    camera_stream = VideoFileStream(configs=configs, video_path=f"../data/sample_videos/{camera_name}.mp4", loop=True).start()
    
    try:
        while True:
            original, original_img_as_tensor, preprocessed = camera_stream.read()

            if not camera_stream.grabbed:
                break

            else:
                camera_output = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

                # Get the current timestamp
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                # Add text to the frame
                camera_output = add_text_to_frame(camera_output, camera_name, timestamp)

                sender.send_image(camera_name, camera_output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera_stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send video stream from camera.")
    parser.add_argument('camera_name', type=str, help='Name of the camera')
    
    args = parser.parse_args()
    camera_name = args.camera_name

    send_video(configs.command_station_address, configs.video_parsing_app_port, camera_name)  # Replace '127.0.0.1' and '5555' with the actual IP and port
