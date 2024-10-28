import cv2
import os
from natsort import natsorted

def create_video_from_images(image_folder, output_video, fps=2):
    # Get list of all files in the directory
    files = os.listdir(image_folder)
    
    # Filter out only jpg images and sort them
    images = natsorted([img for img in files if img.endswith(".jpg")])

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 'XVID' can also be used for avi format
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through images and write to video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer
    video.release()

# Usage example
image_folder = '/media/rockson/Data_drive/Research/DecentralizedTextReIDNet/fromdeeplearning'
output_video = 'output_video.mp4'
create_video_from_images(image_folder, output_video)
