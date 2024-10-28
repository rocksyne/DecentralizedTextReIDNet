""""
Doc.:   This script does reid on saved pedistrian video.py
"""


# System modules
import cv2
import os, sys
import random

import torch
import numpy as np
from PIL import ImageDraw
from PIL import ImageDraw, ImageFont

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from config import sys_configuration
from evaluation.evaluations import calculate_similarity
from datasets.custom_dataloader import process_text_into_tokens
from model.unity import Unity
from nano.nano_webcam import WebcamVideoStream

# Global configurations
configs = sys_configuration()
model = Unity(configs).to(configs.device)
model.eval()



def draw_bounding_boxes(image, bounding_boxes, confidence_scores, top_1=False):
    
    if not isinstance(bounding_boxes,list):
        bounding_boxes = [bounding_boxes]
    
    if not isinstance(confidence_scores,list):
        confidence_scores = [confidence_scores]

    if top_1:
        bounding_boxes = [bounding_boxes[0]]
        confidence_scores = [confidence_scores[0]]
    
    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define font for the confidence score
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for bbox, score in zip(bounding_boxes, confidence_scores):
        # Generate a random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Draw operations (same as before)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"Conf: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        draw.rectangle([x1, y1 - text_height, x1 + text_width + 10, y1], fill=color)
        draw.text((x1 + 5, y1 - text_height), text, fill="white", font=font)

    return image


# Function to process and display frames
def process_frames(camera, token_ids, orig_token_length):
    while True:
        if camera.grabbed is False:
            break

        original, original_img_as_tensor, preprocessed = camera.read()

        with torch.no_grad():
            persons_embeddings_out = model.persons_embeddings(original_img_as_tensor, preprocessed.to(configs.device))
            textual_embedding = model.target_persons_text_embeddings(token_ids, orig_token_length)
        
        if persons_embeddings_out is None:
            camera_output = np.array(original)
            camera_output = cv2.cvtColor(camera_output, cv2.COLOR_RGB2BGR)
            
    
        if persons_embeddings_out != None:
            image_embedding, bbox = persons_embeddings_out
            similarity = calculate_similarity(image_embedding,textual_embedding).numpy()

            # Sort the indeces of similarity score, from the highest score to the lowest
            indices_sorted = sorted(range(len(similarity)), key=lambda i: similarity[i], reverse=True)

            # now sort bounding box list
            ranked_bbs = [bbox[i].tolist() for i in indices_sorted]
            ranked_scores = [similarity[i].tolist()[0] for i in indices_sorted]

            filtered_boxes_with_scores = [(bbox, score) for bbox, score in zip(ranked_bbs, ranked_scores) if score >= 0.10]#configs.reID_confidence_threshold]

            if len(filtered_boxes_with_scores)>0:
                filtered_boxes, filtered_scores = zip(*filtered_boxes_with_scores)
                filtered_boxes, filtered_scores = list(filtered_boxes), list(filtered_scores)

                camera_output = draw_bounding_boxes(original,filtered_boxes,filtered_scores,top_1=True)
                camera_output = np.array(camera_output)
                camera_output = cv2.cvtColor(camera_output, cv2.COLOR_RGB2BGR)
            
            else:
                camera_output = np.array(original)
                camera_output = cv2.cvtColor(camera_output, cv2.COLOR_RGB2BGR)

        else:
            camera_output = np.array(original)
            camera_output = cv2.cvtColor(camera_output, cv2.COLOR_RGB2BGR)


        cv2.imshow('Frame', camera_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':

    camera_stream = WebcamVideoStream(configs=configs,src=0).start() # read the usb camera
    textual_description = "The man is wearing a blue inner shirt, a grey coat and blue jeans"
    token_ids, orig_token_length = process_text_into_tokens(textual_description)
    token_ids = token_ids.unsqueeze(0).to(configs.device).long()
    orig_token_length = torch.tensor([orig_token_length]).to(configs.device)

    try:
        process_frames(camera_stream, token_ids, orig_token_length)

    finally:
        camera_stream.release()
        cv2.destroyAllWindows()



    