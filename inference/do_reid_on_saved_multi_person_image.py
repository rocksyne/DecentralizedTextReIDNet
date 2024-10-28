# System modules
import os, sys, random

# 3rd party modules
import torch
import numpy as np
from PIL import ImageDraw, ImageFont

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from config import sys_configuration
from utils.iotools import read_image
from model.unity import Unity
from utils.miscellaneous_utils import set_seed
from torchvision import transforms
from datasets.custom_dataloader import process_text_into_tokens
from evaluation.evaluations import calculate_similarity
from PIL import ImageDraw
from utils.mhpv2_dataset_utils import CustomMHPv2ImageResizer

# Global configurations
configs = sys_configuration()
set_seed(configs.seed) # using same as https://github.com/xx-adeline/MFPE/blob/main/src/train.py 
MHPv2_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=configs.MHPV2_means,std=configs.MHPV2_stds)])
rgb_img_resizer = CustomMHPv2ImageResizer(dimensions=configs.MHPV2_image_size, image_type='RGB')

# Pretrained model loading and stuff
model = Unity(configs).to(configs.device)
model.eval()

print("")
print('[INFO] Total parameters: {} million'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
print("")


# def read_and_process_image(path:str=None):
#     """Read and pre-process image"""
#     img = read_image(img_path=path, mode='RGB')
#     processed_img = rgb_img_resizer(img)
#     original_img = processed_img.copy()
#     processed_img = MHPv2_transform(processed_img)
#     processed_img = processed_img.unsqueeze(0) # make shape (1,3,H,W)
#     return original_img, processed_img


def read_and_process_image(path:str=None):
    """Read and pre-process image"""
    img = read_image(img_path=path, mode='RGB')
    processed_img = rgb_img_resizer(img)
    original_img = processed_img.copy()
    processed_img = MHPv2_transform(processed_img)
    processed_img = processed_img.unsqueeze(0) # make shape (1,3,H,W)

    # original image as tensor
    original_img_as_tensor = transforms.ToTensor()(original_img)
    return original_img, original_img_as_tensor, processed_img
    


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





if __name__ == '__main__':
    
    #img_path = "../data/sample_images/Pervasive-Computing_June-2020.jpg"
    #textual_description = "The man is wearing a blue long sleeved shirt, black belt, khaki pants and black shoes. He is also wearing eye glasses."
    img_path = "/home/rockson/Desktop/officerfemalestairs.jpg"
    textual_description = "The woman is a black long-sleeved shirt, black pants, and black shoes."
   
    original_img, original_img_as_tensor, processed_img = read_and_process_image(img_path)

    token_ids, orig_token_length = process_text_into_tokens(textual_description)
    token_ids = token_ids.unsqueeze(0).to(configs.device).long()
    orig_token_length = torch.tensor([orig_token_length]).to(configs.device)


    with torch.no_grad():
        persons_embeddings_out = model.persons_embeddings(original_img_as_tensor.to(configs.device), processed_img.to(configs.device))
        textual_embedding = model.target_persons_text_embeddings(token_ids, orig_token_length)
    
    if persons_embeddings_out != None:
        image_embedding, bbox = persons_embeddings_out
        similarity = calculate_similarity(image_embedding,textual_embedding).numpy()

        # Sort the indeces of similarity score, from the highest score to the lowest
        indices_sorted = sorted(range(len(similarity)), key=lambda i: similarity[i], reverse=True)

        # now sort bounding box list
        ranked_bbs = [bbox[i].tolist() for i in indices_sorted]
        ranked_scores = [similarity[i].tolist()[0] for i in indices_sorted]

        filtered_boxes_with_scores = [(bbox, score) for bbox, score in zip(ranked_bbs, ranked_scores) if score >= 0.01]#configs.reID_confidence_threshold]

        if len(filtered_boxes_with_scores)>0:
            filtered_boxes, filtered_scores = zip(*filtered_boxes_with_scores)
            filtered_boxes, filtered_scores = list(filtered_boxes), list(filtered_scores)

            original_img = draw_bounding_boxes(original_img,filtered_boxes,filtered_scores,top_1=True)
    
    original_img.show()


search_from_multi_person_camera_feed.jpg
