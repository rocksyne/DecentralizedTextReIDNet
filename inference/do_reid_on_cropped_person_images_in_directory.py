# System modules
import os, sys, random, time

# 3rd party modules
import torch
import numpy as np
from PIL import ImageDraw, ImageFont
from tqdm import tqdm as tqdm

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
    


def list_images_in_directory(directory):
    # Define image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    files = os.listdir(directory)
    images = [os.path.join(directory,file) for file in files if file.lower().endswith(image_extensions)]
    return images


if __name__ == '__main__':
    
    image_parent_directory = "../data/sample_gallery"
    image_list = list_images_in_directory(image_parent_directory)

    textual_description = "The woman has long black hair. She is wearing a pink dress with a white sweater and black shoes."
    
    token_ids, orig_token_length = process_text_into_tokens(textual_description)
    token_ids = token_ids.unsqueeze(0).to(configs.device).long()
    orig_token_length = torch.tensor([orig_token_length]).to(configs.device)

    with torch.no_grad():
        textual_embedding = model.target_persons_text_embeddings(token_ids, orig_token_length)
    
    ranking_result = []
    for image_path in tqdm(image_list):
        original_img, original_img_as_tensor, processed_img = read_and_process_image(image_path)

        with torch.no_grad():
            persons_embeddings_out = model.persons_embeddings(original_img_as_tensor.to(configs.device), processed_img.to(configs.device))
        
        if persons_embeddings_out != None:
            image_embedding, bbox = persons_embeddings_out
            similarity = calculate_similarity(image_embedding,textual_embedding).numpy()[0][0]

            ranking_result.append([image_path,similarity])


    sorted_paths_and_scores = sorted(ranking_result, key=lambda x: x[1], reverse=True)

    for idx, path in enumerate(sorted_paths_and_scores[:10],start=1):
        print(f"{idx}. Path:{path[0]}  Score:{path[1]}")
