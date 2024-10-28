# System modules
import os, sys, time

# 3rd party modules
import torch
from tqdm import tqdm as tqdm
import numpy as np

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from config import sys_configuration
from utils.iotools import read_image
from model.unity import Unity
from utils.miscellaneous_utils import set_seed
from torchvision import transforms
from evaluation.evaluations import calculate_similarity
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

    token_ids = torch.load("token_ids.pt").to(configs.device)
    orig_token_length = torch.load("orig_token_length.pt").to(configs.device)

    

    # on first execution, this process takes long because of file loading. 
    # so we shall do the inference in and loop and then tahe the average. 
    times = []
    for counter, _time in enumerate(range(20)):
        start = time.time()
        with torch.no_grad():
            textual_embedding = model.target_persons_text_embeddings(token_ids, orig_token_length)
        end = time.time()

        text_time = end - start
        

        ranking_result = []
        
        for image_path in tqdm(image_list):
            
            original_img, original_img_as_tensor, processed_img = read_and_process_image(image_path)

            start = time.time()
            with torch.no_grad():
                persons_embeddings_out = model.persons_embeddings(original_img_as_tensor.to(configs.device), processed_img.to(configs.device))
            print(f"elapsed time is: ", time.time() - start)
            
            if persons_embeddings_out != None:
                image_embedding, bbox = persons_embeddings_out
                similarity = calculate_similarity(image_embedding,textual_embedding).numpy()[0][0]

                ranking_result.append([image_path,similarity])
            end = time.time()
            image_time = end - start
            times.append(image_time)

        print(f"Text:{text_time} Image:{np.mean(times)}")




    #     sorted_paths_and_scores = sorted(ranking_result, key=lambda x: x[1], reverse=True)

    #     for idx, path in enumerate(sorted_paths_and_scores[:5],start=1):
    #         print(f"{idx}. Path:{path[0]}  Score:{path[1]}")

    #     end = time.time()
    #     elapsed = end - start
    #     print("Total Time elapsed:",elapsed)
    #     times.append(elapsed)

    # # for debug
    # for loop, elapsed_time in enumerate(times):
    #     print(f"{loop}. {elapsed_time}")