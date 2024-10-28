# System modules
import os, sys

# 3rd party modules
import torch
from PIL import ImageDraw
from torchvision import transforms

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from config import sys_configuration
from utils.iotools import read_image
from model.unity import Unity
from utils.mhpv2_dataset_utils import CustomMHPv2ImageResizer

# Global configurations
config = sys_configuration()
torch.multiprocessing.set_sharing_strategy('file_system')
unity_model = Unity(config).to(config.device).eval()
rgb_img_resizer = CustomMHPv2ImageResizer(dimensions=config.MHPV2_image_size, image_type='RGB')
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=config.MHPV2_means,std=config.MHPV2_stds)])


def read_and_process_image(path:str=None):
    """Read and pre-process image"""
    img = read_image(img_path=path, mode='RGB')
    processed_img = rgb_img_resizer(img)
    original_img = processed_img.copy()
    processed_img = transform (processed_img)
    processed_img = processed_img.unsqueeze(0) # make shape (1,3,H,W)
    original_img_as_tensor = transforms.ToTensor()(original_img)
    return original_img, original_img_as_tensor, processed_img
    

def draw_bounding_box_on_image(image, bounding_boxes):
    draw = ImageDraw.Draw(image)
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image


if __name__ == '__main__':
    
    img_path = "/media/rockson/Data_drive/Research/DecentralizedTextReIDNet/data/sample_images/Pervasive-Computing_June-2020.jpg"
    original_img, original_img_as_tensor, processed_img = read_and_process_image(img_path)

    with torch.no_grad():
        image_embedding, bboxes = unity_model.persons_embeddings(original_img_as_tensor.to(config.device), processed_img.to(config.device))
        img_with_bb = draw_bounding_box_on_image(original_img,bboxes)
        img_with_bb.show()





