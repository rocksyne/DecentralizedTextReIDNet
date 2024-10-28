"""
Doc.:   Codebase adapted from https://github.com/anosorae/IRRA/tree/main
"""

import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.iotools import read_image
from utils.miscellaneous_utils import pad_tokens
from datasets.tiktoken_tokenizer import TikTokenizer
from datasets.simple_tokenizer import SimpleTokenizer
from datasets.bert_tokenizer import BERTTokenizer



def place_image_on_canvas_old(img):

    # Get the size of the image
    width, height = img.size
    
    # Create a 512x512 black canvas
    canvas = Image.new('RGB', (512, 512), (0, 0, 0))
    
    # If the height is greater than 512, resize the image
    if height > 512:
        new_height = 512
        new_width = int((new_height / height) * width)
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
    # Calculate random position to place the image
    max_x = max(512 - img.width, 0)  # Ensure the image is placed within the canvas
    max_y = max(512 - img.height, 0)  # Ensure the image is placed within the canvas
    random_x = random.randint(0, max_x)
    random_y = random.randint(0, max_y)
    
    # Paste the image onto the canvas
    canvas.paste(img, (random_x, random_y))
    return canvas


def place_image_on_canvas(image, new_size=(512, 512), placement_strategy='random'):
    """
    Places an image on a larger canvas of new_size. Resizes the image if its height is greater than 512,
    maintaining the aspect ratio relative to the height.

    Args:
        image (PIL.Image): The original image.
        new_size (tuple): The desired size (width, height).
        placement_strategy (str): 'random' or 'center' for image placement.

    Returns:
        PIL.Image: The image placed on the new canvas.
    """

    # Resize the image if its height is greater than 512, maintaining aspect ratio
    if image.height > 512:
        # Calculate new dimensions maintaining aspect ratio
        aspect_ratio = image.width / image.height
        new_height, _ = new_size
        new_width = int(aspect_ratio * new_height)
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a new canvas with the desired size and black background
    canvas = Image.new('RGB', new_size, (0, 0, 0))

    # Calculate the position
    if placement_strategy == 'center':
        # Place the image at the center of the canvas
        x = (new_size[0] - image.width) // 2
        y = (new_size[1] - image.height) // 2
    elif placement_strategy == 'random':
        # Place the image at a random position within the canvas boundaries
        x = np.random.randint(0, max(1, new_size[0] - image.width))
        y = np.random.randint(0, max(1, new_size[1] - image.height))
    else:
        raise ValueError("Invalid placement_strategy: choose 'random' or 'center'")

    # Paste the image onto the canvas
    canvas.paste(image, (x, y))

    return canvas



class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 tokens_length_max: int = 100, 
                 tokenizer_type:str="bert"):
        
        self.dataset = dataset
        self.transform = transform
        self.tokens_length_max = tokens_length_max
        self.tokenizer_type = tokenizer_type

        # create the appropriate tokenizer type
        if tokenizer_type == "bert":
            self.tokenizer = BERTTokenizer()

        elif tokenizer_type == "simple_tokenizer":
            self.tokenizer = SimpleTokenizer()
        
        elif tokenizer_type == "tiktoken_cl100k":
            self.tokenizer = TikTokenizer(encoding_base='cl100k_base')

        elif tokenizer_type == "tiktoken_p50k":
            self.tokenizer = TikTokenizer(encoding_base='p50k_base')
        
        elif tokenizer_type == "tiktoken_r50k":
            self.tokenizer = TikTokenizer(encoding_base='r50k_base')
        
        else:
            raise NotImplemented("No implemetation for `{}` tokenization type")
            

        
    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        pid = torch.from_numpy(np.array([pid])).long()
        img = read_image(img_path)
        img = place_image_on_canvas(img)
        original_img = img.copy() # copy to get the original image
        if self.transform is not None:
            img = self.transform(img)

        # convert original image to tensor without scaling to [0,1],
        # Tensor divides by pixels by 255 by the way. We will avoid that.
        # Reason? Keeping things simple for now until I figure somethings out, design-wise.
        original_img = np.array(original_img) # (H,W,C)
        original_img = torch.from_numpy(original_img).permute(2, 0, 1).to(torch.uint8) #  #(C, H, W)

        tokens = self.tokenizer(caption) # eg. torch.tensor([1165, 13, 564, 74, ..., 1167])
        token_ids, orig_token_length  = pad_tokens(tokens, self.tokens_length_max)

        ret = {'pids': pid,
               'image_ids': image_id,
               'img_paths': img_path,
               'preprocessed_images': img,
               'original_images':original_img,
               'token_ids': token_ids.to(torch.long),
               'orig_token_lengths': orig_token_length,
               'captions':caption}

        return ret


    


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        img = place_image_on_canvas(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, img_path


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 tokenizer_type:str="bert",
                 tokens_length_max: int = 100):
        
        self.caption_pids = caption_pids
        self.captions = captions
        self.tokenizer_type = tokenizer_type
        self.tokens_length_max = tokens_length_max

        # create the appropriate tokenizer type
        if tokenizer_type == "bert":
            self.tokenizer = BERTTokenizer()

        elif tokenizer_type == "simple_tokenizer":
            self.tokenizer = SimpleTokenizer()
        
        elif tokenizer_type == "tiktoken_cl100k":
            self.tokenizer = TikTokenizer(encoding_base='cl100k_base')

        elif tokenizer_type == "tiktoken_p50k":
            self.tokenizer = TikTokenizer(encoding_base='p50k_base')
        
        elif tokenizer_type == "tiktoken_r50k":
            self.tokenizer = TikTokenizer(encoding_base='r50k_base')
        
        else:
            raise NotImplemented("No implemetation for `{}` tokenization type")


    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        label, caption = self.caption_pids[index], self.captions[index]
        tokens = self.tokenizer(caption)
        token_ids, orig_token_length = pad_tokens(tokens, self.tokens_length_max)

        return label, token_ids, orig_token_length