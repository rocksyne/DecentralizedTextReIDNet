"""
Doc.:   Codebase adapted from https://github.com/anosorae/IRRA/tree/main
"""

# System modules
import os

# 3rd party modules
from torch.utils.data import Dataset, DataLoader

# Application modules
from PIL import Image
from utils.miscellaneous_utils import get_transform
from utils.miscellaneous_utils import pad_tokens
from datasets.bert_tokenizer import BERTTokenizer


class ImageFolderDataset(Dataset):
    def __init__(self, directory, transform=None):
        super(ImageFolderDataset, self).__init__()

        if not os.path.exists(directory) or not os.path.isdir(directory):
            raise ValueError(f"`{directory}` is not a valid directory.")

        self.directory = directory
        self.transform = transform
        # Exclude .txt files
        self.images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and not f.endswith('.txt')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name).convert('RGB')  # Convert image to RGB

        if self.transform:
            image = self.transform(image)

        return image, img_name



class WebUploadedImageData(Dataset):
    def __init__(self, image_list, transform=None):
        super(WebUploadedImageData, self).__init__()

        # TODO: Check that all list elements are images
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image = Image.open(self.image_list[idx]).convert('RGB')  # Convert image to RGB
        if self.transform:
            image = self.transform(image)

        return image, idx



class TextQuery(object):
    def __init__(self,tokens_length_max: int = 100):
        self.tokens_length_max = tokens_length_max
        self.tokenizer = BERTTokenizer()

    def __call__(self, text_query):
        tokens = self.tokenizer(text_query)
        token_ids, orig_token_length = pad_tokens(tokens, self.tokens_length_max)
        return token_ids, orig_token_length


# build the dataloader
def build_img_dataloader_from_dir(config:dict=None, images_directory_path:str=None):
    """Build the dataloader"""
    
    transform = get_transform('inference',config)
    dataset = ImageFolderDataset(directory=images_directory_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return data_loader


# build the dataloader
def build_img_dataloader_from_uploads(config, image_list):
    """Build the dataloader"""
    transform = get_transform('inference',config)
    dataset = WebUploadedImageData(image_list, transform=transform)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return data_loader


def process_text_into_tokens(text:str=None):
    text_processor = TextQuery()
    token_ids, orig_token_length = text_processor(text)
    return token_ids, orig_token_length




    

