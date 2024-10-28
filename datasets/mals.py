import os
import json
import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from textblob import TextBlob
import numpy as np
from typing import List, Tuple, Dict
from utils.iotools import read_image, download_punkt_if_not_exists
from utils.miscellaneous_utils import pad_tokens
from datasets.bert_tokenizer import BERTTokenizer
from datasets.randaugment import RandomAugment
from datasets.random_erase import RandomErasing

class MALSDataset(Dataset):
    def __init__(self, 
                 dataset_parent_directory: str = "/media/rockson/Data_drive/MALS/", 
                 split_type: str = '4x',
                 tokens_length_max: int = 56,
                 transforms: transforms = None):
        """
        Dataset initialization.
        """
        download_punkt_if_not_exists() # neccessary
        self.dataset_parent_directory = dataset_parent_directory
        self.tokens_length_max = tokens_length_max
        self.annot_file_path = os.path.join(dataset_parent_directory, "gene_attrs", f"g_{split_type}_attrs.json")
        self.image_directory = os.path.join(dataset_parent_directory, "gene_crop", f"{split_type}")
        self.annotations = self._load_json(self.annot_file_path)
        self.processed_annotations = self._process_annotation()
        self.transforms = transforms
        self.tokenizer = BERTTokenizer()

    def _load_json(self, file_path: str) -> List[dict]:
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _correct_sentence(self, text: str) -> str:
        blob = TextBlob(text)
        corrected_text = ' '.join(
            sentence.string.capitalize() + ('.' if not sentence.string.endswith('.') else '')
            for sentence in blob.sentences
        )
        return corrected_text.strip()
    
    def _process_annotation(self) -> List[Tuple[int, str, str, str]]:
        """
        Convert annotations to [(person_id, image_id, image_path, caption), ...]
        """
        dataset: List[Tuple[int, str, str, str]] = []
        for person_id, data in tqdm(enumerate(self.annotations), total=len(self.annotations)):
            caption = self._correct_sentence(data['caption'])
            img_path = os.path.join(self.dataset_parent_directory, data['image'])
            image_id = data['image_id']
            dataset.append((person_id, image_id, img_path, caption))
        return dataset

    def __len__(self) -> int:
        return len(self.processed_annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        person_id, image_id, img_path, caption = self.processed_annotations[idx]
        person_id = torch.tensor([person_id], dtype=torch.long)

        img = read_image(img_path)
        if self.transforms:
            img = self.transforms(img)

        tokens = self.tokenizer(caption)
        token_ids, orig_token_length = pad_tokens(tokens, self.tokens_length_max)
        
        return {
            'person_ids': person_id,
            'image_ids': image_id,
            'img_path': img_path,
            'images': img,
            'token_ids': token_ids.to(torch.long),
            'orig_token_length': orig_token_length,
            'caption': caption
        }


def mals_data_loader(split_type: str = '4x', operation: str = 'train', config: Dict[str, str] = {}) -> Tuple[DataLoader, int]:
    if operation == 'train':
        shuffle = True
        augs = ['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']
        transformation = transforms.Compose([
            transforms.Resize((384, 128), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=augs),
            transforms.ToTensor(),
            transforms.Normalize((0.4416847, 0.41812873, 0.4237452), (0.3088255, 0.29743394, 0.301009)),
            RandomErasing(probability=0.6, mean=[0.0, 0.0, 0.0])])
    else:
        shuffle = False
        transformation = transforms.Compose([
            transforms.Resize((384, 128), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.4416847, 0.41812873, 0.4237452), (0.3088255, 0.29743394, 0.301009))])

    dataset = MALSDataset(dataset_parent_directory=config.MALS_dataset_parent_dir, split_type=split_type, transforms=transformation)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle, num_workers=config['num_workers'])
    return data_loader, len(dataset.processed_annotations)


if __name__ == "__main__":
    config = {'batch_size': 32, 'num_workers': 4}
    mals_data_loader, num_classes = mals_data_loader(config=config)
    print(next(iter(mals_data_loader)))
