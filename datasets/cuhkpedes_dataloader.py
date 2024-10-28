"""
Doc.:   Codebase adapted from https://github.com/anosorae/IRRA/tree/main
"""
import os, sys

# 3rd party modules
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from utils.miscellaneous_utils import collate, get_transform
from datasets.cuhkpedes import CUHKPEDES
from datasets.bases import ImageTextDataset, ImageDataset, TextDataset


# build the dataloader
def build_cuhkpedes_dataloader(config:dict=None):
    """Build the dataloader"""
    
    dataset_object = CUHKPEDES(config)

    # Lets get the transforms
    train_transform = T.Compose([T.ToTensor(),T.Normalize(mean=config.MHPV2_means,std=config.MHPV2_stds)])
    inference_transform = T.Compose([T.ToTensor(),T.Normalize(mean=config.MHPV2_means,std=config.MHPV2_stds)])
    train_num_classes = len(dataset_object.train_id_container)

    train_set = ImageTextDataset(dataset_object.train,
                                 train_transform,
                                 tokenizer_type=config.tokenizer_type,
                                 tokens_length_max=config.tokens_length_max)
    
    train_data_loader = DataLoader(train_set,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   collate_fn=collate)

    ds = dataset_object.val if config.model_testing_data_split == 'val' else dataset_object.test
    inference_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],inference_transform)
    inference_txt_set = TextDataset(ds['caption_pids'], ds['captions'], tokenizer_type=config.tokenizer_type, tokens_length_max=config.tokens_length_max)

    inference_img_loader = DataLoader(inference_img_set,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers)
    inference_txt_loader = DataLoader(inference_txt_set,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers)
    
    return train_data_loader, inference_img_loader, inference_txt_loader, train_num_classes

