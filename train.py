"""
C. Feb. 2024
License: Please see LICENSE file
Doc.: Train combined person detection and re-identification model
"""

# 3rd party modules 
import torch
from tqdm import tqdm as tqdm

# Application modules
from config import sys_configuration
from datasets.mhpv2_dataloader import build_mhpv2_dataloader
from model.decentralized_TextReIDNet import DecentralizedTextReIDNet

# Global configurations
config = sys_configuration()
unified_model = DecentralizedTextReIDNet(config)
train_set_loader, val_set_loader = build_mhpv2_dataloader(config)
print('[INFO] Total params: {} M'.format(sum(p.numel() for p in unified_model.parameters()) / 1000000.0))



if __name__ == '__main__':

    for current_epoch in range(1, config.epoch+1):

        # Training
        with tqdm(train_set_loader, unit='batch') as train_set_loader_progress:

            for train_data_batch in train_set_loader_progress:
                ...
