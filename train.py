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
from utils.mhpv2_dataset_utils import clean_data_recieved_from_collator
from datasets.mhpv2_dataloader import build_mhpv2_dataloader
from model.decentralized_TextReIDNet import DecentralizedTextReIDNet

# Global configurations
config = sys_configuration()
unified_model = DecentralizedTextReIDNet(config).to(config.device)
train_set_loader, val_set_loader = build_mhpv2_dataloader(config)
print('[INFO] Total params: {} M'.format(sum(p.numel() for p in unified_model.parameters()) / 1000000.0))



if __name__ == '__main__':

    for current_epoch in range(1, config.epoch+1):

        # Training
        with tqdm(train_set_loader, unit='batch') as train_set_loader_progress:

            unified_model.train()
            
            for train_data_batch in train_set_loader_progress:

                images = train_data_batch['images'].to(config.device)
                # Prepare / clean the human instance data recieved from collator
                human_bbs_n_labels = train_data_batch['human_instance_bbs_n_classes']
                human_instance_masks = train_data_batch['human_instance_masks']
                human_gt_bboxes, human_gt_masks, juman_gt_labels = clean_data_recieved_from_collator(human_bbs_n_labels,human_instance_masks)

                model_output = unified_model(images)

                raise Exception("We stop here")
