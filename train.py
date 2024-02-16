"""
C. Feb. 2024
License: Please see LICENSE file
Doc.: Train combined person detection and re-identification model
"""

# 3rd party modules 
# 3rd party modules
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.cuda.amp import GradScaler

# Application modules
from config import sys_configuration
from utils.mhpv2_dataset_utils import clean_data_recieved_from_collator
from datasets.mhpv2_dataloader import build_mhpv2_dataloader
from model.decentralized_TextReIDNet import DecentralizedTextReIDNet

# Some system configurations
# See https://pytorch.org/docs/stable/multiprocessing.html
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.autograd.set_detect_anomaly(True)
# warnings.filterwarnings("ignore") 
scaler = GradScaler() 

# Global configurations
config = sys_configuration()
model = DecentralizedTextReIDNet(config).to(config.device)
train_set_loader, val_set_loader = build_mhpv2_dataloader(config)



print('[INFO] Total params: {} M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))



# Trainig configuration stuff
# Adapting https://github.com/xx-adeline/MFPE/blob/main/src/train.py#L36
cnn_params = list(map(id, model.efficientnet_backbone.parameters()))
other_params = filter(lambda p: id(p) not in cnn_params, model.parameters())
other_params = list(other_params)
#other_params.extend(list(identity_loss_fnx.parameters()))
param_groups = [{'params': other_params, 'lr': config.lr},
                {'params': model.efficientnet_backbone.parameters(), 'lr': config.lr*0.1}]
optimizer = optim.AdamW(param_groups, betas=(config.adam_alpha, config.adam_beta))
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.epoch_decay)
current_best_top1_accuracy:float = 0.

if __name__ == '__main__':

    for current_epoch in range(1, config.epoch+1):

        # Training
        with tqdm(train_set_loader, unit='batch') as train_set_loader_progress:

            model.train()

            for train_data_batch in train_set_loader_progress:

                images = train_data_batch['images'].to(config.device)
                # Prepare / clean the human instance data recieved from collator
                human_bbs_n_labels = train_data_batch['human_instance_bbs_n_classes']
                human_instance_masks = train_data_batch['human_instance_masks']
                human_gt_bboxes, human_gt_masks, juman_gt_labels = clean_data_recieved_from_collator(human_bbs_n_labels,human_instance_masks)

                # Zero-grad before making prediction with model
                # https://pytorch.org/docs/stable/optim.html
                optimizer.zero_grad()

                # Use mixed precision training
                precision_dtype = torch.bfloat16 if config.device== 'cpu' else torch.float16
                with torch.autocast(device_type=config.device,  dtype=precision_dtype):
                    human_outs, human_mask_feat_pred = model(images)

                    # calculate loss
                
                # scaler.scale(train_total_loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

