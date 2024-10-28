"""
C. Feb. 2024
License: Please see LICENSE file
Doc.: Train combined person detection and re-identification model
"""
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
from utils.miscellaneous_utils import save_model_checkpoint
from model.human_detection_network import HumanDetectionNetwork
from evaluation.focal_and_dice_loss import  FocalAndDiceLoss
from utils.miscellaneous_utils import SavePlots

# Some system configurations
# See https://pytorch.org/docs/stable/multiprocessing.html
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.autograd.set_detect_anomaly(True)
# warnings.filterwarnings("ignore") 
scaler = GradScaler() 

# Global configurations
config = sys_configuration()
model = HumanDetectionNetwork(config).to(config.device)
human_parsing_loss =  FocalAndDiceLoss()
train_set_loader, val_set_loader = build_mhpv2_dataloader(config)

# Logging info
loss_plots = SavePlots(name='loss_plot.png', 
                       save_path=config.plot_save_path,  
                       legends=["Train Loss", "Validation Loss"],
                       horizontal_label="Epochs",
                       vertical_label="Losses",
                       title="Training Losses Over Epochs")

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
current_best_loss:float = 100.

if __name__ == '__main__':

    for current_epoch in range(1, config.epoch+1):
        t_human_instance_losses:list[float] = []
        t_human_category_losses:list[float] = []
        t_human_total_losses:list[float] = []
        v_human_instance_losses:list[float] = []
        v_human_category_losses:list[float] = []
        v_human_total_losses:list[float] = []

        # Training
        with tqdm(train_set_loader, unit='batch') as train_set_loader_progress:

            model.train()

            for train_data_batch in train_set_loader_progress:

                train_set_loader_progress.set_description(f"Train - Epoch {current_epoch} of {config.epoch}")

                images = train_data_batch['images'].to(config.device)
                # Prepare / clean the human instance data recieved from collator
                human_bbs_n_labels = train_data_batch['human_instance_bbs_n_classes'].to(config.device)
                human_instance_masks = train_data_batch['human_instance_masks']
                human_gt_bboxes, human_gt_masks, human_gt_labels = clean_data_recieved_from_collator(human_bbs_n_labels,human_instance_masks,config)

                # Zero-grad before making prediction with model
                # https://pytorch.org/docs/stable/optim.html
                optimizer.zero_grad()

                # Use mixed precision training
                precision_dtype = torch.bfloat16 if config.device== 'cpu' else torch.float16
                with torch.autocast(device_type=config.device,  dtype=precision_dtype):
                    cate_pred, kernel_pred, human_mask_feat_pred = model(images)
                    human_loss_inputs = (cate_pred, kernel_pred, human_mask_feat_pred, human_gt_bboxes, human_gt_labels, human_gt_masks)
                    loss_ins, loss_cate = human_parsing_loss(*human_loss_inputs)
                    total_loss = loss_ins + loss_cate
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                t_human_instance_losses.append(loss_ins.item())
                t_human_category_losses.append(loss_cate.item())
                t_human_total_losses.append(total_loss.item())

                # Prepare the progress bar display values
                values = {"T. Instance Loss":np.mean(t_human_instance_losses),
                          "T. Category Loss":np.mean(t_human_category_losses),
                          "T. Total Loss":np.mean(t_human_total_losses)}
                
                train_set_loader_progress.set_postfix(values) # update progress bar

        # Training
        with tqdm(val_set_loader, unit='batch') as val_set_loader_progress:

            model.eval()

            for val_data_batch in val_set_loader_progress:

                val_set_loader_progress.set_description(f"Test -- Epoch {current_epoch} of {config.epoch}")

                images = val_data_batch['images'].to(config.device)
                # Prepare / clean the human instance data recieved from collator
                human_bbs_n_labels = val_data_batch['human_instance_bbs_n_classes'].to(config.device)
                human_instance_masks = val_data_batch['human_instance_masks']
                human_gt_bboxes, human_gt_masks, human_gt_labels = clean_data_recieved_from_collator(human_bbs_n_labels,human_instance_masks,config)

                # Use mixed precision training
                with torch.no_grad():
                    cate_pred, kernel_pred, human_mask_feat_pred = model(images)
                    human_loss_inputs = (cate_pred, kernel_pred, human_mask_feat_pred, human_gt_bboxes, human_gt_labels, human_gt_masks)
                    loss_ins, loss_cate = human_parsing_loss(*human_loss_inputs)
                    total_loss = loss_ins + loss_cate
                
                v_human_instance_losses.append(loss_ins.item())
                v_human_category_losses.append(loss_cate.item())
                v_human_total_losses.append(total_loss.item())

                # Prepare the progress bar display values
                values = {"V. Instance Loss":np.mean(v_human_instance_losses),
                          "V. Category Loss":np.mean(v_human_category_losses),
                          "V. Total Loss":np.mean(v_human_total_losses)}
                
                val_set_loader_progress.set_postfix(values) # update progress bar 

        # Save the best state of the model
        if total_loss.item() < current_best_loss:
            save_model_checkpoint(model,config.model_save_path,"Decentralized_Model.pth.tar")
            current_best_loss = total_loss.item()
        
        # Save the plot
        loss_plots(current_epoch,[np.mean(t_human_total_losses), np.mean(v_human_total_losses)])

        print("")

