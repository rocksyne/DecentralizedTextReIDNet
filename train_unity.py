"""
C. January 2024
License: Please see LICENSE file
Doc.: Train TextReIDNet
"""

# System modules
import logging
import datetime

# 3rd party modules
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.cuda.amp import GradScaler
from torchvision import transforms

# Application modules
from config import sys_configuration
from utils.miscellaneous_utils import set_seed
from utils.miscellaneous_utils import  setup_logger
from datasets.cuhkpedes_dataloader import build_cuhkpedes_dataloader
from model.unity import Unity
from evaluation.ranking_loss import RankingLoss
from evaluation.identity_loss import IdentityLoss
from evaluation.evaluations import calculate_similarity
from evaluation.evaluations import evaluate
from utils.miscellaneous_utils import save_model_checkpoint
from utils.miscellaneous_utils import SavePlots



# +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++[Global Configurations]++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# See https://pytorch.org/docs/stable/multiprocessing.html
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.autograd.set_detect_anomaly(True)
#warnings.filterwarnings("ignore") 
config:dict = sys_configuration(dataset_name="CUHK-PEDES")
set_seed(config.seed) # using same as https://github.com/xx-adeline/MFPE/blob/main/src/train.py 
scaler = GradScaler() 

train_data_loader, inference_img_loader, inference_txt_loader, train_num_classes = build_cuhkpedes_dataloader(config)

# Model and loss stuff
model = Unity(config).to(config.device)
identity_loss_fnx = IdentityLoss(config=config, class_num=train_num_classes).to(config.device)
ranking_loss_fnx = RankingLoss(config)


# Trainig configuration stuff
# Adapting https://github.com/xx-adeline/MFPE/blob/main/src/train.py#L36
# The human detection model will not train its paramters, so lets ignore the param settings
efficientnet_params = list(map(id, model.efficientnet_backbone.parameters()))
other_params = filter(lambda p: id(p) not in efficientnet_params, model.parameters())
other_params = list(other_params)
other_params.extend(list(identity_loss_fnx.parameters()))
param_groups = [{'params': other_params, 'lr': config.lr},
                {'params': model.efficientnet_backbone.parameters(), 'lr': config.lr*0.1}]

optimizer = optim.AdamW(param_groups, betas=(config.adam_alpha, config.adam_beta))
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.epoch_decay)
current_best_top1_accuracy:float = 0.

# Logging info
loss_plots = SavePlots(name='hp_based_loss_plot.png', 
                       save_path=config.plot_save_path,  
                       legends=["Ranking Loss", "Identity Loss", "Total Loss"],
                       horizontal_label="Epochs",
                       vertical_label="Losses",
                       title="Training Losses Over Epochs")

accuracy_plots = SavePlots(name='hp_based_accurracy_plot.png', 
                           save_path=config.plot_save_path,  
                           legends=["Top-1", "Top-5", "Top-10"],
                           horizontal_label="Epochs",
                           vertical_label="Accuracies",
                           title="Accuracies Over Epochs")

time_stamp = str(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
train_logger:logging = setup_logger(name='train_logger',log_file_path=config.train_log_path, write_mode=config.write_mode)
test_logger:logging  = setup_logger(name='test_logger',log_file_path=config.test_log_path, write_mode=config.write_mode)
train_logger.info("\n Started on {} \n {} \n".format(time_stamp,"="*35))
test_logger.info("\n Started on {} \n {}".format(time_stamp,"="*35))

if config.save_best_test_results_only:
    test_logger.info("\n Note: saving best test (inference) results only: \n")

if config.log_config_paprameters:
    for key in config.keys():
        train_logger.info("{}: {}".format(key, config[key]))
        test_logger.info("{}: {}".format(key, config[key]))
    train_logger.info("\n")
    test_logger.info("\n")


if __name__ == '__main__':

    print("")
    print('[INFO] Total parameters: {} million'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("")

    for current_epoch in range(1,config.epoch+1):

        # Conatainers for saving losses
        train_ranking_loss_list:list[float] = []
        train_identity_loss_list:list[float] = []
        train_total_loss_list:list[float] = []
        val_ranking_loss_list:list[float] = []
        val_identity_loss_list:list[float] = []
        val_total_loss_list:list[float] = []
 
        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++[Train Model]+++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        # Use tqdm to see train progress
        # https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e
        model.train() # ut in train mode
        with tqdm(train_data_loader, unit='batch') as train_data_loader_progress:
            train_data_loader_progress.set_description(f"Train - Epoch {current_epoch} of {config.epoch}")

            for train_data_batch in train_data_loader_progress:
                original_images = train_data_batch['original_images'].to(config.device)
                preprocessed_images = train_data_batch['preprocessed_images'].to(config.device)
                labels = train_data_batch['pids'].to(config.device)
                token_ids = train_data_batch['token_ids'].to(config.device)
                orig_token_length = train_data_batch['orig_token_lengths'].to(config.device)

                # Zero-grad before making prediction with model
                # https://pytorch.org/docs/stable/optim.html
                optimizer.zero_grad()

                # Use mixed precision training
                precision_dtype = torch.bfloat16 if config.device== 'cpu' else torch.float16
                with torch.autocast(device_type=config.device,  dtype=precision_dtype):
                    visaual_embeddings, textual_embeddings, filtered_labels = model(original_images=original_images,
                                           preprocessed_images=preprocessed_images,
                                           text_ids = token_ids,
                                           text_length = orig_token_length,
                                           labels = labels)


                    # Calculate losses
                    train_ranking_loss = ranking_loss_fnx(visaual_embeddings, textual_embeddings, filtered_labels)
                    train_identity_loss = identity_loss_fnx(visaual_embeddings, textual_embeddings, filtered_labels)
                    train_total_loss = (config.ranking_loss_alpha*train_ranking_loss + config.identity_loss_beta*train_identity_loss)
                
                scaler.scale(train_total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # save losses
                train_ranking_loss_list.append(train_ranking_loss.item())
                train_identity_loss_list.append(train_identity_loss.item())
                train_total_loss_list.append(train_total_loss.item())

                # Prepare the progress bar display values
                values = {"Ranking Loss":np.mean(train_ranking_loss_list),
                          "Identity Loss":np.mean(train_identity_loss_list),
                          "Total Loss":np.mean(train_total_loss_list)}
                
                train_data_loader_progress.set_postfix(values) # update progress bar
        
        # write results for this ecpoch into log file
        txt_2_write = "Epoch: {} Ranking Loss: {:.4} Identity Loss: {:.4} Total Loss: {:.4}".format(current_epoch,
                                                                                                    np.mean(train_ranking_loss_list),
                                                                                                    np.mean(train_identity_loss_list),
                                                                                                    np.mean(train_total_loss_list))
        train_logger.info(txt_2_write)
        loss_plots(current_epoch,[np.mean(train_ranking_loss_list),
                                  np.mean(train_identity_loss_list),
                                  np.mean(train_total_loss_list)])
        
        save_model_checkpoint(model,config.model_save_path,"Unity_Model.pth.tar")