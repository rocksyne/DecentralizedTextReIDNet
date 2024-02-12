"""
Doc.: Miscellaneous utilities for codebase
"""

# System modules
import os
import random
import logging
import datetime
from typing import Any

# 3rd party modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T


def set_seed(seed_value:int=3407):
    """
    Doc.: Set seed for for reproducibility
    Args.: seed_value: seed value 
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# +++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++[Utility Functions]+++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++
def setup_logger(name:str=None, 
                 log_file_path:str=None, 
                 write_mode:str='overwrite', 
                 level:logging=logging.INFO,
                 timestamp:str=None):
    """
    Doc.:   Function to setup as many loggers as desired.
            The log files are cleared every time the 

    Args.:  • config: configuration parameters
            • name: name of the logger
            • log_file_path: file path to the log file
            • write_mode: eaither append to existing file or overwrite. Values: `overwrite` or `append`
            • level: logging level
            • timestamp: log the timestamp also

    Returns: logger object
    """
    if name is None:
        raise ValueError("`name` can not be None")
    
    if log_file_path is None:
        raise ValueError("`log_file_path` can not be None")
    
    if level is None:
        raise ValueError("`level` can not be None")

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers according to the write mode
    if write_mode == 'overwrite':
        file_handler = logging.FileHandler(log_file_path, mode='w')
    elif write_mode == 'append':
        file_handler = logging.FileHandler(log_file_path, mode='a')
    else:
        raise NotImplementedError("No implementaion for `write_mode`={}".format(write_mode))
    
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    if timestamp:
        logger.info("")
        logger.info("Logging started at ",timestamp)
        logger.info("")

    return logger


def pad_tokens(tokens, tokens_length_max):
    """
    This entire code snippet is adapted from the URL below.
    https://github.com/xx-adeline/MFPE/blob/main/src/data/dataset.py#L65
    Enhancement: maske tensors contigeous for memory efficiency

    TODO: [x]Change variable names to make more applicable

    Args.:  tokens: Textual descriptions converted into tokens. Eg. [1165, 13, 564, 74, ..., 1167]
            tokens_length_max: The maximum number of tokens needed, eg, 100

    Return: list: padded tokens and the original length of the tokens before padding to tokens_length_max
    """
    tokens_length = len(tokens)
    tokens = torch.from_numpy(np.array(tokens)).view(-1).contiguous().float() # make contiguous
    if tokens_length < tokens_length_max:
        zero_padding = torch.zeros(tokens_length_max - tokens_length)
        tokens = torch.cat([tokens, zero_padding], 0)
    else:
        tokens = tokens[:tokens_length_max]
        tokens_length = tokens_length_max
    return tokens, tokens_length


def mhpv2_collate(bacth_data):
    """
    Doc.:   Data collator for the dataloader
            Read article at the link below to understand the need for this DataCollator class
            https://plainenglish.io/blog/understanding-collate-fn-in-pytorch-f9d1742647d3
    """
    # keys = set([key for b in batch for key in b.keys()])
    # # turn list of dicts data structure to dict of lists data structure
    # dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    # # do propper casting of the data type
    # batch_tensor_dict = {}
    # for k, v in dict_batch.items():
    #     if isinstance(v[0], int):
    #         batch_tensor_dict.update({k: torch.tensor(v)})
    #     elif torch.is_tensor(v[0]):
    #         batch_tensor_dict.update({k: torch.stack(v)})
    #     elif isinstance(v[0], str):
    #          batch_tensor_dict.update({k: v})
    #     else:
    #         raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")
    images = [data['images'] for data in bacth_data]
    images = torch.stack(images)

    image_paths = [data['image_paths'] for data in bacth_data]

    """
    Then number of human masks and the corresponding bounding boxes differ.
    However, pytorch's data loader reqire that for a batch of data, all the data
    should have equal dimensions. The only thing I can think of now is to pad defficit
    dimensions with 'junk' data and clean it up later when I have retrieved it.
    """
    human_instance_masks = [data['human_instance_masks'] for data in bacth_data]
    if len(human_instance_masks)<=0:
        raise ValueError(f"There are no human instance masks.")
    if (len(human_instance_masks[0].shape)!=3): 
        raise ValueError("Expected shape of each mask is (1,H,W) but got {}".format(human_instance_masks[0].shape))

    # get the maximum number of masks for an image in this batch
    batch_size = len(human_instance_masks) 
    _, height, width = human_instance_masks[0].shape
    max_masks_in_current_batch = max(len(d) for d in human_instance_masks)
    human_instance_masks_canvas = torch.ones((batch_size,max_masks_in_current_batch,height,width))*-1 # canvas of -1s

    for idx, im in enumerate(human_instance_masks):
        if im.shape[0]>0:
            human_instance_masks_canvas[idx,:im.shape[0],:,:] = im
        else:
            raise Exception("Invalid mask. Please do debug.")
        
    
    """
    Now, we need to do the same for the bounding boxes and the classes
    """
    human_instance_bbs_n_classes = [data['human_instance_bbs_n_classes'] for data in bacth_data]
    if len(human_instance_bbs_n_classes) <= 0:
        raise ValueError("No bounding boxes nor class labels found.")
    if len(human_instance_bbs_n_classes) != len(human_instance_masks):
        raise ValueError("Dimension incompactibility")
    
    max_num_human_instance_bbs_n_classes= max(len(d) for d in human_instance_bbs_n_classes)
    if max_num_human_instance_bbs_n_classes > 0:
        human_instance_bbs_n_classes_padded = torch.ones((batch_size,max_num_human_instance_bbs_n_classes,6))*-1 # canvas of -1s

        for idx, bb in enumerate(human_instance_bbs_n_classes):
            if bb.shape[0]>0:
                human_instance_bbs_n_classes_padded[idx,:bb.shape[0],:] = bb
            else:
                raise ValueError("Invalid bbs and masks")


    collated_data = dict(images=images,
                         image_paths=image_paths,
                         human_instance_masks=human_instance_masks_canvas,
                         human_instance_bbs_n_classes=human_instance_bbs_n_classes_padded)

    return collated_data



def get_transform(dataset_split:str=None, config:dict=None):
    """
    Doc.:   Get the appropriate transform
    Args.:  • dataset_split: The dataset split. `train`, `val`, `test`
            • config: The configuration (dict) object for system configuration
    Return: torchvision.transforms
    """
    if dataset_split not in ['train','inference']:
        raise ValueError("Invalid dataset_split. Expected value to be `train`, `inference` but got `{}`".format(dataset_split))
    
    if config is None:
        raise ValueError("`config` can not be None.")
    
    if dataset_split == 'train':
        transform = T.Compose([T.Resize(config.image_size, T.InterpolationMode.BICUBIC),
                               T.Pad(10),
                               T.RandomCrop(config.image_size),
                               T.RandomHorizontalFlip(),
                               T.ToTensor(),
                               T.Normalize(config.mean,config.std)])
        
    else: # this is for val and test
        transform = T.Compose([T.Resize(config.image_size, T.InterpolationMode.BICUBIC),
                               T.ToTensor(),
                               T.Normalize(config.mean,config.std)])
        
    return transform


def save_model_checkpoint(model=None, save_dir:str=None)->None:
    """
    Doc.:   Save the trained model for inference
    Args.:  • model: instance of the model to save
            • save_path: path to where the model should be saved
    """
    if os.path.isdir(save_dir) is False:
        raise FileNotFoundError("`{}` does not exist.".format(save_dir))
    torch.save(model.state_dict(),os.path.join(save_dir,"TextReIDNet_State_Dicts.pth.tar")) # Save state dicts
    #torch.save(model,os.path.join(save_dir,"TextReIDNet_Full_Model.pth")) # Save the full model



# # +++++++++++++++++++++++++++++++++++++++++++++++++
# # ++++++++++++++++[Utility Classes]++++++++++++++++
# # +++++++++++++++++++++++++++++++++++++++++++++++++
# class SavePlots(object):
#     def __init__(self, 
#                  name: str = "save_plot.png", 
#                  save_path: str = None,  
#                  legends: list = None,
#                  horizontal_label: str = "Epochs",
#                  vertical_label: str = "Losses",
#                  title: str = "Training Losses Over Epochs"):
        
#         if save_path is None:
#             save_path = "."  # default to current directory if none provided
#         elif not os.path.isdir(save_path):
#             raise FileNotFoundError(f"`{save_path}` does not exist.")
        
#         self.name: str = os.path.join(save_path, name)  # absolute path name
        
#         if legends is None or not legends:
#             raise ValueError("`legends` cannot be `None` or empty.")
#         self.legends: list = legends
        
#         self.epochs: list = []
#         self.horizontal_label: str = horizontal_label
#         self.vertical_label: str = vertical_label
#         self.title: str = title

#         # create a dynamic holder for values to be plotted
#         self.dynamic_variable: dict[int, list[float]] = {indx: [] for indx in range(len(self.legends))}
    
#     def update(self, epoch: int, values: list[float]):
#         """
#         Doc.: Populate the variables to hold numbers
#         Args.:  
#         • epoch: current epoch
#         • values: current list of values
#         """
#         if values is None:
#             raise ValueError("`values` cannot be `None`.")
        
#         self.epochs.append(epoch)
#         zipped_data = zip(range(len(self.legends)), values)

#         for indx, value in zipped_data:
#             self.dynamic_variable[indx].append(value)

#     def __call__(self, epoch: int = 0, values: list[float] = None):
#         """
#         Doc.: Function call to save the plots
#         Args.:  
#         • epoch: The current epoch
#         • values: The list of values to be plotted. Each value corresponds to a list for epochs
#         """
#         if values is None:
#             raise ValueError("`values` cannot be `None`.")
#         if len(values) != len(self.legends):
#             raise ValueError(f"The list of values does not match the number of intended plots. No. of values is {len(values)} and no. of legends is {len(self.legends)}.")
        
#         self.update(epoch=epoch, values=values)

#         plt.figure(figsize=(10, 5))
#         # plot values to graph
#         for indx in self.dynamic_variable:
#             plt.plot(self.epochs, self.dynamic_variable[indx], label=self.legends[indx])
        
#         # plot stuff
#         plt.xlabel(self.horizontal_label)
#         plt.ylabel(self.vertical_label)
#         plt.title(self.title)
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(self.name)
#         plt.close() # https://heitorpb.github.io/bla/2020/03/18/close-matplotlib-figures/

