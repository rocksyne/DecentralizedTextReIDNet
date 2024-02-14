
"""
Doc.:   Calculate the normalization prameters, that is the means and the
        standard deviation of each color channel
"""
# System modules
import os, sys

# 3rd party modules
import torch
from tqdm import tqdm as tqdm

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from config import sys_configuration
from datasets.mhpv2_dataloader import build_mhpv2_dataloader

# Global params
config = sys_configuration()
train_set_loader, val_set_loader = build_mhpv2_dataloader(config)


def calculate_mean_std(dataloader):
    # Placeholder tensors for the sum and squared sum of all pixels
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for _1_batch in tqdm(dataloader):
        data = _1_batch['images']
        # Sum up the values across all channels [Batch,H,W]
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        # Sum up the squared values across all channels
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    # Calculate mean and std
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean.tolist(), std.tolist()



if __name__ == '__main__':

    mean, std = calculate_mean_std(train_set_loader)
    print(f"Mean: {mean}")
    print(f"Std: {std}")


