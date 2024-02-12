"""
Doc.:   Codebase adapted from https://github.com/anosorae/IRRA/tree/main
"""
import sys, os

# 3rd party modules
from torch.utils.data import DataLoader

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from utils.miscellaneous_utils import mhpv2_collate
from datasets.mhpv2 import MHPv2



# build the dataloader
def build_mhpv2_dataloader(config:dict=None):
    """Build the dataloader"""

    # Training set / split
    train_set = MHPv2(config=config, dataset_split='train')
    train_set_loader = DataLoader(train_set,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   collate_fn=mhpv2_collate)
    
    # validation set / split
    val_set = MHPv2(config=config, dataset_split='val')
    val_set_loader = DataLoader(val_set,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   collate_fn=mhpv2_collate)
    
    
    return train_set_loader, val_set_loader

