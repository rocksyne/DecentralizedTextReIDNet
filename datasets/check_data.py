import os, sys
from tqdm import tqdm as tqdm
sys.path.insert(0, os.path.abspath('../'))
from config import sys_configuration
from mhpv2_dataloader import build_mhpv2_dataloader
config = sys_configuration()
train_set_loader, val_set_loader = build_mhpv2_dataloader(config)
for train_data in tqdm(train_set_loader):
    #print("{} {} {}".format(train_data['images'].shape, train_data['human_instance_masks'].shape, train_data['human_instance_bbs_n_classes'].shape))
    ...