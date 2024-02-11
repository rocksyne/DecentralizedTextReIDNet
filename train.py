"""
C. Feb. 2024
License: Please see LICENSE file
Doc.: Train combined person detection and re-identification model
"""

# 3rd party modules 
import torch

# Application modules
from config import sys_configuration
from model.decentralized_TextReIDNet import DecentralizedTextReIDNet

config = sys_configuration()

unified_model = DecentralizedTextReIDNet(config)
input_image = torch.randn(1, 3, 384, 128)
print('[INFO] Total params: {} M'.format(sum(p.numel() for p in unified_model.parameters()) / 1000000.0))

unified_model(input_image)