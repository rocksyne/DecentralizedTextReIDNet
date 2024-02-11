import torch
from model.decentralized_TextReIDNet import DecentralizedTextReIDNet

unified_model = DecentralizedTextReIDNet()
input_image = torch.randn(1, 3, 512, 512)
print('[INFO] Total params: {} M'.format(sum(p.numel() for p in unified_model.parameters()) / 1000000.0))

unified_model(input_image)