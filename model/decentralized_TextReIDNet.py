"""
Doc.:   TODO: I don't know what to name it yet so lets see
"""

# 3rd party modules
import torch
from torch import nn

# Apppication modules
from model.efficientnet_backbone import EfficientNetBackbone
from model.feature_pyramid_network import FeaturePyramidNetwork


class DecentralizedTextReIDNet(nn.Module):
    def __init__(self,)->list[torch.tensor]:
        """
        Doc.:   Unified model for performning person detection and segmentation, 
                as well as person re-id using text as query.
        
        TODO: do propoer documentation
        """
        super(DecentralizedTextReIDNet, self).__init__()
        self.efficientnet_backbone = EfficientNetBackbone()
        self.FPN = FeaturePyramidNetwork()

    def forward(self,image):
        S3_output, S5_output, S7_output, S9_output = self.efficientnet_backbone(image)
        P_3, P_5, P_7, P_9 = self.FPN([S3_output, S5_output, S7_output, S9_output])



