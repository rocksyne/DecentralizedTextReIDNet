"""
Doc.:   TODO: I don't know what to name it yet so lets see
"""

# 3rd party modules
import torch
from torch import nn

# Apppication modules
from model.language_network import GRULanguageNetwork
from model.efficientnet_backbone import EfficientNetBackbone
from model.feature_pyramid_network import FeaturePyramidNetwork


class DecentralizedTextReIDNet(nn.Module):
    def __init__(self,configs=None)->list[torch.tensor]:
        """
        Doc.:   Unified model for performning person detection and segmentation, 
                as well as person re-id using text as query.
        
        TODO: do propoer documentation
        """
        super(DecentralizedTextReIDNet, self).__init__()
        self.configs:dict = configs
        self.efficientnet_backbone = EfficientNetBackbone()
        self.FPN = FeaturePyramidNetwork()
        # self.language_network = GRULanguageNetwork(configs)

    def forward(self,image):
        ...

    
    def visual_model(self,image:torch.Tensor=None)->torch.Tensor:
        """Extract visual features using efficientnet"""
        S3_output, S5_output, S7_output, S9_output = self.efficientnet_backbone(image)
        return S3_output, S5_output, S7_output, S9_output

    
    def person_detection_model(self,image:torch.tensor)->torch.tensor:
        """
        Doc.:   This sub-model is responsible for person detection and segmentation.
                It performs multi-human parsing. TODO: show the paper
        """
        S3_output, S5_output, S7_output, S9_output = self.visual_model(image)
        P_3, P_5, P_7, P_9 = self.FPN([S3_output, S5_output, S7_output, S9_output]) 



