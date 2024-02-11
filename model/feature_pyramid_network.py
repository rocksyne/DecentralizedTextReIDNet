# 3rd party modules
import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, out_channels_per_stage_level:list[int]=[24,80,192,1280])->list[torch.tensor]:
        """
        Doc.:   Feature Pyramid Network (FPN) for building a rich multi-scale feature hierarchy 
                for person detection and segmentation. See https://arxiv.org/pdf/1612.03144.pdf.
        
        Args.:  out_channels_per_stage_level: The number of output channels for stages 3,5,7 and 9 of the Efficient net. 
        """
        super(FeaturePyramidNetwork, self).__init__()
        S3_channels, S5_channels, S7_channels, S9_channels = out_channels_per_stage_level
        
        self.S3_lateral_1x1_conv = 1
        self.S5_lateral_1x1_conv = 1
        self.S7_lateral_1x1_conv = 1
        self.S9_lateral_1x1_conv = 1

        # 1x1 convolutions fro the lateral outputs

    def forward(self, x):
        ...