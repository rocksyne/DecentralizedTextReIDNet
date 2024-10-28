from typing import Tuple
from typing import List


# 3rd party modules
import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class EfficientNetBackbone(nn.Module):
    def __init__(self, stages_to_extract_from:list=[3,5,7,9])->list:
        """
        Doc.:   The visiual backbone. https://arxiv.org/pdf/1905.11946.pdf
                We use pre-trained EfficientNet-B0 as the backbone and we extract from 
                stages 3, 5, 7, and 9. We ignore 9.  Please see the documentation at
                ../docs/efficientnet_B0_architecture.txt for the complete architecture

                Architecture (summary) of EfficientNet-B0 baseline network.
                ======================================================================================
                | Stage |  Operator          |  Resolution  | #Channels  | #Layers  | Output shape   |
                ======================================================================================
                | 1     |  Conv3x3           |  224 x 224    | 32         |  1      | 32 x 112 x 112 |
                | 2     |  MBConv1, k3x3     |  112 x 112    | 16         |  1      | 16 x 112 x 112 | 
                | 3     |  MBConv6, k3x3     |  112 x 112    | 24         |  2      | 24 x 56 x 56   |
                | 4     |  MBConv6, k5x5     |  56 x 56      | 40         |  2      | 40 x 28 x 28   |
                | 5     |  MBConv6, k3x3     |  28 x 28      | 80         |  3      | 80 x 14 x 14   |
                | 6     |  MBConv6, k5x5     |  14 x 14      | 112        |  3      | 112 x 14 x 14  |
                | 7     |  MBConv6, k5x5     |  14 x 14      | 192        |  4      | 192 x 7 x 7    |
                | 8     |  MBConv6, k3x3     |  7 x 7        | 320        |  1      | 320 x 7 x 7    |
                | 9     |  Conv1x1, Pool, FC |  7 x 7        | 1280       |  1      | 1280 x 7 x 7   |
                ======================================================================================

                For (array) indexing purposes, we shall count the stages from 0~8 as
                shown in ../docs/efficientnet_B0_architecture.txt

        Args.:  stages_to_extract_from: The stages of EfficientNet-B0 we want to extract from.
                                        Because of indexing, we do -1 on each stage from the table.
                                        The referencing the table, the stages we are interested in are [1,3,5,7,9],
                                        but for indexing purposes, we subtract 1 to get [0,2,4,6,9]

        Return: List of stages of EfficientNet-B0 and of a tensor type
        """
        super(EfficientNetBackbone, self).__init__()
        if not isinstance(stages_to_extract_from,list):
            raise TypeError("`stages_to_extract_from` should be a list[int].")
        
        self.stages_to_extract_from = [x-1 for x in stages_to_extract_from] # take care of indexing
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.feature_blocks = self.model.features

    def forward(self, x):
        features:List[torch.Tensor] = []
        for block in self.feature_blocks:
            x = block(x)
            features.append(x)

        output:List[torch.Tensor] = [features[stage] for stage in self.stages_to_extract_from]
        return output # eg. features from stages [3,5,7,9] according to table above