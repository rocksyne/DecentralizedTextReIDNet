# 3rd party modules
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class VisualNetwork(nn.Module):
    def __init__(self):
        """
        Doc.:   The visiual backbone. https://arxiv.org/pdf/1905.11946.pdf
                We use pre-trained EfficientNet-B0 as the backbone and we extract from stage 9

                Architecture of EfficientNet-B0 baseline network.
                =====================================================================
                | Stage |  Operator          |  Resolution  | #Channels  | #Layers |
                =====================================================================
                | 1     |  Conv3x3           |  224 x 224    | 32         |  1      |
                | 2     |  MBConv1, k3x3     |  112 x 112    | 16         |  1      |
                | 3     |  MBConv6, k3x3     |  112 x 112    | 24         |  2      |
                | 4     |  MBConv6, k5x5     |  56 x 56      | 40         |  2      |
                | 5     |  MBConv6, k3x3     |  28 x 28      | 80         |  3      |
                | 6     |  MBConv6, k5x5     |  14 x 14      | 112        |  3      |
                | 7     |  MBConv6, k5x5     |  14 x 14      | 192        |  4      |
                | 8     |  MBConv6, k3x3     |  7 x 7        | 320        |  1      |
                | 9     |  Conv1x1, Pool, FC |  7 x 7        | 1280       |  1      |
                =====================================================================
        """
        super(VisualNetwork, self).__init__()
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.feature_blocks = self.model.features

    def forward(self, x):
        features = []
        for block in self.feature_blocks:
            x = block(x)
            features.append(x)
            
        return features[-1] # final layer output
    