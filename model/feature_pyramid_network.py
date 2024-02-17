# 3rd party modules
import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from model.model_utils import DepthwiseSeparableConv2d


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, out_channels_per_stage_level: list[int] = [24, 80, 192, 1280], feature_size: int = 256) -> list[torch.tensor]:
        """
        Doc.:   Feature Pyramid Network (FPN) for building a rich multi-scale feature hierarchy 
                for person detection and segmentation. See https://arxiv.org/pdf/1612.03144.pdf.
                See ../docs/FPN.png for the typical schematics of FPN

        Args.:  out_channels_per_stage_level: The number of output channels for stages 3,5,7 and 9 of the Efficient net. 
                feature_size: The number of channels or features as the output, AKA, out_channels

        TODO: Make code more modular and scalable to accomodate new stage values
        """
        super(FeaturePyramidNetwork, self).__init__()
        S3_channels, S5_channels, S7_channels, S9_channels = out_channels_per_stage_level

        # 1x1 convolutions fro the lateral outputs
        self.S3_lateral_1x1_conv = nn.Conv2d(in_channels=S3_channels, out_channels=feature_size, kernel_size=1)
        self.S5_lateral_1x1_conv = nn.Conv2d(in_channels=S5_channels, out_channels=feature_size, kernel_size=1)
        self.S7_lateral_1x1_conv = nn.Conv2d(in_channels=S7_channels, out_channels=feature_size, kernel_size=1)
        self.S9_lateral_1x1_conv = nn.Conv2d(in_channels=S9_channels, out_channels=feature_size, kernel_size=1)

        # upsampling method for top-down layers
        self._2x_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self._4x_upsample = nn.Upsample(scale_factor=4, mode='bilinear')

        self.conv3x3 = DepthwiseSeparableConv2d(in_channels=feature_size,
                                                out_channels=feature_size,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                bias=True)

        self.P_10_conv = DepthwiseSeparableConv2d(in_channels=S9_channels,
                                                  out_channels=feature_size,
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1,
                                                  bias=True)

    def forward(self, stage_outputs_as_input: list[torch.Tensor]):

        if not isinstance(stage_outputs_as_input, list):
            raise TypeError(
                "`stage_outputs_as_input` must be a list and contain 4 elements.")

        S3_output, S5_output, S7_output, S9_output = stage_outputs_as_input

        # For the pyramid, we shall do top to bottom, so we start with stage 9.
        # Lets call the pyramid output as P_x, where x is the stage number
        # Do P_10. P_10 is just 3x3 convolution on S9_output
        P_10 = self.P_10_conv(S9_output.to(torch.float32)).to(torch.float32)

        # Stage 9 pyramid
        P_9 = self.S9_lateral_1x1_conv(S9_output.to(torch.float32))
        P_9 = self.conv3x3(P_9.to(torch.float32)).to(torch.float32)

        # Since stage 7 and 9 have the same spatial, that is HxW
        # dimensions, there is no need for feature upsampling.
        P_7 = self.S7_lateral_1x1_conv(S7_output.to(torch.float32))
        P_7 = P_9 + P_7
        P_7 = self.conv3x3(P_7.to(torch.float32)).to(torch.float32)

        # Remember to upsample P_7 to increase the spatial dimension, so
        # that the the spatial dimension will be compactible with lateral P_5
        P_5 = self.S5_lateral_1x1_conv(S5_output.to(torch.float32))
        upsampled_P_7 = self._2x_upsample(P_7.to(torch.float32))
        P_5 = upsampled_P_7 + P_5
        P_5 = self.conv3x3(P_5.to(torch.float32)).to(torch.float32)

        # Upsample P_5 to increase the spatial dimension, so
        # that the the spatial dimension will be compactible with lateral P_3
        P_3 = self.S3_lateral_1x1_conv(S3_output.to(torch.float32))
        upsampled_P_5 = self._4x_upsample(P_5.to(torch.float32))
        P_3 = upsampled_P_5 + P_3
        P_3 = self.conv3x3(P_3.to(torch.float32)).to(torch.float32)

        """
        The shapes using (3x512x512) image shape is 
        P_3 = torch.Size([N, C, 128, 128])
        P_5 = torch.Size([N, C, 32, 32])
        P_7 = torch.Size([N, C, 16, 16])
        P_9 = torch.Size([N, C, 16, 16])
        P_10 = torch.Size([N, C, 8, 8])

        As seen above, P_5 is 4 times smaller than P_3. I'd sort of prefer 2x smaller.
        If we upsample P_5 to torch.Size([N, C, 64, 64]), then P_7 at torch.Size([N, C, 16, 16])
        would also be 4x smaller instead of 2x. So and adhoc solution is to upscale P_5 and P_7.
        """
        P_5 = self._2x_upsample(P_5.to(torch.float32))
        P_7 = self._2x_upsample(P_7.to(torch.float32))

        # print(" P_3:{}, P_5:{}, P_7:{}, P_9:{}".format( P_3.shape, P_5.shape, P_7.shape, P_9.shape))
        return P_3, P_5, P_7, P_9, P_10
