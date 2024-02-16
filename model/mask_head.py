"""
Credit: 
Code was used in MHParsNet (https://github.com/rocksyne/MHParsNet) and adapted from
https://github.com/OpenFirework/pytorch_solov2/blob/master/modules/solov2_head.py
"""
import torch
import torch.nn as nn
from model.nninit import normal_init
from model.model_utils import DepthwiseSeparableConv2d


class MaskFeatHead(nn.Module):
    def __init__(self,
                 in_channels,  # 256
                 out_channels,  # 128
                 start_level,  # 0
                 end_level,  # 3
                 num_classes):
        """
        Doc.:   The start level and end level determine the number of convolution stacks
                start_level 0 is always the default, and end_level determines how many 
                convolution stacks to be used.
        """
        super(MaskFeatHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = nn.Sequential(DepthwiseSeparableConv2d(self.in_channels, self.out_channels, 3, stride=1, padding=1, bias=False),
                                         nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                                         nn.ReLU(inplace=True))
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels+2 if i == 3 else self.in_channels
                    one_conv = nn.Sequential(
                        DepthwiseSeparableConv2d(chn, self.out_channels, 3, stride=1, padding=1, bias=False),
                        nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                        nn.ReLU(inplace=True))
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module('upsample' + str(j), one_upsample)
                    continue

                one_conv = nn.Sequential(DepthwiseSeparableConv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1, bias=False),
                                         nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                                         nn.ReLU(inplace=True))
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(DepthwiseSeparableConv2d(self.out_channels, self.num_classes, 1, stride=1, padding=0, bias=False),
                                       nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                                       nn.ReLU(inplace=True))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, inputs):
        assert len(inputs) == (self.end_level - self.start_level + 1)

        feature_add_all_level = self.convs_all_levels[0](inputs[0])

        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3:
                input_feat = input_p
                x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
                y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([input_feat.shape[0], 1, -1, -1])
                x = x.expand([input_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                input_p = torch.cat([input_p, coord_feat], 1)

            feature_add_all_level = feature_add_all_level + self.convs_all_levels[i](input_p)

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred
