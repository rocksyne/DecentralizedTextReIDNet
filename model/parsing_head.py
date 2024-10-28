"""
Credit: 
Code was used in MHParsNet (https://github.com/rocksyne/MHParsNet) and adapted from
https://github.com/OpenFirework/pytorch_solov2/blob/master/modules/solov2_head.py
"""
from typing import List
from typing import Tuple

# 3rd party modules
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Application modules
from model.model_utils import multi_apply
from model.model_utils import DepthwiseSeparableConv


def points_nms(heat, kernel=2):
    hmax = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)(heat)
    hmax = hmax[:, :, :-1, :-1]
    keep = (hmax== heat).float()
    return heat * keep




class ParsingHead(nn.Module):

    def __init__(self,
                 num_classes=None,
                 in_channels=None,  # 256 fpn outputs
                 seg_feat_channels=256,  # seg feature channels
                 stacked_convs=2,  # light set 2
                 num_grids=None,  # [40, 36, 24, 16, 12],
                 ins_out_channels=64,  # 128
                 norm_cfg=None):

        super(ParsingHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs  # 2
        self.kernel_out_channels = self.ins_out_channels * 1 * 1

        self.ins_loss_weight = 3.0  # loss_ins['loss_weight']  #3.0
        self.norm_cfg = norm_cfg
        self._init_layers()
        

    def _init_layers(self):

        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
                                                            
            self.kernel_convs.append(nn.Sequential(DepthwiseSeparableConv(in_channels=chn, out_channels=self.seg_feat_channels, kernel_size=3, bias=False),
                                                   nn.BatchNorm2d(num_features=self.seg_feat_channels),
                                                   nn.ReLU(inplace=False)))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(DepthwiseSeparableConv(in_channels=chn, out_channels=self.seg_feat_channels, kernel_size=3, bias=False),
                                                nn.BatchNorm2d(num_features=self.seg_feat_channels),
                                                nn.ReLU(inplace=False)))

        self.solo_cate = DepthwiseSeparableConv(in_channels=self.seg_feat_channels, out_channels=self.cate_out_channels, kernel_size=3, bias=True)
        self.solo_kernel = DepthwiseSeparableConv(in_channels=self.seg_feat_channels, out_channels=self.kernel_out_channels, kernel_size=3, bias=True)
        #self.scale_down_conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feats:Tuple[torch.Tensor], eval:bool=False):
        cate_preds_list:List[torch.Tensor] = []
        kernel_preds_list:List[torch.Tensor] = []
        new_feats:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        # Iterate over each feature and its corresponding index
        for idx, new_feat in enumerate(new_feats, start=0):

            # Directly call self.forward_single with explicit arguments
            cate_pred, kernel_pred = self.forward_single(new_feat, idx, eval)
            
            # Collect the outputs
            cate_preds_list.append(cate_pred)
            kernel_preds_list.append(kernel_pred)

        return cate_preds_list, kernel_preds_list

    def split_feats(self, feats:List[torch.Tensor]):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear', align_corners=False))

    def forward_single(self, x, idx:int, eval:bool=False):
        ins_kernel_feat = x
        # ins branch
 
        # x_range = np.linspace(-1, 1, ins_kernel_feat.shape[-1], dtype=np.float32)
        # y_range = np.linspace(-1, 1, ins_kernel_feat.shape[-2], dtype=np.float32)
        # y, x = np.meshgrid(y_range, x_range)

        # x = torch.tensor(x, device=ins_kernel_feat.device)
        # y = torch.tensor(y, device=ins_kernel_feat.device)
        # y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        # x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])

        # coord_feat = torch.cat([x, y], 1)
        # ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear', align_corners=False)

        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)
        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
            #cate_pred = cate_pred.permute(0, 2, 3, 1)
        return [cate_pred, kernel_pred]