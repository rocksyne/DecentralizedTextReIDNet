"""
Credit: 
Code was used in MHParsNet (https://github.com/rocksyne/MHParsNet) and adapted from
https://github.com/OpenFirework/pytorch_solov2/blob/master/modules/solov2_head.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import multi_apply
from model.model_utils import DepthwiseSeparableConv2d


INF = 1e8


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d


class ParsingHead(nn.Module):

    def __init__(self,
                 config=None,
                 # number of classes, plus the background (59 fo MHPv2)
                 num_classes=None,
                 in_channels=None,  # 256 fpn outputs
                 seg_feat_channels=256,  # seg feature channels
                 stacked_convs=2,  # solov2 light set 2
                 strides=(4, 8, 16, 32, 64),  # [8, 8, 16, 32, 32],
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128),
                               (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grids=None,  # [40, 36, 24, 16, 12],
                 ins_out_channels=64,  # 128
                 norm_cfg=None):

        super(ParsingHead, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs  # 2
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges

        # self.loss_cate = FocalLoss(use_sigmoid=True,gamma=2.0, alpha=0.25, loss_weight=1.0)   #build_loss Focal_loss

        self.ins_loss_weight = 3.0  # loss_ins['loss_weight']  #3.0
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(nn.Sequential(DepthwiseSeparableConv2d(chn, self.seg_feat_channels, 3, stride=1, padding=1, bias=norm_cfg is None),
                                                   nn.GroupNorm(num_channels=self.seg_feat_channels, num_groups=32),
                                                   nn.ReLU()))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(DepthwiseSeparableConv2d(chn, self.seg_feat_channels, 3, stride=1, padding=1, bias=norm_cfg is None),
                                                 nn.GroupNorm(num_channels=self.seg_feat_channels, num_groups=32), nn.ReLU()))

        self.solo_cate = DepthwiseSeparableConv2d(self.seg_feat_channels, self.cate_out_channels, 3, stride=1, padding=1, bias=True)

        self.solo_kernel = DepthwiseSeparableConv2d(self.seg_feat_channels, self.kernel_out_channels, 3, stride=1, padding=1, bias=True)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        cate_pred, kernel_pred = multi_apply(self.forward_single, new_feats, list(range(len(self.seg_num_grids))), eval=eval, upsampled_size=upsampled_size)
        return cate_pred, kernel_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear', align_corners=False))

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
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
        return cate_pred, kernel_pred