"""
Doc.:   Utility files for model / architecture
"""
import os
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from functools import partial
from six.moves import map, zip


# +++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++[Utility Functions]+++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++
def weights_init_kaiming(m, pi=0.01):
    """
    Doc.:   Custom weigh initialization method.
            Adapted from https://github.com/xx-adeline/MFPE/blob/main/src/model/model.py#L9
            to include RetinaNet-specific bias initialization https://arxiv.org/pdf/1708.02002.pdf

    Args.:  • pi: reference https://arxiv.org/pdf/1708.02002.pdf for details
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        
        # Apply RetinaNet-specific bias initialization to the final convolutional layer
        # Reference: https://arxiv.org/pdf/1708.02002.pdf
        if hasattr(m, 'is_final_conv') and m.is_final_conv:
            bias_value = -init.log((1 - pi) / pi)
            init.constant_(m.bias.data, bias_value)

    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay 
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
            map the multiple outputs of the ``func`` into different
            list. Each list contains the same type of outputs corresponding
            to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


        
# +++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++[Utility Classes]++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++
class Swish(nn.Module):
    def __init__(self, beta:float=1.0):
        """
        Doc.:   Swish activation function. https://arxiv.org/pdf/1710.05941.pdf
                Swish is an activation function f(x) = x·sigmoid(βx),  where β is a learnable parameter. 
                Nearly all implementations do not use the learnable parameter, in which case the activation 
                function is f(x) = x·sigmoid(βx). https://paperswithcode.com/method/swish

        Args.:  • beta: learnable or constant value
        """
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels:int=None, out_channels:int=None, kernel_size:int=3, bias:bool=True, is_final_conv:bool=False):
        """
        Doc.:   Custom depthwise separable convolution. https://arxiv.org/pdf/1610.02357.pdf
                Depthwise Separable Convolution splits the computation into two steps: depthwise convolution applies a 
                single convolutional filter per each input channel and pointwise convolution is used to create a 
                linear combination of the output of the depthwise convolution. https://paperswithcode.com/method/depthwise-separable-convolution

        Args.:  • in_channels: number of input channels
                • out_channels: number of output channels
                • bias: use bias or not
                • is_final_conv: treat this as the final layer or not. If last layer, use RetinaNet styled initialization. https://arxiv.org/pdf/1708.02002.pdf
        """
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_convolution = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias, padding=1)
        self.pointwise_convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=1, bias=bias)
        self.depthwise_separable_conv = torch.nn.Sequential(self.depthwise_convolution, self.pointwise_convolution)

        # Treat this as the final layer
        self.is_final_conv = is_final_conv
        if bias and is_final_conv:
            self.is_final_conv = True

        # initialize some weights
        self.depthwise_separable_conv.apply(weights_init_kaiming)

    def forward(self, x):
        return self.depthwise_separable_conv(x)



class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels:int=None, 
                 out_channels:int=None, 
                 kernel_size:int=None, 
                 stride:int=1, 
                 padding:int=0, 
                 bias:bool=True):
        """
        Doc.:   Custom depthwise separable convolution. https://arxiv.org/pdf/1610.02357.pdf
                Depthwise Separable Convolution splits the computation into two steps: depthwise convolution applies a 
                single convolutional filter per each input channel and pointwise convolution is used to create a 
                linear combination of the output of the depthwise convolution. https://paperswithcode.com/method/depthwise-separable-convolution

        Args.:  • in_channels: number of input channels
                • out_channels: number of output channels
                • kernel_size: size of the kernel (int or typle)
                • stride: stride of the kernel
                • padding: padding to be used
                • bias: use bias or not
                • bias: do a bias or not
                • is_final_conv: treat this as the final layer or not. If last layer, use RetinaNet styled initialization. https://arxiv.org/pdf/1708.02002.pdf
        """
        super().__init__()

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        groups=in_channels,  # Each input channel has its own filter
                                        bias=bias)

        # Pointwise convolution
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1,  # 1x1 convolution
                                        bias=bias)
        
        self.depthwise_separable_conv = torch.nn.Sequential(self.depthwise_conv, self.pointwise_conv)
        self.depthwise_separable_conv.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x 










