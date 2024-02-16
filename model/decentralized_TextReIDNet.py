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
from model.mask_head import MaskFeatHead
from model.parsing_head import ParsingHead


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

        self.human_bbox_head = ParsingHead(num_classes=2, # either human exists or not
                                in_channels=256,
                                seg_feat_channels=256,
                                stacked_convs=2,
                                strides=[8, 8, 16, 32, 32],
                                scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
                                num_grids=[40, 36, 24, 16, 12],
                                ins_out_channels=128)
        
        self.human_mask_feat_head = MaskFeatHead(in_channels=256,
                                                  out_channels=128,
                                                  start_level=0,
                                                  end_level=1,
                                                  num_classes=128)
        
        #self.language_network = GRULanguageNetwork(configs)

    def forward(self,image):
        # EfficentNet output
        stage_level_visual_features:list[torch.tensor] = self.efficientnet_backbone(image) # output: S3_out, S5_out, S7_out, S9_out
        human_outs, human_mask_feat_pred = self.person_detection_model(stage_level_visual_features)
        return human_outs, human_mask_feat_pred

    
    def person_detection_model(self,stage_level_visual_features:list[torch.tensor])->torch.tensor:
        """
        Doc.:   This sub-model is responsible for person detection and segmentation.
                It performs multi-human parsing. TODO: show the paper
        """
        P_3, P_5, P_7, P_9, P_10 = self.FPN(stage_level_visual_features) 
        pyramids_as_tuples = tuple([P_3, P_5, P_7, P_9, P_10])
        human_outs = self.human_bbox_head(pyramids_as_tuples)
        human_mask_feat_pred = self.human_mask_feat_head(pyramids_as_tuples[self.human_mask_feat_head.start_level:self.human_mask_feat_head.end_level + 1])
        return human_outs, human_mask_feat_pred
        





