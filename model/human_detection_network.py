"""
Doc.:   TODO: I don't know what to name it yet so lets see
"""
from typing import List

# 3rd party modules
import torch
from torch import nn
from typing import NamedTuple
from typing import Tuple

# Apppication modules
from model.efficientnet_backbone import EfficientNetBackbone
from model.feature_pyramid_network import FeaturePyramidNetwork
from model.mask_head import MaskFeatHead
from model.parsing_head import ParsingHead
from model.process_parsing_result import ProcessParsingResult


class TestConfig(NamedTuple):
        nms_pre:int
        score_thr:float
        mask_thr:float
        update_thr:float
        kernel:str
        sigma:float
        max_per_img:int


class MetaData(NamedTuple):
        img_shape:Tuple[int,int]
        scale_factor:float
        ori_shape:Tuple[int,int]


class HumanDetectionNetwork(nn.Module):
    def __init__(self,configs=None)->list:
        """
        Doc.:   Model for performning person detection.

        Args.:  configs: system configurations found in ../config.py

        TODO: do propoer documentation
        """
        super(HumanDetectionNetwork, self).__init__()
        self.configs:dict = configs
        self.efficientnet_backbone = EfficientNetBackbone()
        self.FPN = FeaturePyramidNetwork()

        self.human_bbox_head = ParsingHead(num_classes=2, # either human exists or not
                                in_channels=256,
                                seg_feat_channels=256,
                                stacked_convs=2,
                                num_grids=[40, 36, 24, 16, 12],
                                ins_out_channels=128)
                
        self.human_mask_feat_head = MaskFeatHead(in_channels=256,
                                                 out_channels=128,
                                                 start_level=0,
                                                 end_level=1,
                                                 num_classes=128)
        
        self.process_parsing_result  = ProcessParsingResult(num_classes = 2,
                                                            ins_out_channels = 128,
                                                            num_grids = [40, 36, 24, 16, 12],
                                                            strides = [8, 8, 16, 32, 32])

        self.test_configurations = TestConfig(nms_pre=configs.nms_pre,
                                        score_thr=configs.score_thr,
                                        mask_thr=configs.mask_thr,
                                        update_thr=configs.update_thr,
                                        kernel= configs.kernel, 
                                        sigma=configs.sigma,
                                        max_per_img=configs.max_per_img)
        
        self.img_meta_data = MetaData(img_shape=configs.MHPV2_image_size, 
                                      scale_factor=1,
                                      ori_shape = configs.MHPV2_image_size)
        

        
    def forward(self,image, eval:bool=False, multi_person:bool=False):
        """
        Doc.:   Returns a set of bounding boxes for detected persons or 
                cate_pred, kernel_pred and human_mask_feat_pred predictions.
                see self.person_detection_model() for details.
        
        Args.:  - image: pro-processed image as torch.tensor. Shape is (N,C,H,W)
                - eval: values are True or False. Setting eval=True puts this class into evaluation mode.
                        Evaluation mode means that we get bounding box predictions as output rather than
                        predictions on which we can calculate losses. Setting eval=False does other-wise.
                        For proper context or details, please see MHParsNet (https://github.com/rocksyne/MHParsNet).
                - multi_person: values are True or False. If multi_person=True, model returns the bounding boxes for
                                all human instances in a single image. This is useful for makinf inference on a multi-person
                                image. If one desires the boding box for just one person in multi-person image, then
                                multi_person=False. This returns the bounding box cordinates for the person with the
                                heighest accuracy prediction.
        """
        stage_level_visual_features:List[torch.Tensor] = self.efficientnet_backbone(image) # output: S3_out, S5_out, S7_out, S9_out

        # Get the batch of the data
        data_batch = image.shape[0]

        meta_data = [self.img_meta_data for _ in range(data_batch)]
        person_detection_model_output = self.person_detection_model(stage_level_visual_features,eval=eval, meta_data=meta_data, multi_person=multi_person)
        return person_detection_model_output

    
    def person_detection_model(self,stage_level_visual_features:List[torch.Tensor], eval:bool=False, meta_data:list=None, multi_person=False)->torch.tensor:
        """
        Doc.:   This sub-model is responsible for person detection and segmentation.
                It performs multi-human parsing but it is limited to person detection for now. 
                TODO: show the paper
                Method should return the bounding box cordinates of identified persons only

        Args.:  - stage_level_visual_features: feature pyramid layer outputs
                - eval: infrence or trainign mode
        """
        P_3, P_5, P_7, P_9, P_10 = self.FPN(stage_level_visual_features) 
        pyramids_as_tuples = tuple([P_3, P_5, P_7, P_9, P_10])

        cate_pred, kernel_pred = self.human_bbox_head(pyramids_as_tuples,eval=eval)
        human_mask_feat_pred = self.human_mask_feat_head(pyramids_as_tuples[self.human_mask_feat_head.start_level:self.human_mask_feat_head.end_level + 1])

        # In evaluation mode, return the bounding boxes of the guman instances
        if eval:
            human_seg_inputs = (cate_pred, kernel_pred, human_mask_feat_pred, meta_data, False, multi_person)
            bounding_boxes = self.process_parsing_result(*human_seg_inputs)
            return bounding_boxes
        
        # Else return only predictions so that we compute losses
        # We do this split for computational reasons
        else:
            return cate_pred, kernel_pred, human_mask_feat_pred

        





