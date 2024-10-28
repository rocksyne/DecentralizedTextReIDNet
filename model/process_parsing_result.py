"""
Credit:
Parts of code for MHParsNet was adopted from
from https://github.com/OpenFirework/pytorch_solov2/blob/master/modules/solov2_head.py
"""
import torch
import torch.nn as nn
from typing import Tuple
from typing import List
import torch.nn.functional as F

from model.model_utils import matrix_nms
from torchvision.ops import masks_to_boxes


class ProcessParsingResult(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 ins_out_channels: int = 128,
                 num_grids: list = [40, 36, 24, 16, 12],
                 strides: list = (4, 8, 16, 32, 64),  # [8, 8, 16, 32, 32],
                 score_thr:float = 0.3,
                 img_shape:tuple = (512,512),
                 scale_factor:float = 1.,
                 ori_shape:tuple = (512,512),
                 nms_pre:int = 500,
                 mask_thr:float = 0.5,
                 update_thr:float = 0.05,
                 kernel:str = 'gaussian',  # values: `gaussian` or `linear`
                 sigma:float = 2.0,
                 max_per_img:int = 30):

        super(ProcessParsingResult, self).__init__()
        self.cate_out_channels = num_classes - 1
        self.kernel_out_channels = ins_out_channels * 1 * 1
        self.seg_num_grids = num_grids
        self.strides = strides
        self.score_thr = score_thr
        self.img_shape = img_shape
        self.scale_factor = scale_factor
        self.ori_shape = ori_shape
        self.nms_pre = nms_pre
        self.mask_thr = mask_thr
        self.update_thr = update_thr
        self.kernel = kernel
        self.sigma = sigma
        self.max_per_img = max_per_img


    def forward(self, cate_preds:list, kernel_preds:list, seg_pred:torch.Tensor, img_metas:list, rescale:bool, multi_person:bool):

        #print("data types: cate_preds:{} kernel_preds:{} seg_pred:{} img_metas:{}".format(type(cate_preds), type(kernel_preds), type(seg_pred), type(img_metas)))
        result_list:List[List[torch.Tensor]] = []
        bounding_boxes_list:list = []
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        for img_id in range(len(img_metas)):
            cate_pred_list = [cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)]
            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
            kernel_pred_list = [ kernel_preds[i][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach() for i in range(num_levels)]
            
            img_shape = self.img_shape
            scale_factor = self.scale_factor
            ori_shape = self.ori_shape

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list, kernel_pred_list, featmap_size, img_shape, ori_shape, scale_factor, rescale)

            # we expect 3 list elements. If we get only one, then
            # what we got is a dummy.
            if len(result) == 1:
                continue

            elif len(result) == 3:
                result_list.append(result)


        if len(result_list) != 0:
            for result in result_list:

                seg_label = result[0]
                seg_label = seg_label.to(torch.uint8)
                score = result[2]
                vis_inds = score > self.score_thr
                seg_label = seg_label[vis_inds]
                cate_score = score[vis_inds]

                # After applying the threshold, we may get eleminate all the
                # predicted boxes. In such cases we dont want to be returning an empty list
                # So a work around is to return none for that image and continue with the rest

                # if we have 1 person, save the bounding boxes
                # On the other hand if we have many person and we are allowed 
                # to focus on all of them, then save such bbs as well
                if seg_label.shape[0] == 0:
                    bounding_boxes_list.append(None)
                    continue
                
                # Get bounding boxes for the bb predictions
                # that have passed the threshould. For some strange reason, sometimes I get bbs
                # as tensor([[341, 160, 403, 227]]) instead of tensor([341, 160, 403, 227]). 
                # So let us reshape so that we have uniformity. We make int also for indexing purpose
                bounding_boxes = masks_to_boxes(seg_label).int()

                if bounding_boxes.shape[0] == 1:
                    bounding_boxes_list.append(bounding_boxes.reshape(-1))
                    continue

                else: 
                    if multi_person:
                        for bb in bounding_boxes:
                            bounding_boxes_list.append(bb)
                    
                    else:
                        # choose just the topmost person identified and save bbs
                        max_confidence_index = torch.argmax(cate_score)
                        selected_bounding_box = bounding_boxes[max_confidence_index]
                        bounding_boxes_list.append(selected_bounding_box)

       
        return bounding_boxes_list


    def get_seg_single(self,
                       cate_preds:torch.Tensor,
                       seg_preds:torch.Tensor,
                       kernel_preds:torch.Tensor,
                       featmap_size:List[int],
                       img_shape:Tuple[int,int],
                       ori_shape:Tuple[int,int],
                       scale_factor:float,
                       rescale:bool=False)->List[torch.Tensor]:

        # # I am using this approach because I get the below warning if I do "if cate_preds.shape[0] != kernel_preds.shape[0]"
        # # TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. 
        # # We can't record the data flow of Python values, so this value will be treated as a constant in the future. 
        # # This means that the trace might not generalize to other inputs! if cate_preds.shape[0] != kernel_preds.shape[0
        # shape_mismatch = torch.tensor(cate_preds.shape[0]) != torch.tensor(kernel_preds.shape[0])
        # if shape_mismatch.item():
        #     raise Exception("Data shapes are no compactibele")

        # overall info.
        h, w = img_shape
            
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > self.score_thr)
        cate_scores = cate_preds[inds]

        if cate_scores.shape[0] == 0:
            return [torch.zeros(1, dtype=torch.uint8)]

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        #size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        size_trans = torch.tensor(self.seg_num_grids, device=cate_labels.device, dtype=cate_labels.dtype).pow(2).cumsum(0)

        #strides = kernel_preds.new_ones(size_trans[-1])
        strides = torch.ones(size_trans[-1], device=kernel_preds.device, dtype=kernel_preds.dtype)

        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds,stride=1).squeeze(0).sigmoid()
        
        seg_masks = seg_preds > self.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return [torch.zeros(1, dtype=torch.uint8)]

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if sort_inds.shape[0] > self.nms_pre:
            sort_inds = sort_inds[:self.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores, kernel=self.kernel, sigma=self.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= self.update_thr
        if keep.sum() == 0:
            return [torch.zeros(1, dtype=torch.uint8)]
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if sort_inds.shape[0] > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),size=upsampled_size_out, mode='bilinear', align_corners=False)[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds, size=ori_shape[:2], mode='bilinear', align_corners=False).squeeze(0)
        seg_masks = seg_masks > self.mask_thr
        return [seg_masks, cate_labels, cate_scores]


