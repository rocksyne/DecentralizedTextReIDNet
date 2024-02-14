# system modules
import os
import datetime
import json
from statistics import mean

# installed modules
import torch
from PIL import Image
import numpy as np
import cv2 as cv
from tqdm import tqdm
import pathlib
from torch.nn.utils import clip_grad
from scipy import ndimage



# =========================================================================
# ===================[ Function Based Utilities ]==========================
# =========================================================================
def get_part_instance_masks_and_BBs_with_classes(segmentation: torch.Tensor = None,
                                                 smallest_bb_area: int = 10,
                                                 helper_data: dict = None,
                                                 segmentation_path: str = None):
    """
    Get/extract segmentation masks and bounding boxes from segmentations
    Ref.: https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html

    Args.:  <torch.Tensor>segmentation: segmentation of shape [1,H,W]
                    <int>smallest_bb_area: the smallest allowable bounding box area
                    <str>segmentation_path: name of the segmentation image for debugging purposes

    Return: <dict> `instance_ids`, `instance_masks`, and `bounding_boxes_n_classes`
    TODO: Take care of person class indexing
    """
    # security checkpoint - you shall not pass hahahahaha
    if not isinstance(segmentation, torch.Tensor):
        raise TypeError("Segmentation must be a torch.Tensor")

    if not segmentation_path is None:
        raise ValueError("Segmentation file path is required")

    obj_ids = torch.unique(segmentation)  # get unique segmentations
    obj_ids = obj_ids[1:]  # remove the background class

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = segmentation.type(torch.uint8) == obj_ids[:, None, None]
    masks = masks.type(torch.uint8)

    # genereate valid bounding boxes only if we have masks
    assert(masks.shape[0] > 0), "No masks available for {}".format(
        segmentation_path)
    mask_2_bb = masks_to_boxes(masks)  # get bounding boxes
    assert(masks.shape[0] == mask_2_bb.shape[0]
           ), "Masks count do not correspond to BB count for {}".format(segmentation_path)

    # Get only the valid bounding boxes. The boxes are of shape (x1, y1, x2, y2), where (x1, y1)
    # specify the top-left box corner, and (x2, y2) specify the bottom-right box corner.
    # see https://pytorch.org/vision/main/generated/torchvision.ops.masks_to_boxes.html and
    # https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/boxes.py for details
    hw = segmentation.shape[1:]  # (H,W) dimension of the image or segmentation
    box_list, mask_list, class_list, human_id_list = [], [], [], []
    for idx in range(len(mask_2_bb)):
        bbox = mask_2_bb[idx]
        mask = masks[idx]

        inst_class_id = obj_ids[idx]
        part_n_human_id = helper_data[int(inst_class_id)]
        # convert list to np for easy convertion to torch.Tensor
        part_n_human_id = np.array(part_n_human_id)
        part_n_human_id = torch.from_numpy(part_n_human_id)
        part_id, human_id = part_n_human_id

        #  part_id -= 1 # take care of indexing
        #  human_id -= 1 # take care of indexing

        x1, y1, x2, y2 = bbox

        if (0 <= x1 < x2) and (0 <= y1 < y2) is False:
            continue

        box_list.append(bbox)
        mask_list.append(mask)
        class_list.append(part_id)
        human_id_list.append(human_id)

    # type cast for torch operations
    box_list = torch.stack(box_list)
    mask_list = torch.stack(mask_list)
    class_list = torch.stack(class_list)
    human_id_list = torch.stack(human_id_list)

    box_list = clip_boxes(box_list, hw)  # clip boxes to remove negatives
    box_list, mask_list, class_list, human_id_list = remove_small_box(
        box_list, mask_list, class_list, human_id_list, smallest_bb_area)

    # create a canvas for bounding boxes.
    # The shape of this canvas is (num_boxes,6),
    # where 5 represents (x1, y1, x2, y2,object_or_class_ID,human_id)
    bounding_boxes = torch.zeros((len(box_list), 6))
    bounding_boxes[:, :4] = box_list
    bounding_boxes[:, -2] = class_list
    bounding_boxes[:, -1] = human_id_list

    # # re-assemble the masks, taking into account all the smaller masks
    # # that have been removed.
    # instance_masks = merge_masks_with_instances(mask_list,class_list) # dim = (H,W)
    # instance_masks = instance_masks[None,:,:] # expand dimension to become (C,H,W)

    return {"instance_masks": mask_list, "bounding_boxes_n_classes": bounding_boxes}


def get_human_instance_masks_and_BBs_with_classes(segmentation: torch.Tensor = None,
                                                  smallest_bb_area: int = 10,
                                                  segmentation_path: str = None):
    """
    Get/extract segmentation masks and bounding boxes from segmentations
    Ref.: https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html

    Args.:  <torch.Tensor>segmentation: segmentation of shape [1,H,W]
                    <int>smallest_bb_area: the smallest allowable bounding box area
                    <str>segmentation_path: name of the segmentation image for debugging purposes

    Return: <dict> `instance_ids`, `instance_masks`, and `bounding_boxes_n_classes`
    """
    # security checkpoint - you shall not pass hahahahaha
    if isinstance(segmentation, torch.Tensor) is False:
        raise TypeError("Invalid value for `segmentations` parameter")

    if segmentation_path is None:
        raise ValueError("Please specify your segmentaionf file path")

    obj_ids = torch.unique(segmentation)  # get unique segmentations
    # obj_ids_plus_bkgrnd = obj_ids.copy()
    obj_ids = obj_ids[1:]  # remove the background class

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = segmentation.type(torch.uint8) == obj_ids[:, None, None]

    # genereate valid bounding boxes only if we have masks
    assert(masks.shape[0] > 0), "No masks available for {}".format(segmentation_path)
    mask_2_bb = masks_to_boxes(masks)  # get bounding boxes
    assert(masks.shape[0] == mask_2_bb.shape[0]), "Masks count do not correspond to BB count for {}".format(segmentation_path)

    # Get only the valid bounding boxes. The boxes are of shape (x1, y1, x2, y2), where (x1, y1)
    # specify the top-left box corner, and (x2, y2) specify the bottom-right box corner.
    # see https://pytorch.org/vision/main/generated/torchvision.ops.masks_to_boxes.html and
    # https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/boxes.py for details
    hw = segmentation.shape[1:]  # (H,W) dimension of the image or segmentation
    box_list, mask_list, class_list, human_id_list = [], [], [], []
    for idx in range(len(mask_2_bb)):
        bbox = mask_2_bb[idx]
        mask = masks[idx]

        x1, y1, x2, y2 = bbox

        if (0 <= x1 < x2) and (0 <= y1 < y2) is False:
            continue

        box_list.append(bbox)
        mask_list.append(mask)
        # there is just 1 class for humans
        class_list.append(torch.from_numpy(np.array(1, dtype=np.int64)))
        human_id_list.append(obj_ids[idx])

    # type cast for torch operations
    box_list = torch.stack(box_list)
    mask_list = torch.stack(mask_list)
    class_list = torch.stack(class_list)
    human_id_list = torch.stack(human_id_list)

    box_list = clip_boxes(box_list, hw)  # clip boxes to remove negatives
    box_list, mask_list, class_list, human_id_list = remove_small_box(box_list, 
                                                                      mask_list, 
                                                                      class_list, 
                                                                      human_id_list, 
                                                                      smallest_bb_area)
    # create a canvas for bounding boxes.
    # The shape of this canvas is (num_boxes,6),
    # where 5 represents (x1, y1, x2, y2,object_or_class_ID,human_id)
    bounding_boxes = torch.zeros((len(box_list), 6))
    bounding_boxes[:, :4] = box_list
    bounding_boxes[:, -2] = class_list
    bounding_boxes[:, -1] = human_id_list

    # # re-assemble the masks, taking into account all the smaller masks
    # # that have been removed.
    # instance_masks = merge_masks_with_instances(mask_list,class_list) # dim = (H,W)
    # instance_masks = instance_masks[None,:,:] # expand dimension to become (C,H,W)
    return mask_list, bounding_boxes


def clip_boxes(boxes, hw):
    """
    Clip (limit) the values in a bounding box array. Given an interval, values outside the
    interval are clipped to the interval edges. For example, if an interval of [0, 1] is
    specified, values smaller than 0 become 0, and values larger than 1 become 1.
    """
    boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0, max=hw[1] - 1)
    boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0, max=hw[0] - 1)
    return boxes


def remove_small_box(boxes, masks, labels, human_ids, area_limit):
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = box_areas > area_limit
    return boxes[keep], masks[keep], labels[keep], human_ids[keep]


def normalize_BB_to_01(bounding_boxes, image_dimension):
    """Normalizes bounding boxes (BB) to intervals between 0 and 1"""
    h, w = image_dimension
    bounding_boxes[:, [0, 2]] = torch.div(bounding_boxes[:, [0, 2]], w)
    bounding_boxes[:, [1, 3]] = torch.div(bounding_boxes[:, [1, 3]], h)
    return bounding_boxes


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Doc.:	Taken from https://pytorch.org/vision/main/_modules/torchvision/ops/boxes.html#masks_to_boxes
                Compute the bounding boxes around the provided masks.

                Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` 
                        format with `0 <= x1 < x2` and `0 <= y1 < y2`.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
        and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]
    bounding_boxes = torch.zeros(
        (n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)
    return bounding_boxes


def clean_data_recieved_from_collator(bbs_n_labels:None,instance_masks:None):

	if bbs_n_labels.shape[0] != instance_masks.shape[0]:
		raise ValueError("Invalid parameters for either `bbs_n_labels` or `instance_masks`")

	fetched_batch = bbs_n_labels.shape[0]
	gt_bboxes = []
	gt_labels = []
	gt_masks = []
	for idx in range(fetched_batch):
		current_bbs_n_labels = bbs_n_labels[idx, :, :]
		current_bbs_n_labels = current_bbs_n_labels[current_bbs_n_labels[:, -1] != -1] # remove all negative vaulues that were used for padding
		
		# prepare gt bounding boxes
		bbox = current_bbs_n_labels[:,:4]
		gt_bboxes.append(bbox)

		# prepare gt labels
		label = current_bbs_n_labels[:,4].long()
		num_valid_labels = label.shape[0]
		gt_labels.append(label)

		# prepare GT masks
		# lets simply use the number of labels to fetch the number of valid masks
		# remember we have some labels that have -1 values for collation padding purposes
		masks = instance_masks[idx]
		masks = masks[:num_valid_labels,:,:] # fetch only valid masks. All arrays with -1 values are removed
		masks = masks.to(torch.uint8)
		gt_masks.append(masks)
	
	return gt_bboxes, gt_masks, gt_labels


# =========================================================================
# =======================[ Class Based Utilities ]=========================
# =========================================================================
class CustomMHPv2ImageResizer(object):
    def __init__(self, dimensions:tuple=(512, 512), image_type:str='RGB'):
        """
        Doc.:	Resize image and place on a black canvas to get a square shape.
        		In resizing label segmentations, the type of iterpolation is important. A wrong interpolation
				will result in loosing the instance segmentations. See discussion on this at
				https://stackoverflow.com/a/67076228/3901871. For RGB images, use any desired
				interpolation method, but for labels, use nearest neighbor. In the resize_image()
				method, `nearest` is set as the default interpolation method.
                
        Args. 	- dimensions: the expected width and height image
				- inter: the intepolation method
        """
        self.target_height, self.target_width = dimensions
        self.image_type = image_type

        if self.image_type == 'RGB':
            self.interpolation = Image.BILINEAR
            self.canvas_color = (0,0,0) # (R,G,B)
        
        elif self.image_type == 'L':
            self.interpolation = Image.NEAREST
            self.canvas_color = 0 # greyscale black
                
    def __call__(self,image:Image=None)->Image:
        # Determine the original dimensions
        original_width, original_height = image.size

        # Calculate the scaling factor and new size to maintain aspect ratio
        scaling_factor = min(self.target_width / original_width,
                             self.target_height / original_height)
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)

        # Resize the image with the specified interpolation method
        resized_image = image.resize(
            (new_width, new_height), self.interpolation)

        # Create a new image with a black background
        new_image = Image.new(self.image_type, (self.target_width, self.target_height), self.canvas_color)

        # Calculate the position to paste the resized image in the center
        paste_x = (self.target_width - new_width) // 2
        paste_y = (self.target_height - new_height) // 2

        # Paste the resized image onto the new image, centered
        new_image.paste(resized_image, (paste_x, paste_y))

        return new_image
