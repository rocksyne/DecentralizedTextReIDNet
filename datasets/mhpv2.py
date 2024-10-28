"""
Doc.:   Dataset manager for all Multiple-Human Parsing Dataset v2.0 (MHPv2) 
        See https://lv-mhp.github.io/ for details.

TODO: Provide more documentation
"""

# system modules
from __future__ import print_function, division
import os
import sys

# 3rd party modules
import torch
import numpy as np
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from utils.iotools import read_image
from utils.iotools import get_id_list_from_MHPv2_txt_file
from utils.mhpv2_dataset_utils import get_human_instance_masks_and_BBs_with_classes
from utils.mhpv2_dataset_utils import CustomMHPv2ImageResizer


# Global configurations
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MHPv2(Dataset):

    def __init__(self,
                 config:dict=None,
                 dataset_split: str = "train",
                 do_transfrom:bool = True):
        """
        Prepare the MHPv2 dataset for training and inference. 

        One image in the `images` directory corresponds to multiple annotation files in `parsing_annos` directory.
        The name of each annotation image starts with the same prefix as the image in the `images` directory.
        
        For example, given `train/images/96.jpg`, there are 3 humans in the image. The correspoding annotation files in 
        `train/parsing_annos/` are
            ・ 96_03_01.png, where 96 is the corresponding image name (96.jpg), 3 is # of person, and 1 is person # 1 ID
            ・ 96_03_02.png, where 96 is the corresponding image name (96.jpg), 3 is # of person, and 2 is person # 2 ID
            ・ 96_03_03.png, where 96 is the corresponding image name (96.jpg), 3 is # of person, and 3 is person # 3 ID

        Args.:  <tuple>dimension: output dimension of the dataset
                <str>dataset_split: type of dataset split. Acceptable values [`train`,`validation`]
                <bool>remove_problematic_images: remove images that are known to cause problems

        Return: <dict>samples: batch sample of the dataset. sample includes
                                ・ images, shape is [N,3,H,W]
                                ・ images_path, shape is [N,1]
                                ・ segmentation_label, shape is [N,1,H,W]
                                ・ segmentation_label_path, shape is [N,1]
                                ・ instance_ids, shape is [N]
                                ・ bounding_boxes, shape is [N,5]. 5 is 4 BB co-ordinates and 1 class
                                ・ segmentation_masks, shape is [N,]

                                TODO: get the dimension issue sorted out

        Corrupt images removed:
            1. train/images/1396.jpg
            2. train/images/18613.jpg
            3. train/images/19012.jpg
            4. train/images/19590.jpg
            5. train/images/24328.jpg

        Ref. / Credit: code adopted from https://github.com/RanTaimu/M-CE2P/blob/master/metrics/MHPv2/mhp_data.py

        Images:       images
        Category_ids: semantic part segmentation labels         Categories:   visualized semantic part segmentation labels
        Human_ids:    semantic person segmentation labels       Human:        visualized semantic person segmentation labels
        Instance_ids: instance-level human parsing labels       Instances:    visualized instance-level human parsing labels
        """
        self.config = config
        self.dataset_split = dataset_split
        self.do_transfrom = do_transfrom
        self.dataset_parent_dir = self.config.MHPv2_dataset_parent_dir
        self.allowed_dataset_split = ["train", "val"]
        self.part_classes = ["Cap/hat", "Helmet", "Face", "Hair", "Left-arm", "Right-arm", "Left-hand", "Right-hand",
                             "Protector", "Bikini/bra", "Jacket/windbreaker/hoodie", "Tee-shirt", "Polo-shirt", 
                             "Sweater", "Singlet", "Torso-skin", "Pants", "Shorts/swim-shorts",
                             "Skirt", "Stockings", "Socks", "Left-boot", "Right-boot", "Left-shoe", "Right-shoe",
                             "Left-highheel", "Right-highheel", "Left-sandal", "Right-sandal", "Left-leg", "Right-leg", "Left-foot", "Right-foot", "Coat",
                             "Dress", "Robe", "Jumpsuit", "Other-full-body-clothes", "Headwear", "Backpack", "Ball", "Bats", "Belt", "Bottle", "Carrybag",
                             "Cases", "Sunglasses", "Eyewear", "Glove", "Scarf", "Umbrella", "Wallet/purse", "Watch", "Wristband", "Tie",
                             "Other-accessary", "Other-upper-body-clothes", "Other-lower-body-clothes"]

        if self.dataset_split not in self.allowed_dataset_split:
            raise ValueError("Invalid value for `dataset_split` parameter")
        
        if (self.dataset_parent_dir is None) or (os.path.exists(self.dataset_parent_dir) is False):
            raise ValueError("Invalid value for `config.dataset_parent_dir` parameter.")

        self.rgb_img_resizer = CustomMHPv2ImageResizer(dimensions=config.MHPV2_image_size, image_type='RGB')
        self.greyscale_img_resizer = CustomMHPv2ImageResizer(dimensions=config.MHPV2_image_size, image_type='L')
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=config.MHPV2_means,std=config.MHPV2_stds)])

        # get all essenstial directories
        self.data_id_file = os.path.join(self.dataset_parent_dir, "list", self.dataset_split+".txt")
        self.image_dir = os.path.join(self.dataset_parent_dir, self.dataset_split, "images")
        self.parsing_annot_dir = os.path.join(self.dataset_parent_dir, self.dataset_split, "parsing_annos")
        self.category_ids_dir = os.path.join(self.dataset_parent_dir, self.dataset_split, "Category_ids")
        self.human_ids_dir = os.path.join(self.dataset_parent_dir, self.dataset_split, "Human_ids") 
        self.instance_ids_dir = os.path.join(self.dataset_parent_dir, self.dataset_split, "Instance_ids") 
        self.data_ids = get_id_list_from_MHPv2_txt_file(self.data_id_file)

        
        
        # some debug information
        print("")
        print("[INFO] Dataset name: Multi-human Parsing (MHP) V2")
        print("[INFO] Dataset split: ",self.dataset_split)
        print("[INFO] Number of classes: {:,}. (*please note that this does not include the background class)".format(len(self.part_classes)))
        print("[INFO] Total valid data samples: {:,}".format(len(self.data_ids)))
        print("[INFO] Some image data may have been corrupted during extracting, so remove them if you can")
        print("")

        

    def __len__(self):
        return len(self.data_ids)


    def __getitem__(self, idx):

        # Read / process (RGB) image
        img_path = os.path.join(self.image_dir,str(self.data_ids[idx])+".jpg")
        img = read_image(img_path,'RGB')
        img = self.rgb_img_resizer(img)

        if self.do_transfrom:
            img = self.transform(img)
        
        else: # just convert to tensor
            img = transforms.ToTensor()(img)

        # masks and bounding boxes for semantic person segmentation
        human_seg_annot_path = os.path.join(self.human_ids_dir,str(self.data_ids[idx])+".png")
        human_seg_annot = read_image(human_seg_annot_path,'L')
        human_seg_annot = self.greyscale_img_resizer(human_seg_annot)
        human_seg_annot = torch.tensor(np.array(human_seg_annot),dtype=torch.int).unsqueeze(0)
        human_instance_masks, human_instance_bbs_n_classes = get_human_instance_masks_and_BBs_with_classes(human_seg_annot, 20, img_path)

        sample = {"images":img,
                  "image_paths": img_path,
                  "human_instance_masks": human_instance_masks,
                  "human_instance_bbs_n_classes": human_instance_bbs_n_classes
                  }
        return sample



