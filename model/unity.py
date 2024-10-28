"""
Rev.:   Sunday 03.03.2024 @ 18:16 CET

Doc.:   This model forms the bases of the practical implementation of our Text-based Person Re-ID 
        model on the NVidia Jetson Nano.

        TODO: I don't know what to name it yet so lets see. For now I call it unity. My thought
        processes are a little hazy, so we will clean them up as we go
"""
# System modules
import os

# 3rd party modules
import PIL
import torch
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Apppication modules
from model.human_detection_network import HumanDetectionNetwork
from model.textreidnet import TextReIDNet


class Unity(nn.Module):
    def __init__(self,configs=None)->list:
        """
        Doc.:   Unified model for performning person detection (using bounding boxes), 
                as well as person re-id using text as query. This work is built on the following:
                [1] Real-time Multi-human Parsing on Embedded Devices, ICASSP2024
                [2] Resource-efficient Text-based Person Re-identification on Embedded Devices, DCOSS2024.

                This model unites MHParsNet from [1] and TextReIDNet from [2]. MHParsNet from [1] is modified for 
                person detection only, using bounding boxes. First, we detectect the segmentation maps of human 
                instances and then we extract the bounding boxes from the instance masks. These bounding boxes are then 
                used to crop human instances from images to be fed to TextReIDNet for person retrieval. 
                We ignore the component of human body part parsing. Addittionaly, we remove the ResNet18 backbone from 
                MHParsNet and replace it with EfficientNet-B0.

        Args.:  -config: configurations from the config file
        """
        super(Unity, self).__init__()
        self.configs:dict = configs
    
        # Prepare and load pretrained weights of the HumanDetectionNetwork.
        self.human_detection_network = HumanDetectionNetwork(configs) # for bb generation
        pretrained_human_detection_network_path = os.path.join(configs.model_save_path,"Human_Detection_Model.pth.tar")
        self.human_detection_network.load_state_dict(torch.load(pretrained_human_detection_network_path, map_location=torch.device('cpu')))

        # Prepare and load pretrained weights of the TextReIDNet.
        self.person_reID_network = TextReIDNet(configs)
        pretrained_person_reID_network_path  = os.path.join(configs.model_save_path,"TextReIDNet_Model.pth.tar")
        self.person_reID_network.load_state_dict(torch.load(pretrained_person_reID_network_path, map_location=torch.device('cpu')))
        self.cuhkpedes_transform = T.Compose([T.Resize(configs.CUHKPEDES_image_size, T.InterpolationMode.BICUBIC),
                                                T.Pad(10),
                                                T.RandomCrop(configs.CUHKPEDES_image_size),
                                                T.RandomHorizontalFlip(),
                                                #T.ToTensor(),
                                                T.Normalize(configs.CUHKPEDES_means,configs.CUHKPEDES_stds)])
        
        self.convert2tensor = T.ToTensor()


    def forward(self):
        raise NotImplementedError("Forward method not implemented.")
    

    def crop_image(self,original_img:PIL,bb_cordinates=torch.Tensor):
        x1 = bb_cordinates[0]
        y1 = bb_cordinates[1]
        x2 = bb_cordinates[2]
        y2 = bb_cordinates[3]
        return TF.crop(original_img, top=y1, left=x1, height=y2-y1, width=x2-x1)


    def persons_embeddings(self,original_image:PIL.Image, preprocessed_image:torch.Tensor):
        person_loc_as_bboxes = self.human_detection_network(image=preprocessed_image, eval=True, multi_person=True)

        person_images_with_bbox = [(self.crop_image(original_image,bbox), bbox) for bbox in person_loc_as_bboxes if bbox is not None and len(bbox) == 4]

        person_images, bboxes = zip(*person_images_with_bbox) if person_images_with_bbox else ([], [])

        if len(person_images) == 0:
            return None
        
        processed_person_images = [self.cuhkpedes_transform(img) for img in person_images]
        processed_person_images = torch.stack(processed_person_images,0)
        
        image_embedding = self.person_reID_network.image_embedding(processed_person_images.to(self.configs.device))
        return image_embedding, bboxes
  


   
    def target_persons_text_embeddings(self,text_ids:torch.tensor=None, text_length:torch.tensor=None)->torch.tensor:
        """
        Doc.:   Generate textual embeddings

        Args.:  • text_ids:     Text tokens (ids) Shape = (B,N), where B is batch and N is number of token IDs
                                See TextReIDNet/model/textpreidnet.py/datasets/bases.py for documentation
                • text_length:  Original length of the tokens before they were padded with 0 to the fixed length
                                Shape = (B,). See TextReIDNet/model/textpreidnet.py/datasets/bases.py for documentation
                                
        Return: torch.tensor
        """
        textual_features = self.person_reID_network.text_embedding(text_ids, text_length)
        return textual_features

        





