# System modules

# 3rd party modules
import torch
from torch import nn

# Aplication modules
from model.visual_network import VisualNetwork
from model.language_network import GRULanguageNetwork
from model.model_utils import DepthwiseSeparableConv


class TextReIDNet(nn.Module):
    def __init__(self,configs:dict=None):
        """
        Doc.:   The architecture of TextReIDNet. Deatils in TODO: provide paper ref

                Architecture Summary
                ---------------------
                                                                                    -----------------
                image >> EfficientNetb0 >> DSC >> AP >> DSC >>|=================|---| Ranking loss  |
                                                              | Joint embedding |   -----------------
                text  >> BERT >>|Tokens|>> GRU >> AP >> DSC >>|=================|---| Identity loss |
                                                                                    -----------------
                Legends
                --------------------------------------
                • DSC: Deepthwise Seperable Convolution
                • AP: Adaptive Pooling (max)
                • GRU: Language network made from Gated Recurrent Units

                BERT is used within the text data preprocessing pipeline, rather than being integrated into TextReIDNet.
        """
        super(TextReIDNet, self).__init__()

        if configs is None:
            raise ValueError("`configs` parameter can not be None")
        
        self.configs = configs
        self.visual_network = VisualNetwork()
        self.language_network = GRULanguageNetwork(configs)
        self.visual_features_downscale = DepthwiseSeparableConv(1280, 1024)
        self.adaptive_max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.depthwise_seperable_convolution = DepthwiseSeparableConv(1024, self.configs.feature_length)

    def forward(self, image, text_ids:torch.tensor=None, text_length:torch.tensor=None)->list:
        """
        Doc.:   Perform feature extraction and conevert to embeddings

        Args.:  • image:        Batch of images. shape = (B,C,H,W), where B is the batch
                • text_ids:     Text tokens (ids) Shape = (B,N), where B is batch and N is number of token IDs
                                See TextReIDNet/model/textpreidnet.py/datasets/bases.py for documentation
                • text_length:  Original length of the tokens before they were padded with 0 to the fixed length
                                Shape = (B,). See TextReIDNet/model/textpreidnet.py/datasets/bases.py for documentation

        Return: list[torch.tensor] of visual and textual embeddings
        """
        visaual_embeddings = self.image_embedding(image)
        textual_embeddings = self.text_embedding(text_ids, text_length)
        return visaual_embeddings, textual_embeddings
    

    def image_embedding(self,image:torch.tensor=None)-> torch.tensor:
        """
        Doc.:   Generate image embeddings
        Args.:  • image:    Batch of images. shape = (B,C,H,W), where B is the batch  
        Return: torch.Tensor
        """
        visual_features = self.visual_network(image)
        visual_features = self.visual_features_downscale(visual_features)
        visual_features = self.adaptive_max_pooling(visual_features)
        visual_features = self.depthwise_seperable_convolution(visual_features).squeeze(-1).contiguous()
        return visual_features
    

    def text_embedding(self,text_ids:torch.tensor=None, text_length:torch.tensor=None)->torch.tensor:
        """
        Doc.:   Generate textual embeddings

        Args.:  • text_ids:     Text tokens (ids) Shape = (B,N), where B is batch and N is number of token IDs
                                See TextReIDNet/model/textpreidnet.py/datasets/bases.py for documentation
                • text_length:  Original length of the tokens before they were padded with 0 to the fixed length
                                Shape = (B,). See TextReIDNet/model/textpreidnet.py/datasets/bases.py for documentation
                                
        Return: torch.tensor
        """
        textual_features = self.language_network(text_ids,text_length)
        textual_features, _ = torch.max(textual_features, dim=2, keepdim=True)
        textual_features = self.depthwise_seperable_convolution(textual_features).squeeze(-1).contiguous()
        return textual_features


    