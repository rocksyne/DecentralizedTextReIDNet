"""
Credit: Design consideration inspired by https://arxiv.org/pdf/2303.08466.pdf
"""

# 3rd party modules
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRULanguageNetwork(nn.Module):
    def __init__(self,config:dict=None):
        """
        Doc.:   GRU Network
        Args.:  • config: dot-element accessible dictionary for configurations
        """
        super(GRULanguageNetwork, self).__init__()

        if config is None:
            raise ValueError("`config` can not be none")
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gru = nn.GRU(config.embedding_dim, config.feature_length, num_layers=config.num_layers, bidirectional=True, bias=False)


    def forward(self, text_ids:torch.Tensor=None, text_length:torch.Tensor=None)->torch.Tensor:
        """
        Doc.:   Do GRU computaion and return learned features

        Args.:  • text_ids:     Text tokens (ids) Shape = (B,N), where B is batch and N is number of token IDs
                                See TextReIDNet/model/textpreidnet.py/datasets/bases.py for documentation
                • text_length:  Original length of the tokens before they were padded with 0 to the fixed length
                                Shape = (B,). See TextReIDNet/model/textpreidnet.py/datasets/bases.py for documentation
        
        Return: torch.Tensor
        """
        text_embedding = self.embedding(text_ids)
        text_embedding = self.dropout(text_embedding)
        feature = self.do_RNN_computation(text_embedding, text_length, self.gru)
        return feature


    def do_RNN_computation(self, text_embedding, text_length, gru):
        """
        Doc.:   Do GRU-RNN computation

        Args.:  • text_embedding: output of the embeddings layer
                • text_length:  Original length of the tokens before they were padded with 0 to the fixed length
                                Shape = (B,). See TextReIDNet/model/textpreidnet.py/datasets/bases.py for documentation
                • gru: instance of GRU

        Return: torch.Torch
        """
        text_length = text_length.view(-1)
        _, sort_index = torch.sort(text_length, dim=0, descending=True)
        _, unsort_index = sort_index.sort()

        sortlength_text_embedding = text_embedding[sort_index, :]
        sort_text_length = text_length[sort_index]
        packed_text_embedding = pack_padded_sequence(sortlength_text_embedding,sort_text_length.cpu(),batch_first=True)

        packed_feature, hn = gru(packed_text_embedding)  # Only hn for GRU
        total_length = text_embedding.size(1)
        sort_feature = pad_packed_sequence(packed_feature, batch_first=True, total_length=total_length) # Including [feature, length]

        unsort_feature = sort_feature[0][unsort_index, :]
        unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]+ unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2
        return unsort_feature.permute(0, 2, 1).contiguous().unsqueeze(3)
