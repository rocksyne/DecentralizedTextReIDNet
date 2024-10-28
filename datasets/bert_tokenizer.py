"""
Doc.:   BERT tokenizer
        Adapterd from https://github.com/Suo-Wei/SRCF
        Paper link: https://link.springer.com/chapter/10.1007/978-3-031-19833-5_42
"""

import torch
import transformers as ppb

class BERTTokenizer(object):
    def __init__(self):
        super(BERTTokenizer, self).__init__()
        _ , tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    
    def __call__(self,text:str=None) -> torch.LongTensor:
        tokens = self.tokenizer.encode(text)
        return tokens
       
    
