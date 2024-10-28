"""
Doc.:   tokenizer provided by OpenAI (https://github.com/openai/tiktoken)
        TODO: provide for special characters for <startoftext> <endoftext>
"""
import tiktoken
import torch

class TikTokenizer(object):

    def __init__(self, encoding_base:str=None, truncate=True):
        super(TikTokenizer, self).__init__()
        self.truncate = truncate
        self.encoding = tiktoken.get_encoding(encoding_base)

    def encode(self,text:str=None):
        return self.encoding.encode(text)
    
    def decode(self,tokens:list=[]):
        return self.encoding.decode(tokens)
    
    def __call__(self,text:str=None) -> torch.LongTensor:
        tokens = self.encoding.encode(text)
        return tokens
    
