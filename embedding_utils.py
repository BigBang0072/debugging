import numpy as np
import urllib 


class EmbeddingUtil():
    '''
    This class will manage all the embedding realted utilities.
    From loading (and or downloading if not available) and easy interface
    for any other class later.
    '''

    def __init__(self,embedding_args):
        self.embedding_args = embedding_args
    
    def _download_and_prepare_emb(self,):
        '''
        This function will download the embedding and prepare the 
        embedding in a easy to use manner.
        '''
        