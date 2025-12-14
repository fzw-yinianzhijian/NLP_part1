import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        N=self.n_head
        
        assert C%N==0
        
        ## TODO: Implement causal self attention
        
        Q=self.query(x)
        K=self.key(x)
        V=self.value(x)
        
        Q=Q.view(B,T,N,C//N).transpose(1,2)
        K=K.view(B,T,N,C//N).transpose(1,2)
        V=V.view(B,T,N,C//N).transpose(1,2)
        
        # B N T C//N 
        
        att=Q@K.transpose(-2,-1)/math.sqrt(K.size(-1))
        
        att=att.masked_fill(self.mask[:,:,:T,:T]==0,float(-1e6))
        
        att=F.softmax(att,dim=-1)
        att=self.attn_drop(att)
        
        y=att@V
        y=y.transpose(1,2).contiguous().view(B,T,C)
        
        y=self.resid_drop(self.proj(y))
        
        return y
        ### TODO END
