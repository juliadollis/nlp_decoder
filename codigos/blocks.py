# TransformerBlock
# TODO: Complete o método forward da classe TransformerBlock abaixo.

import torch.nn as nn
from .mha import MultiHeadAttention
from .layers import FeedForward, LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        ##################################################################
        # TODO: Observe a arquitetura e monte os dois blocos: atenção e feed forward
        # Use a imagem da arquitetura para te auxiliar
        # Complete as linhas abaixo

        # ATTENTION BLOCK

        # Shortcut connection for attention block
        shortcut = x
        x = 
        x =   # Shape [batch_size, num_tokens, emb_size]
        x = 
        x =  x + shortcut.  # Add the original input back

        # FEED FORWARD BLOCK

        # Shortcut connection for feed forward block
        shortcut = x
        x = 
        x = 
        x = 
        x =   # Add the original input back

        return x