# MultiHeadAttention
## TODO: Complete o método forward da classe MultiHeadAttention abaixo.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out deve ser divisível pelo número de cabeças"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  
        self.dropout = nn.Dropout(dropout)

        # Máscara causal (triângulo superior)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        ########################################################################
        # TODO: Forward da Atenção Multi-Head
        #
        # 1) Projeções Q, K, V  
        #    - Aplique W_query/W_key/W_value em x.
        #    - Shapes após projeção: (b, num_tokens, d_out) para cada um.
        #
        # 2) Separar em hads  
        #    - reshapes: (b, num_tokens, d_out) -> (b, num_heads, num_tokens, head_dim)
        #    - use .view(...) e depois .transpose(1, 2) para ficar (b, num_heads, num_tokens, head_dim).
        #    - faça isso para Q, K e V.
        #
        # 3) Scores de atenção  
        #    - attn_scores = Q @ K^T / sqrt(head_dim).
        #    - Shape: (b, num_heads, num_tokens, num_tokens).
        #
        # 4) Máscara causal  
        #    - recorte a máscara para num_tokens e converta para boolean.
        #    - aplique com masked_fill_ nas posições mascaradas com -inf.
        #
        # 5) Softmax + dropout  
        #    - softmax em dim=-1 para virar pesos; aplique dropout.
        #
        # 6) Aplicar pesos em V  
        #    - context = attn_weights @ V  -> (b, num_heads, num_tokens, head_dim).
        #
        # 7) Juntar cabeças  
        #    - transpose de volta para (b, num_tokens, num_heads, head_dim),
        #      depois .contiguous().view(b, num_tokens, d_out).
        #
        # 8) Projeção final 
        #    - context = self.out_proj(context).
        #
        # Retorno: (b, num_tokens, d_out)
        ########################################################################

        context_vec = ...
        return context_vec