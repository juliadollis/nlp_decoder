# TODO: Para completar:
# - MHA (mha.py)
# - TransformerBlock (blocks.py)
# - GPTModel (model.py)

import argparse
from importlib.metadata import version
import tiktoken
import torch
import torch.nn as nn
from codigos import (
    TransformerBlock,
    LayerNorm, FeedForward, GELU,
    GPTModel,
    GPTDatasetV1, create_dataloader_v1,
    generate_text_simple,
)

# -------------------------------------------------------
# Argumentos de linha de comando
# -------------------------------------------------------
parser = argparse.ArgumentParser(description="Testes iniciais GPT")
parser.add_argument(
    "--start-context",
    type=str,
    default="Olá, eu sou",
    help="Texto inicial para geração. Default: 'Olá, eu sou'"
)
parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=6,
    help="Número máximo de novos tokens a gerar"
)
args = parser.parse_args()

# -------------------------------------------------------
# Configuração do modelo
# -------------------------------------------------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# -------------------------------------------------------
# Teste do TransformerBlock
# -------------------------------------------------------
print("=== Testing TransformerBlock ===")
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("===============================\n")

# -------------------------------------------------------
# Teste do tokenizer
# -------------------------------------------------------
print("=== Testing tokenizer ===")
tokenizer = tiktoken.get_encoding("gpt2")

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
print("=========================\n")

# -------------------------------------------------------
# Teste do GPTModel
# -------------------------------------------------------
print("=== Testing GPTModel ===")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

total_size_mb = (total_params * 4) / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")
print("=========================\n")

# -------------------------------------------------------
# Teste de geração de texto
# -------------------------------------------------------
print("=== Testing text generation ===")
start_context = args.start_context
print(f"Usando start_context: \"{start_context}\"")

encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=args.max_new_tokens,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output (token ids):", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print("Decoded:\n", decoded_text)
print("==============================\n")