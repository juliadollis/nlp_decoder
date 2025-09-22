import os, math, time, random
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
import urllib.request
from importlib.metadata import version
from blocks import TransformerBlock
from layers import LayerNorm, FeedForward, GELU
from model import GPTModel
from data import GPTDatasetV1, create_dataloader_v1
from generate import generate_text_simple   
from train_utils import calc_loss_batch, plot_losses, calc_loss_loader, train_model_simple   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


tokenizer = tiktoken.get_encoding("gpt2")

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference








# **Model**

block_size = 64    # context window for training (<= model's context_length)
batch_size = 32    # adjust according to VRAM

cfg = GPT_CONFIG_124M.copy()
# cfg.update({"emb_dim": 384, "n_heads": 6, "n_layers": 6})

cfg["context_length"] = max(cfg["context_length"], block_size)

model = GPTModel(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Modelo com {n_params/1e6:.1f} M parâmetros")



# data and tokenizer
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import itertools

# ========== CONFIG ==========
SAMPLE_SIZE = 500  #num examples

# ========== TOKENIZER ==========
tokenizer = tiktoken.get_encoding("gpt2")

# ========== LOAD DATA (STREAMING) ==========
print("Carregando dataset em streaming...")
stream_ds = load_dataset("chenuneris/news-brazillian-clean", split="train", streaming=True)

texts = []
for ex in tqdm(itertools.islice(stream_ds, SAMPLE_SIZE), total=SAMPLE_SIZE, desc="Lendo dataset"):
    texts.append(ex["text"])

text_data = "\n".join(texts)

print(text_data[:500])  

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("\n--- RESULTADOS ---")
print("Characters:", total_characters)
print("Tokens:", total_tokens)



train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)





# Sanity check

if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")
    


print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)



train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)





model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes


torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)




torch.manual_seed(123)
#model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=100, eval_iter=100,
    start_context="O que ", tokenizer=tokenizer
)


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)





model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("O nazismo", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))