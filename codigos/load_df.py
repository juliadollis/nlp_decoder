# data and tokenizer
from datasets import load_dataset
from tqdm import tqdm
import itertools

# ========== CONFIG ==========
SAMPLE_SIZE = 5000   #num examples

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