from transformers import AutoTokenizer
from data_utils import load_all_transcripts
from pathlib import Path

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
df = load_all_transcripts(Path("../Transcript Data"))

token_lengths = df["text"].astype(str).apply(lambda x: len(tokenizer.encode(x)))

print(f"Total chunks:  {len(token_lengths)}")
print(f"Max tokens:    {token_lengths.max()}")
print(f"Mean tokens:   {token_lengths.mean():.1f}")
print(f"Median tokens: {token_lengths.median():.1f}")
print(f"Over 512:      {(token_lengths > 512).sum()} chunks ({100*(token_lengths > 512).mean():.1f}%)")