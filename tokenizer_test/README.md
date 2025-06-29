# Tokenizer Test

This directory contains experiments for tokenization, a key preprocessing step for large language models (LLMs). It uses the [Hugging Face `tokenizers` library](https://github.com/huggingface/tokenizers) to implement Byte Pair Encoding (BPE) and WordPiece tokenizers, with examples for training, encoding, and sampling text. The goal is to provide a playground for understanding and hacking tokenization for AI/ML projects.

## Purpose
Tokenization converts raw text into tokens (subword units or characters) for LLMs. This directory includes:
- Training a tokenizer on a custom dataset (e.g., Indian names).
- Encoding text into tokens for model input.
- Sampling new text sequences using trained tokenizers.
- Visualizing tokenization results.

## Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install tokenizers numpy
  ```

## Installation
1. Navigate to this directory:
   ```bash
   cd tokenizer_test
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start
Encode a sample text using a pre-trained tokenizer:
```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("vocab.json")
encoded = tokenizer.encode("Hello, AI Bootcamp!")
print(encoded.tokens)  # ['Hello', ',', 'AI', 'Boot', '##camp', '!']
```

## Advanced Usage
### Training a Custom BPE Tokenizer
Train a BPE tokenizer on a dataset (e.g., `Names.txt` with Indian names):
```python
from tokenizers import CharBPETokenizer
from tokenizers.pre_tokenizers import Whitespace

# Initialize tokenizer
tokenizer = CharBPETokenizer()

# Set pre-tokenizer to handle whitespace
tokenizer.pre_tokenizer = Whitespace()

# Train on Names.txt (ensure lowercase, no special chars)
tokenizer.train(["../datasets/Names.txt"], vocab_size=1000, min_frequency=2)

# Save the tokenizer
tokenizer.save("vocab.json")

# Encode a sample name
encoded = tokenizer.encode("Aarav")
print(encoded.tokens)  # ['Aa', 'rav']
```

### Encoding with Special Tokens
Add special tokens (e.g., `<|START|>`, `<|END|>`) for LLM compatibility:
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Initialize with special tokens
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["<|START|>", "<|END|>"])
tokenizer.train_from_iterator(["Aarav", "Priya"], trainer=trainer)

# Encode with special tokens
encoded = tokenizer.encode("<|START|>Aarav<|END|>")
print(encoded.tokens)  # ['<|START|>', 'Aa', 'rav', '<|END|>']
```

### Sampling New Text
Generate new names using the trained tokenizer’s vocabulary and a simple probability model (e.g., bigram-inspired):
```python
import random
import numpy as np
from tokenizers import Tokenizer

# Load trained tokenizer
tokenizer = Tokenizer.from_file("vocab.json")

# Get vocabulary
vocab = tokenizer.get_vocab()
tokens = list(vocab.keys())
probs = np.ones(len(tokens)) / len(tokens)  # Uniform probabilities

# Sample a sequence
def sample_name(max_length=10):
    sequence = ["<|START|>"]
    for _ in range(max_length):
        next_token = np.random.choice(tokens, p=probs)
        if next_token == "<|END|>":
            break
        sequence.append(next_token)
    return "".join(sequence[1:])  # Skip <|START|>

# Generate 5 names
for _ in range(5):
    print(sample_name())  # e.g., Aarav, Priy, Rav, etc.
```

## Usage Patterns
### Pattern 1: Preprocessing for LLMs
Tokenize a dataset for model training:
```bash
python test_tokenizer.py --input ../datasets/Names.txt --output tokens.json
```

### Pattern 2: Visualizing Tokens
Visualize tokenization results:
```python
import json
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("vocab.json")
encoded = tokenizer.encode("Priya Sharma")
with open("tokens.json", "w") as f:
    json.dump({"text": "Priya Sharma", "tokens": encoded.tokens}, f)
```

### Anti-Patterns to Avoid
- Don’t use unfiltered datasets with special characters (e.g., spaces, hyphens); preprocess with lowercase and alphabetic chars only (see your bigram model code).
- Avoid large `vocab_size` without sufficient data; start with 1000-5000 for small datasets.
- Don’t ignore special tokens for LLMs; they’re critical for context boundaries.

## Troubleshooting
- **Issue**: `vocab.json` not found.
  - **Solution**: Run the training script first or verify the file path.
- **Issue**: Tokens include special characters.
  - **Solution**: Clean input data (e.g., `Names.txt`) with:
    ```python
    with open("Names.txt", "r") as f:
        names = [name.lower().strip() for name in f if name.strip().isalpha()]
    with open("cleaned_Names.txt", "w") as f:
        f.write("\n".join(names))
    ```

## Additional Resources
- [Hugging Face Tokenizers Docs](https://huggingface.co/docs/tokenizers)
- [Karpathy’s Neural Networks Tutorial](https://karpathy.ai) for bigram inspiration
- [Issues](https://github.com/RUTUPARNk/ai-bootcamp/issues) for bugs or suggestions

## License
MIT © RUTUPARNk