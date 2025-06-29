# Tokenizer Test: Self-contained BPE Tokenizer

This directory demonstrates a minimal, self-contained Byte Pair Encoding (BPE) tokenizer (see `minbpe.py`) and how to use it for text tokenization and as a building block for simple language models (e.g., bigram models).

## What does this script do?
- Implements a minimal BPE tokenizer in a single file (`minbpe.py`).
- Trains the tokenizer on a dataset of names (from `Names.txt`).
- Tokenizes example input text and decodes it back.
- Includes simple test cases to verify correctness.

## How to run
1. **Ensure `Names.txt` is present:**
   - Place your `Names.txt` (one name per line) at the root of your workspace (same level as `ai-bootcamp/`).
2. **Run the script:**
   ```bash
   python tokenize.py
   ```
   or for training/bigram modeling:
   ```bash
   python train.py
   ```

## Example input → output
```
Input: Hello world
Tokens: [some, token, ids]  # (token ids will vary)
Decoded: Hello world
```

## Test cases
The script includes these checks:
```python
assert tokenizer.decode(tokenizer.encode("Hello world")) == "Hello world"
assert isinstance(tokenizer.encode("test"), list)
```
If these pass, you'll see `Test cases passed.` in the output.

## Next Steps
- Use this tokenizer to create input pairs for a bigram or next-token model.
- Add a `train.py` to connect tokenization → training → generation.
- See the comments in `train.py` for a stub of a training loop.

---
**For the repo root README:**
- State the goal: "Minimal BPE tokenizer and simple language modeling experiments."
- Status: Working, easy to run, and extensible.
- How to run: See above.
- Next step: Plug tokenizer into a training loop, experiment with sampling/generation, and iterate! 