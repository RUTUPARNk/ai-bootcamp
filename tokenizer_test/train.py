from pathlib import Path
from minbpe import BPETokenizer

def load_names(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    # Path to Names.txt (assume it's at workspace root)
    names_path = Path(__file__).parent.parent.parent / 'Names.txt'
    names = load_names(names_path)
    # Train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train('\n'.join(names), vocab_size=100)
    # Tokenize all names
    all_tokens = [tokenizer.encode(name) for name in names]
    print(f"First 3 tokenized names:")
    for name, tokens in zip(names[:3], all_tokens[:3]):
        print(f"{name} -> {tokens}")
    # Example: create input-target pairs for bigram model
    X, Y = [], []
    for tokens in all_tokens:
        for i in range(len(tokens) - 1):
            X.append(tokens[i])
            Y.append(tokens[i+1])
    print(f"First 10 input-target pairs: {list(zip(X, Y))[:10]}")
    # --- Training loop stub ---
    # for epoch in range(num_epochs):
    #     ... # implement your model training here
    # ---
    print("Stub: Add your model and training loop here.")

if __name__ == "__main__":
    main() 