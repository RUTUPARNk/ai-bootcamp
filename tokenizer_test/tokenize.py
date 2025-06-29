import sys
sys.path.append('..')  # To allow import if minbpe.py is placed one level up or in the same dir

from pathlib import Path

# Try to import minbpe from local copy (user should place minbpe.py in this dir or parent)
try:
    from minbpe import BPETokenizer
except ImportError:
    raise ImportError("Please download minbpe.py from https://github.com/karpathy/minbpe and place it in this directory or the parent directory.")

def load_names(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def train_and_tokenize(names, vocab_size=100):
    # Join all names into a single string separated by newlines
    text = '\n'.join(names)
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=vocab_size)
    return tokenizer

def main():
    # Path to Names.txt (assume it's at workspace root)
    names_path = Path(__file__).parent.parent.parent / 'Names.txt'
    names = load_names(names_path)
    tokenizer = train_and_tokenize(names, vocab_size=100)
    # Example: tokenize a name
    example = "Hello world"
    tokens = tokenizer.encode(example)
    print(f"Input: {example}\nTokens: {tokens}\nDecoded: {tokenizer.decode(tokens)}")
    # Test cases
    assert tokenizer.decode(tokenizer.encode("Hello world")) == "Hello world"
    assert isinstance(tokenizer.encode("test"), list)
    print("Test cases passed.")

if __name__ == "__main__":
    main() 