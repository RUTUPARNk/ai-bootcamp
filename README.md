# AI Bootcamp

Welcome to the **AI Bootcamp** repository, a collection of hands-on experiments and projects for learning and building AI/ML solutions. This repo is designed for enthusiasts, students, and developers diving into artificial intelligence, with a focus on practical implementations like tokenization, language models, and data processing. Whether you're exploring bigram models, neural networks, or advanced tokenizers, this repo is your playground for hacking AI.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Purpose
This repository is a learning hub for AI/ML concepts, inspired by tutorials like Andrej Karpathy’s. It includes code, datasets, and experiments for:
- Tokenization (e.g., BPE, WordPiece) for language models.
- Neural network implementations (e.g., inspired by micrograd).
- Data preprocessing and visualization for ML pipelines.

The goal is to provide clear, executable examples that you can tinker with, extend, or integrate into larger projects, all while keeping the code accessible and educational.

## Getting Started

### Prerequisites
- Python 3.8+
- Dependencies: `pip install -r requirements.txt`
- Optional: Git, Docker, or a local LLM setup (e.g., Ollama) for advanced experiments

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/RUTUPARNk/ai-bootcamp.git
   cd ai-bootcamp
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Explore the directories (e.g., `cd tokenizer_test` for tokenization experiments).

### Quick Start
Try a simple tokenizer test from the `tokenizer_test` directory:
```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer_test/vocab.json")
encoded = tokenizer.encode("Hello, AI Bootcamp!")
print(encoded.tokens)  # ['Hello', ',', 'AI', 'Boot', '##camp', '!']
```

## Repository Structure
- **`tokenizer_test/`**: Experiments with tokenization (BPE, WordPiece) for LLMs, including vocabulary generation and sampling.
- **`datasets/`**: Sample datasets (e.g., Indian names, text corpora) for training models.
- **`notebooks/`**: Jupyter notebooks for interactive ML experiments.
- **`scripts/`**: Utility scripts for data preprocessing and model training.

## Usage Patterns
### Pattern 1: Tokenizing Text for LLMs
Run the tokenizer test to preprocess text:
```bash
cd tokenizer_test
python test_tokenizer.py
```
See `tokenizer_test/README.md` for advanced usage, including sampling new text.

### Pattern 2: Building a Neural Network
Explore `notebooks/micrograd_example.ipynb` for a minimal neural network implementation inspired by micrograd.

### Anti-Patterns to Avoid
- Don’t skip dependency installation; some scripts require `huggingface/tokenizers` or `numpy`.
- Avoid using outdated model weights or vocab files; check version compatibility.

## Contributing
We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. To contribute:
1. Fork the repo.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Submit a pull request to `main`.

## Troubleshooting
- **Issue**: Tokenizer fails to load `vocab.json`.
  - **Solution**: Verify the file path and ensure `huggingface/tokenizers` is installed.
- **Issue**: Outdated dependencies.
  - **Solution**: Run `pip install -r requirements.txt --upgrade`.

## Additional Resources
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) for advanced tokenization.
- [Karpathy’s Neural Networks Tutorial](https://karpathy.ai) for inspiration.
- [Issues](https://github.com/RUTUPARNk/ai-bootcamp/issues) for reporting bugs or suggestions.

## License
MIT © RUTUPARNk
<!---LeetCode Topics Start-->
# LeetCode Topics
## Array
|  |
| ------- |
| [0380-insert-delete-getrandom-o1](https://github.com/RUTUPARNk/ai-bootcamp/tree/master/0380-insert-delete-getrandom-o1) |
## Hash Table
|  |
| ------- |
| [0380-insert-delete-getrandom-o1](https://github.com/RUTUPARNk/ai-bootcamp/tree/master/0380-insert-delete-getrandom-o1) |
## Math
|  |
| ------- |
| [0380-insert-delete-getrandom-o1](https://github.com/RUTUPARNk/ai-bootcamp/tree/master/0380-insert-delete-getrandom-o1) |
## Design
|  |
| ------- |
| [0380-insert-delete-getrandom-o1](https://github.com/RUTUPARNk/ai-bootcamp/tree/master/0380-insert-delete-getrandom-o1) |
## Randomized
|  |
| ------- |
| [0380-insert-delete-getrandom-o1](https://github.com/RUTUPARNk/ai-bootcamp/tree/master/0380-insert-delete-getrandom-o1) |
<!---LeetCode Topics End-->