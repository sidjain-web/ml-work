# datasets

Training data for the projects that need a corpus.

## What's here

| File | What it is | Needs network |
| --- | --- | --- |
| `tiny_corpus.txt` | Bundled ~33 KB corpus of classic fables (retold, public domain). Runs offline out of the box. | No |
| `build_corpus.py` | Regenerates `tiny_corpus.txt`. | No |
| `download_data.py` | Fetches tiny-shakespeare (~1.1 MB) → `input.txt`. Better for real training. | Yes |

## Usage

Train `nano_gpt` on the bundled corpus (works immediately):
```bash
python llm/nano_gpt/train.py --data datasets/tiny_corpus.txt --iters 3000
```

Or get the larger dataset first, then train on it:
```bash
python datasets/download_data.py
python llm/nano_gpt/train.py --data datasets/input.txt --iters 5000
```

## Which projects need data?

- **`llm/nano_gpt`** — needs a text file (use the files here). If none is given,
  it falls back to a tiny built-in string so it still runs.
- **`llm/text_classifier`** — generates a synthetic dataset in code; no file needed.
- **`cnn/image_classifier`** — auto-downloads CIFAR-10 via torchvision at runtime
  (into `./data`, which is git-ignored); no bundled file needed.

The bundled corpus repeats a set of fables so a small model can overfit and show
visible learning quickly. For meaningful text generation, train on the larger
downloaded corpus instead.
