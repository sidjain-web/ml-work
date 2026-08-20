# ml-projects

A monorepo of small, from-scratch **LLM** and **CNN** projects in Python. Each
project is self-contained, readable, and runnable on its own — built for
learning and as reusable building blocks rather than as a heavyweight framework.

## Layout

```
ml-projects/
├── common/                    # shared helpers (seeding, device, param counts)
├── llm/
│   ├── nano_gpt/              # decoder-only GPT: model, train, sample
│   ├── bpe_tokenizer/         # byte-level BPE tokenizer (pure Python)
│   ├── attention/             # attention from scratch (PyTorch + NumPy ref)
│   ├── text_classifier/       # transformer-encoder classifier
│   └── rag_mini/              # retrieval-augmented generation (pure NumPy)
└── cnn/
    ├── image_classifier/      # compact CNN for CIFAR-10
    ├── resnet/                # ResNet-18/34/50 from scratch
    ├── vision_transformer/    # ViT — the bridge between CNNs and transformers
    └── augmentation/          # image augmentations (pure NumPy)
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run any project directly — for example:

```bash
python llm/nano_gpt/train.py --iters 2000
python cnn/resnet/resnet.py
python llm/bpe_tokenizer/tokenizer.py
```

## Runs-anywhere projects (no torch needed)

These use only NumPy / the standard library and include self-tests you can run
immediately:

| Project | Command |
| --- | --- |
| `llm/bpe_tokenizer` | `python llm/bpe_tokenizer/tokenizer.py` |
| `llm/attention` (NumPy ref) | `python llm/attention/reference_numpy.py` |
| `llm/rag_mini` | `python llm/rag_mini/rag.py` |
| `cnn/augmentation` | `python cnn/augmentation/transforms.py` |

The remaining projects require PyTorch (see `requirements.txt`).

## How the pieces fit together

- The **attention** module is the core op behind `nano_gpt`, `text_classifier`,
  and the `vision_transformer`.
- The **bpe_tokenizer** produces sub-word units to feed `nano_gpt`.
- The **vision_transformer** applies the same transformer machinery as the LLM
  projects to image patches — a natural bridge between the two halves.
- The **augmentation** transforms feed the CNN training loops.

## License

MIT — see `LICENSE`.
