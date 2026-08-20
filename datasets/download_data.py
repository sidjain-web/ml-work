"""Download a real text corpus for nano_gpt (needs network).

Fetches the tiny-shakespeare dataset (~1.1 MB, public domain) and saves it to
`input.txt` in this folder. This is the standard corpus used to demo small
character-level language models.

    python download_data.py
    python ../llm/nano_gpt/train.py --data datasets/input.txt --iters 5000
"""
from __future__ import annotations

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")
    print(f"Downloading tiny-shakespeare from:\n  {URL}")
    try:
        urllib.request.urlretrieve(URL, out_path)
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("No network? Use the bundled datasets/tiny_corpus.txt instead, or")
        print("regenerate it with: python build_corpus.py")
        return

    with open(out_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Saved -> {out_path}")
    print(f"characters: {len(text):,}")
    print(f"unique chars: {len(set(text))}")


if __name__ == "__main__":
    main()
