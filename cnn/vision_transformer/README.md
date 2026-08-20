# vision_transformer

A **Vision Transformer (ViT)** from scratch — the bridge between this repo's CNN
and LLM halves. It applies the exact same transformer machinery used for text to
image patches.

## Files
- `vit.py` — `PatchEmbedding`, `EncoderBlock`, and `VisionTransformer`.

## Run
```bash
python vit.py
```

## How it works
1. **Patchify** the image into non-overlapping patches (via a strided conv) and
   linearly embed each patch — patches become "tokens".
2. Prepend a learnable **[CLS] token** and add **positional embeddings**.
3. Run a stack of **pre-norm transformer encoder blocks** (multi-head
   self-attention + MLP).
4. **Classify** from the final [CLS] token representation.

ViTs typically need more data or augmentation than CNNs at small scale, so on
CIFAR-10 compare it against `../resnet` and `../image_classifier` with strong
augmentation from `../augmentation`.
