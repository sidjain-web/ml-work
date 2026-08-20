# image_classifier

A small, modern **CNN** for CIFAR-10-sized images: stacked Conv-BN-ReLU blocks,
max-pooling, global average pooling, and a linear head. Batch-norm + GAP keep it
compact (~1M params) and stable to train.

## Files
- `model.py` — `ConvBlock` and `SimpleCNN`.
- `train.py` — CIFAR-10 training with augmentation, AdamW, and cosine LR decay.

## Run
```bash
python model.py                    # forward-pass shape check
python train.py --epochs 15        # downloads CIFAR-10 via torchvision
```
With the default recipe this reaches roughly ~85% test accuracy on CIFAR-10.
If torchvision/CIFAR isn't available, `train.py` falls back to random tensors so
the loop still runs for a smoke test.

## Try next
- Compare against `../resnet` (residual connections) on the same data.
- Add mixup/cutout from `../augmentation`.
