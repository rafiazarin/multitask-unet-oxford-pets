# Multi-Task U-Net: Pet Segmentation + Breed Classification

One network, two jobs. Given a photo of a cat or dog, the model **segments the pet from the background** and **predicts its breed (37 classes)**, using a single shared encoder. Trained and evaluated on the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

📓 [Training notebook](01_training.ipynb) · 📊 [Analysis notebook](02_analysis.ipynb)

---

## Final model

**EfficientNet-B0 U-Net (ImageNet-pretrained)**, selected on *validation* breed accuracy, never on test.

| Metric (test set, 1,109 images) | Score |
|---|---|
| Pet IoU | **0.924** |
| Pet Dice | 0.960 |
| Breed accuracy (top-1) | **92.4%** |
| Breed accuracy (top-5) | 99.0% |
| Species accuracy (cat vs dog) | 99.6% |
| Macro F1 (breed) | 0.923 |

Project targets were ~0.70–0.80 IoU and ~70% breed accuracy. Both were exceeded.

---

## Controlled ablation: what actually made the difference?

Five runs, **each changing exactly one thing** from the run before it, so every change in the table can be traced to a single cause.

| # | Run | What changed | Pet IoU | Breed Acc |
|---|---|---|---|---|
| 1 | Base U-Net | baseline | 0.855 | 40.4% |
| 2 | Attention U-Net | + attention gates on skip connections | 0.874 | 16.3% |
| 3 | Attention U-Net + augmentation | + training augmentation | 0.861 | 6.4% |
| 4 | ResNet34-U-Net, random init | encoder architecture | 0.870 | 37.2% |
| 5 | ResNet34-U-Net, ImageNet init | encoder weights only | **0.925** | **91.0%** |

**Key findings:**

- **Transfer learning was the single biggest factor.** Runs 4 and 5 share an identical architecture and training setup and differ *only* in encoder initialization. ImageNet weights raised breed accuracy from 37.2% to 91.0% (+53.8 points, ~37 standard errors).
- **Attention gates helped segmentation but hurt classification.** Pet IoU rose slightly (+0.018), but breed accuracy fell sharply. Those runs were underfitting within the 30-epoch budget.
- **Augmentation closed the train–val gap but didn't raise test accuracy.** Overfitting wasn't the bottleneck for these models; capacity and training length were.

## Backbone comparison

All ImageNet-pretrained, same training setup:

| Encoder | Pet IoU | Breed Acc | Params (M) |
|---|---|---|---|
| ResNet34 | 0.925 | 91.0% | 24.5 |
| MobileNetV3-Large | 0.922 | 90.6% | 6.3 |
| **EfficientNet-B0** | 0.924 | 92.4% | 7.8 |
| DenseNet121 | 0.927 | 93.1% | 13.2 |

DenseNet121 scored highest on test, but **EfficientNet-B0 was selected because it had the best validation accuracy**. Picking a model based on test scores would leak test information into model selection.

## Error analysis

- **Hardest breeds:** American Pit Bull Terrier (47%) and Staffordshire Bull Terrier (69%). They are most often confused with each other, which is a genuinely hard visual distinction.
- **All top confusions are within the same species.** The model almost never mistakes a cat for a dog.
- **Threshold calibration:** a validation sweep found the default 0.5 mask threshold was already optimal, meaning the segmentation probabilities are well calibrated.
- **Reproducibility check:** the analysis notebook reloads the saved checkpoint and reproduces every reported test metric to within 0.00004.

---

## Setup

- **Data:** 7,390 images, 37 breeds; stratified split 5,173 / 1,108 / 1,109 (train / val / test), all 37 breeds in every split
- **Input:** 224 × 224, batch size 16, 30 epochs, mixed precision
- **Loss:** segmentation loss + classification loss (λ = 1.0), label smoothing 0.1
- **Optimizer:** AdamW; pretrained encoder learning rate 10× lower than the decoder
- **Metrics:** Pet IoU, mIoU, Dice, top-1/top-5 accuracy, species accuracy, macro precision/recall/F1

## How to run

Built for **Kaggle** (GPU T4, Internet on). The training notebook downloads the dataset automatically.

1. Run `01_training.ipynb` to train all runs and produce the comparison tables.
2. Run `02_analysis.ipynb` to reload the best checkpoint, then run calibration, error analysis, and inference.

Locally: `pip install -r requirements.txt`, then update the data and checkpoint paths (they default to `/kaggle/working`).

Model weights are not included in this repo because of file size limits.

---

Course project for **CSE428 Image Processing**, BRAC University.
