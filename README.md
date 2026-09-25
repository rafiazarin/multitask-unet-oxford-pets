# Multi-Task U-Net: Pet Segmentation + Breed Classification

One network, two jobs. Given a photo of a cat or dog, the model **segments the pet from the background** and **predicts its breed (37 classes)**, using a single shared encoder. Trained and evaluated on the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

**[Live demo](https://huggingface.co/spaces/rafiazarin/multitask-unet-oxford-pets)** (runs in your browser, no upload) · **[Model on Hugging Face](https://huggingface.co/rafiazarin/multitask-unet-oxford-pets)** · [Training notebook](01_training.ipynb) · [Analysis notebook](02_analysis.ipynb)

---

## Final model

**EfficientNet-B0 U-Net (ImageNet-pretrained encoder).** The run was chosen by *validation* breed accuracy, never by test; the weights are that run's lowest-validation-loss epoch.

| Metric (test set, 1,109 images) | Score |
|---|---|
| Pet IoU | **0.924** |
| Pet Dice | 0.960 |
| Breed accuracy (top-1) | **92.4%** |
| Breed accuracy (top-5) | 99.0% |
| Species accuracy (cat vs dog, derived from the predicted breed) | 99.6% |
| Macro F1 (breed) | 0.923 |

Course targets were ~0.70–0.80 Pet IoU and ~70% breed accuracy.

**Reproducibility.** [`verify_checkpoint.py`](verify_checkpoint.py) reloads the checkpoint with the standalone [`model.py`](model.py), rebuilds the seed-42 split, and reproduces every number above. The one exception is top-5, which comes out at 99.1% (one image differs) because it runs in fp32 on CPU instead of mixed precision on GPU. The analysis notebook reproduces the training-run metrics to within 0.00004.

## Deployment

The demo is a static page that runs the model **client-side with ONNX Runtime Web**, so photos never leave the visitor's device. Source: [`demo/index.html`](demo/index.html).

- **ONNX export verified on the full test set.** [`verify_onnx.py`](verify_onnx.py) runs PyTorch and ONNX Runtime side by side on all 1,109 test images: identical metrics to 4 decimals, the same top-1 prediction on all 1,109 images, maximum logit difference 2.94e-4. Export: [`export_onnx.py`](export_onnx.py).
- **In-browser speed.** Median ~120 ms per image on a MacBook Air M5 (Safari, ONNX Runtime Web 1.30.0, WebAssembly, 1 thread); two 20-run tests gave 119 ms and 120 ms. This times the model only. It excludes resizing the photo, drawing the outputs, and the one-time ~31 MB model download. The demo page has a button to rerun this test on your own device.

---

## Controlled ablation: what actually made the difference?

Five runs, 30 epochs each. Runs 1→2, 2→3 and 4→5 each change exactly one thing. Run 3→4 swaps the whole network (custom Attention U-Net → ResNet34-encoder U-Net, which also removes the attention gates).

| # | Run | What changed | Pet IoU | Breed Acc |
|---|---|---|---|---|
| 1 | Base U-Net | baseline | 0.855 | 40.4% |
| 2 | Attention U-Net | + attention gates on skip connections | 0.874 | 16.3% |
| 3 | Attention U-Net + augmentation | + training augmentation | 0.861 | 6.4% |
| 4 | ResNet34-U-Net, random init | architecture swap (keeps augmentation) | 0.870 | 37.2% |
| 5 | ResNet34-U-Net, ImageNet init | encoder weights only | **0.925** | **91.0%** |

Standard errors below are the rough guide computed in the training notebook for 1,109 test images, not a formal significance test.

- **Transfer learning was the single biggest factor.** Runs 4 and 5 share an identical architecture and training setup and differ *only* in encoder initialization. ImageNet weights raised breed accuracy from 37.2% to 91.0% (+53.8 points, about 37 standard errors) and Pet IoU from 0.870 to 0.925 (about 6 standard errors).
- **Attention gates cut breed accuracy sharply; the segmentation gain is not conclusive.** Breed accuracy fell from 40.4% to 16.3% (about 18 standard errors). Pet IoU rose by 0.018, only about 1.8 standard errors. Both runs were still improving at epoch 30 and ended at different training losses, so this compares how fast each converges within a 30-epoch budget as much as the architectures themselves.
- **Augmentation closed the train–val gap but lowered test accuracy.** The train–validation breed-accuracy gap shrank from 2.5 to 0.2 points, but test breed accuracy dropped from 16.3% to 6.4% (about 10 standard errors). These runs were underfitting, so overfitting was not the bottleneck; capacity and training length were.

## Backbone comparison

All ImageNet-pretrained, same training setup:

| Encoder | Pet IoU | Breed Acc | Params (M) |
|---|---|---|---|
| ResNet34 | 0.925 | 91.0% | 24.5 |
| MobileNetV3-Large | 0.922 | 90.6% | 6.3 |
| **EfficientNet-B0** | 0.924 | 92.4% | 7.8 |
| DenseNet121 | 0.927 | 93.1% | 13.2 |

DenseNet121 scored highest on test, but **EfficientNet-B0 was selected because it had the best validation accuracy**. Picking a model by test score would leak test information into model selection. Each encoder was trained once, so gaps of under a point may not hold across seeds.

## Error analysis

- **Hardest breeds:** American Pit Bull Terrier (46.7%) and Staffordshire Bull Terrier (69.0%), each measured on about 30 test images, so these per-breed figures are noisy. The most frequent single confusion is Pit Bull → Staffordshire (6 images); Staffordshire is most often mistaken for American Bulldog (5 images).
- **The eight most frequent confusions are all within the same species.** Cat-vs-dog accuracy, derived from the predicted breed, is 99.6%.
- **Mask threshold:** a validation sweep from 0.30 to 0.70 found 0.5 was already the best threshold, so the default was kept.

---

## Limitations

- **Single training run per configuration (one seed).** There are no run-to-run variance estimates, so small differences between models may not be reliable.
- **Closed set of 37 breeds.** For any other breed, or an image with no pet, the model still returns one of the 37. There is no out-of-domain rejection yet.
- **Same-distribution test set.** All numbers come from a held-out split of Oxford-IIIT Pet. Accuracy on other kinds of photos has not been measured.
- **Low-resolution masks.** Masks are predicted at 224×224. The demo scales them up, so edges are soft and can include a thin band of background.
- **Demo preprocessing differs slightly.** The browser resizes images differently from the PIL pipeline used for evaluation, so demo outputs can differ a little from the reported setup. ONNX parity was verified with the Python preprocessing.
- **Speed measured on one device.** The latency figure is from one laptop and browser; other devices will differ.

---

## Setup

- **Data:** 7,390 images, 37 breeds; stratified split 5,173 / 1,108 / 1,109 (train / val / test), seed 42, all 37 breeds in every split
- **Input:** 224 × 224, batch size 16, 30 epochs, mixed precision
- **Loss:** segmentation (BCE + Dice) + λ × classification (cross-entropy, label smoothing 0.1), λ = 1.0
- **Optimizer:** AdamW with cosine annealing; pretrained encoder learning rate 1e-4, 10× lower than the decoder's 1e-3
- **Augmentation (runs 3–8):** flips, rotation, random resized crop, colour jitter; training set only
- **Metrics:** Pet IoU, mIoU, Dice, top-1/top-5 accuracy, species accuracy, macro precision/recall/F1

## Repository

| File | What it does |
|---|---|
| `01_training.ipynb` | Trains all 8 runs and builds the comparison tables (Kaggle, T4 GPU) |
| `02_analysis.ipynb` | Reloads the selected checkpoint; threshold sweep, error analysis, inference |
| `model.py` | Standalone inference code for the final model |
| `verify_checkpoint.py` | Reproduces the test metrics from the saved checkpoint |
| `export_onnx.py` | Exports the checkpoint to ONNX |
| `verify_onnx.py` | Checks ONNX vs PyTorch on the full test set |
| `demo/index.html` | Browser demo (ONNX Runtime Web) |

## How to run

**Use the trained model** (weights are hosted on Hugging Face, not in this repo):

```python
import shutil
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "rafiazarin/multitask-unet-oxford-pets"
shutil.copy(hf_hub_download(REPO, "model.py"), "model.py")
import model as M

net = M.load_model(hf_hub_download(REPO, "bb_efficientnet_b0_bestloss.pth"))
mask, top3 = M.predict(net, Image.open("your_pet.jpg"))  # mask: 224x224, 1 = pet
print(top3)
```

**Retrain:** run `01_training.ipynb` on Kaggle (GPU T4, Internet on); it downloads the dataset automatically. Then run `02_analysis.ipynb` with the training notebook's output attached as an input.

**Verify:** the three scripts run on Kaggle (CPU is fine) with the training notebook's output attached. Each file's header explains its inputs.

---

Course project for **CSE428 Image Processing**, BRAC University.
