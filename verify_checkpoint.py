"""Check that model.py reproduces the test metrics reported in 01_training.ipynb.

Runs on Kaggle (CPU is fine) with Internet on and the training notebook's output
(which contains bb_efficientnet_b0_bestloss.pth) attached as an input.
Put model.py in the working directory, then:  python verify_checkpoint.py

It rebuilds the exact stratified split (seed 42), evaluates the checkpoint on the
1,109 test images in full precision, and prints reported vs reproduced metrics.
"""
import os, glob, tarfile, urllib.request, time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import model as M  # model.py from the repo root

# 1. Dataset (same source as training)
DATA = "/kaggle/working/data"
os.makedirs(DATA, exist_ok=True)
for f in ["images.tar.gz", "annotations.tar.gz"]:
    p = os.path.join(DATA, f)
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"https://www.robots.ox.ac.uk/~vgg/data/pets/data/{f}", p)
        with tarfile.open(p) as t:
            try:
                t.extractall(DATA, filter="data")
            except TypeError:  # older Python without the filter argument
                t.extractall(DATA)

# 2. Rebuild samples and split exactly as the notebooks did
samples = []
for p in sorted(glob.glob(f"{DATA}/images/*.jpg")):
    stem = os.path.basename(p)[:-4]
    if stem.startswith("."):
        continue
    tp = f"{DATA}/annotations/trimaps/{stem}.png"
    if os.path.exists(tp):
        samples.append({"image": p, "trimap": tp, "breed": stem.rsplit("_", 1)[0]})
assert sorted({s["breed"] for s in samples}) == M.BREEDS, "breed order mismatch"
b2i = {b: i for i, b in enumerate(M.BREEDS)}
y = np.array([b2i[s["breed"]] for s in samples])
idx = np.arange(len(samples))
tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, stratify=y[tmp_idx], random_state=42)
print(f"Pairs {len(samples)} | Train {len(tr_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

class PetDataset(Dataset):
    def __len__(self): return len(samples)
    def __getitem__(self, i):
        s = samples[i]
        img = Image.open(s["image"]).convert("RGB").resize((M.IMG_SIZE, M.IMG_SIZE), Image.BILINEAR)
        tri = np.array(Image.open(s["trimap"]))
        mask = Image.fromarray(np.where((tri == 1) | (tri == 3), 1, 0).astype(np.uint8))
        mask = mask.resize((M.IMG_SIZE, M.IMG_SIZE), Image.NEAREST)
        img = TF.normalize(TF.to_tensor(img), M.IMAGENET_MEAN, M.IMAGENET_STD)
        return img, torch.from_numpy(np.array(mask)).long(), b2i[s["breed"]]

# 3. Load the checkpoint with the extracted model code
ckpt = glob.glob("/kaggle/input/**/bb_efficientnet_b0_bestloss.pth", recursive=True)
assert ckpt, "checkpoint not found - add the '428 training' notebook output as an input"
model = M.load_model(ckpt[0])
print("Checkpoint:", ckpt[0])
print(f"Params: {sum(p.numel() for p in model.parameters()):,}  (expected 7,773,714)")

# 4. Evaluate on the test split (CPU, fp32)
loader = DataLoader(Subset(PetDataset(), list(test_idx)), batch_size=16, num_workers=2)
is_cat = torch.tensor(M.IS_CAT)
inter = union = pred_px = true_px = 0
top5 = sp = n = 0
cm = np.zeros((37, 37), dtype=np.int64)
t0 = time.time()
with torch.no_grad():
    for imgs, masks, labels in loader:
        seg, cls = model(imgs)
        pm = (torch.sigmoid(seg) > 0.5).long().squeeze(1)
        inter += ((pm == 1) & (masks == 1)).sum().item()
        union += ((pm == 1) | (masks == 1)).sum().item()
        pred_px += (pm == 1).sum().item(); true_px += (masks == 1).sum().item()
        top5 += (cls.topk(5, 1).indices == labels.unsqueeze(1)).any(1).sum().item()
        pred = cls.argmax(1)
        sp += (is_cat[pred] == is_cat[labels]).sum().item()
        n += imgs.size(0)
        cm += np.bincount(labels.numpy() * 37 + pred.numpy(), minlength=37 * 37).reshape(37, 37)
tp = np.diag(cm).astype(float)
prec = tp / np.maximum(cm.sum(0), 1e-9); rec = tp / np.maximum(cm.sum(1), 1e-9)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
got = {"Pet IoU": inter / union, "Pet Dice": 2 * inter / (pred_px + true_px),
       "Cls Acc": tp.sum() / cm.sum(), "Top-5 Acc": top5 / n, "Species Acc": sp / n,
       "Macro F1": f1.mean()}
reported = {"Pet IoU": 0.9239, "Pet Dice": 0.9604, "Cls Acc": 0.9243,
            "Top-5 Acc": 0.9901, "Species Acc": 0.9964, "Macro F1": 0.9227}
print(f"\nTest images: {n} | eval time {time.time() - t0:.0f}s\n")
print(f"{'metric':<12} {'reported':>9} {'reproduced':>11} {'diff':>8}")
for k in reported:
    print(f"{k:<12} {reported[k]:>9.4f} {got[k]:>11.4f} {got[k] - reported[k]:>+8.4f}")

# 5. Single-image path used by the app (analysis notebook showed Bengal 85.0%)
s = samples[int(test_idx[0])]
_, top = M.predict(model, Image.open(s["image"]))
print(f"\nSingle image - true breed: {s['breed']}")
for b, c in top:
    print(f"  {b:<28} {c * 100:5.1f}%")
