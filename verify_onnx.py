"""Check that pets_unet.onnx matches the PyTorch checkpoint on the full test set.

Runs on Kaggle (CPU is fine) with Internet on and the training notebook's output attached.
Requires: pip install onnxruntime. Run export_onnx.py first, then: python verify_onnx.py

Evaluates PyTorch and ONNX Runtime side by side on the 1,109 test images (seed-42 split),
prints both sets of metrics, the largest logit difference, and how many images get a
different top-1 breed. Writes the result to parity.json.
"""
import os, glob, json, tarfile, urllib.request, time
import numpy as np
import torch
import torchvision.transforms.functional as TF
import onnxruntime as ort
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import model as M

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
            except TypeError:
                t.extractall(DATA)

# 2. Same samples and split as the training notebook
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


# 3. Metric accumulator (same definitions as the training notebook)
is_cat = torch.tensor(M.IS_CAT)
def new_acc():
    return {"inter": 0, "union": 0, "pred_px": 0, "true_px": 0, "top5": 0, "sp": 0, "n": 0,
            "cm": np.zeros((37, 37), dtype=np.int64), "preds": []}

def update(a, seg, cls, masks, labels):
    pm = (torch.sigmoid(seg) > 0.5).long().squeeze(1)
    a["inter"] += ((pm == 1) & (masks == 1)).sum().item()
    a["union"] += ((pm == 1) | (masks == 1)).sum().item()
    a["pred_px"] += (pm == 1).sum().item(); a["true_px"] += (masks == 1).sum().item()
    a["top5"] += (cls.topk(5, 1).indices == labels.unsqueeze(1)).any(1).sum().item()
    pred = cls.argmax(1)
    a["preds"] += pred.tolist()
    a["sp"] += (is_cat[pred] == is_cat[labels]).sum().item()
    a["n"] += len(labels)
    a["cm"] += np.bincount(labels.numpy() * 37 + pred.numpy(), minlength=37 * 37).reshape(37, 37)

def finalize(a):
    cm = a["cm"]; tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1e-9); rec = tp / np.maximum(cm.sum(1), 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return {"Pet IoU": a["inter"] / a["union"], "Pet Dice": 2 * a["inter"] / (a["pred_px"] + a["true_px"]),
            "Cls Acc": tp.sum() / cm.sum(), "Top-5 Acc": a["top5"] / a["n"],
            "Species Acc": a["sp"] / a["n"], "Macro F1": f1.mean()}


# 4. Run PyTorch and ONNX Runtime side by side on the test split
ckpt = glob.glob("/kaggle/input/**/bb_efficientnet_b0_bestloss.pth", recursive=True)[0]
net = M.load_model(ckpt)
sess = ort.InferenceSession("pets_unet.onnx", providers=["CPUExecutionProvider"])
loader = DataLoader(Subset(PetDataset(), list(test_idx)), batch_size=16, num_workers=2)
acc_pt, acc_ox = new_acc(), new_acc()
max_seg, max_cls = 0.0, 0.0
t0 = time.time()
with torch.no_grad():
    for imgs, masks, labels in loader:
        seg_t, cls_t = net(imgs)
        seg_o, cls_o = (torch.from_numpy(o) for o in sess.run(None, {"image": imgs.numpy()}))
        max_seg = max(max_seg, (seg_o - seg_t).abs().max().item())
        max_cls = max(max_cls, (cls_o - cls_t).abs().max().item())
        update(acc_pt, seg_t, cls_t, masks, labels)
        update(acc_ox, seg_o, cls_o, masks, labels)

pt, ox = finalize(acc_pt), finalize(acc_ox)
reported = {"Pet IoU": 0.9239, "Pet Dice": 0.9604, "Cls Acc": 0.9243,
            "Top-5 Acc": 0.9901, "Species Acc": 0.9964, "Macro F1": 0.9227}
top1_disagree = sum(a != b for a, b in zip(acc_pt["preds"], acc_ox["preds"]))

print(f"\nTest images: {acc_pt['n']} | time {time.time() - t0:.0f}s\n")
print(f"{'metric':<12} {'reported':>9} {'pytorch':>9} {'onnx':>9} {'onnx-pt':>9}")
for k in reported:
    print(f"{k:<12} {reported[k]:>9.4f} {pt[k]:>9.4f} {ox[k]:>9.4f} {ox[k] - pt[k]:>+9.4f}")
print(f"\nMax |diff| over test set  seg logits {max_seg:.2e} | cls logits {max_cls:.2e}")
print(f"Images where top-1 breed differs (ONNX vs PyTorch): {top1_disagree} of {acc_pt['n']}")

passed = all(abs(ox[k] - pt[k]) <= 0.001 for k in reported)
print("\nPARITY", "PASSED" if passed else "FAILED", "(every metric within 0.001 of PyTorch)")
json.dump({"passed": passed, "pytorch": pt, "onnx": ox, "max_seg_diff": max_seg,
           "max_cls_diff": max_cls, "top1_disagree": top1_disagree}, open("parity.json", "w"), indent=1)
