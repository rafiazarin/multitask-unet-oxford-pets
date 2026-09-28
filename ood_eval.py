"""Out-of-domain (OOD) evaluation for the multi-task U-Net.

In-distribution (ID): the Oxford-IIIT Pet validation and test splits (seed 42, as in training).
OOD sets (evaluation only, never trained on):
  - coco_nonpet:  1,000 COCO val2017 images whose annotations contain no cat or dog
  - other_dogs:   1,000 Stanford Dogs images from breeds NOT among the 37 Oxford breeds
                  (overlapping breeds, plus American Staffordshire Terrier, are excluded by name)
Each OOD set is split randomly 50/50 into ood-val (used only to pick the confidence score) and
ood-test (used only for reporting). Thresholds are set on ID validation only.

Scores (higher = more in-distribution): msp (max softmax probability), max_logit,
energy (logsumexp of logits), and area (fraction of pixels predicted as pet).
Pre-specified rule: reject if score < t_score OR area < t_area, with t_score keeping 95% and
t_area keeping 99% of ID validation images. Writes ood_results.json.
Runs on Kaggle (CPU) with Internet on and the training notebook's output attached.
"""
import os, glob, json, tarfile, zipfile, urllib.request, time
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import model as M

DATA = "/kaggle/working/data"
os.makedirs(DATA, exist_ok=True)
rng = np.random.default_rng(0)

def download(url, path):
    if not os.path.exists(path):
        print("downloading", url)
        urllib.request.urlretrieve(url, path)

# ---------------- ID: Oxford val/test (same split as training) ----------------
for f in ["images.tar.gz", "annotations.tar.gz"]:
    p = os.path.join(DATA, f)
    if not os.path.exists(p):
        download(f"https://www.robots.ox.ac.uk/~vgg/data/pets/data/{f}", p)
        with tarfile.open(p) as t:
            try:
                t.extractall(DATA, filter="data")
            except TypeError:
                t.extractall(DATA)
samples = []
for p in sorted(glob.glob(f"{DATA}/images/*.jpg")):
    stem = os.path.basename(p)[:-4]
    if stem.startswith("."):
        continue
    if os.path.exists(f"{DATA}/annotations/trimaps/{stem}.png"):
        samples.append({"image": p, "breed": stem.rsplit("_", 1)[0]})
assert sorted({s["breed"] for s in samples}) == M.BREEDS, "breed order mismatch"
b2i = {b: i for i, b in enumerate(M.BREEDS)}
y = np.array([b2i[s["breed"]] for s in samples])
idx = np.arange(len(samples))
tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, stratify=y[tmp_idx], random_state=42)
print(f"ID  val {len(val_idx)} | test {len(test_idx)}")

# ---------------- OOD 1: COCO val2017 images with no cat/dog ----------------
download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", f"{DATA}/coco_ann.zip")
with zipfile.ZipFile(f"{DATA}/coco_ann.zip") as z:
    ann = json.load(z.open("annotations/instances_val2017.json"))
pet_cats = {c["id"] for c in ann["categories"] if c["name"] in ("cat", "dog")}
assert len(pet_cats) == 2, pet_cats
pet_imgs = {a["image_id"] for a in ann["annotations"] if a["category_id"] in pet_cats}
nonpet = sorted((im["id"], im["file_name"]) for im in ann["images"] if im["id"] not in pet_imgs)
pick = rng.choice(len(nonpet), 1000, replace=False)
coco_files = [nonpet[i][1] for i in sorted(pick)]
download("http://images.cocodataset.org/zips/val2017.zip", f"{DATA}/val2017.zip")
with zipfile.ZipFile(f"{DATA}/val2017.zip") as z:
    for fn in coco_files:
        if not os.path.exists(f"{DATA}/val2017/{fn}"):
            z.extract(f"val2017/{fn}", DATA)
coco_paths = [f"{DATA}/val2017/{fn}" for fn in coco_files]
print(f"OOD coco_nonpet: {len(coco_paths)} images (from {len(nonpet)} without cat/dog)")

# ---------------- OOD 2: Stanford Dogs, breeds not in the 37 ----------------
EXCLUDE = ["Chihuahua", "Japanese_spaniel", "Pomeranian", "basset", "beagle", "English_setter",
           "German_short-haired_pointer", "cocker_spaniel", "Staffordshire_bullterrier",
           "American_Staffordshire_terrier", "Yorkshire_terrier", "soft-coated_wheaten_terrier",
           "Scotch_terrier", "miniature_pinscher", "keeshond", "boxer", "Great_Pyrenees",
           "Saint_Bernard", "Newfoundland", "Samoyed", "Leonberg", "pug"]
download("http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar", f"{DATA}/stanford_dogs.tar")
with tarfile.open(f"{DATA}/stanford_dogs.tar") as t:
    members = [m for m in t.getmembers() if m.isfile() and m.name.lower().endswith(".jpg")]
    breed_of = lambda m: m.name.split("/")[-2].split("-", 1)[1]
    breeds = sorted({breed_of(m) for m in members})
    missing = [b for b in EXCLUDE if b not in breeds]
    assert not missing, f"exclusion names not found in Stanford Dogs: {missing}"
    keep = sorted([m for m in members if breed_of(m) not in EXCLUDE], key=lambda m: m.name)
    pick = rng.choice(len(keep), 1000, replace=False)
    chosen = [keep[i] for i in sorted(pick)]
    for m in chosen:
        if not os.path.exists(os.path.join(DATA, m.name)):
            t.extract(m, DATA)
dog_paths = [os.path.join(DATA, m.name) for m in chosen]
print(f"OOD other_dogs: {len(dog_paths)} images from {len(breeds) - len(EXCLUDE)} non-overlapping breeds "
      f"({len(EXCLUDE)} excluded of {len(breeds)})")

# ---------------- scoring ----------------
ckpt = glob.glob("/kaggle/input/**/bb_efficientnet_b0_bestloss.pth", recursive=True)[0]
net = M.load_model(ckpt)

@torch.no_grad()
def score(paths, bs=32):
    out = {"msp": [], "max_logit": [], "energy": [], "area": [], "pred": []}
    for i in range(0, len(paths), bs):
        x = torch.cat([M.preprocess(Image.open(p)) for p in paths[i:i + bs]])
        seg, cls = net(x)
        out["msp"] += torch.softmax(cls, 1).max(1).values.tolist()
        out["max_logit"] += cls.max(1).values.tolist()
        out["energy"] += torch.logsumexp(cls, 1).tolist()
        out["area"] += (torch.sigmoid(seg) > 0.5).float().mean((1, 2, 3)).tolist()
        out["pred"] += cls.argmax(1).tolist()
    return {k: np.array(v) for k, v in out.items()}

t0 = time.time()
S = {
    "id_val": score([samples[i]["image"] for i in val_idx]),
    "id_test": score([samples[i]["image"] for i in test_idx]),
}
yv, yt = y[val_idx], y[test_idx]
for name, paths in [("coco_nonpet", coco_paths), ("other_dogs", dog_paths)]:
    s = score(paths)
    perm = rng.permutation(len(paths))            # random 50/50 split into ood-val / ood-test
    va, te = perm[: len(paths) // 2], perm[len(paths) // 2:]
    S[name + "_val"] = {k: v[va] for k, v in s.items()}
    S[name + "_test"] = {k: v[te] for k, v in s.items()}
print(f"scored everything in {time.time() - t0:.0f}s")
assert abs((S["id_test"]["pred"] == yt).mean() - 0.9243) < 0.002, "ID test accuracy does not match 0.9243"

def auroc(id_s, ood_s):
    return roc_auc_score(np.r_[np.ones(len(id_s)), np.zeros(len(ood_s))], np.r_[id_s, ood_s])

def fpr_at_95(id_s, ood_s):
    t = np.percentile(id_s, 5)
    return float((ood_s >= t).mean())

SCORES = ["msp", "max_logit", "energy", "area"]
OOD = ["coco_nonpet", "other_dogs"]
res = {"auroc": {}, "fpr95": {}}
for split, id_key in [("val", "id_val"), ("test", "id_test")]:
    print(f"\nAUROC / FPR@95%TPR  (ID {split} vs OOD {split} halves)")
    print(f"  {'score':<10}" + "".join(f"{o:>26}" for o in OOD))
    for sc in SCORES:
        row = ""
        for o in OOD:
            a = auroc(S[id_key][sc], S[f"{o}_{split}"][sc])
            f = fpr_at_95(S[id_key][sc], S[f"{o}_{split}"][sc])
            res["auroc"][f"{split}/{o}/{sc}"] = a
            res["fpr95"][f"{split}/{o}/{sc}"] = f
            row += f"{f'{a:.3f} / {f:.3f}':>26}"
        print(f"  {sc:<10}" + row)

# pick the confidence score on OOD-val only (mean AUROC over both sets); area is used separately
cands = ["msp", "max_logit", "energy"]
best = max(cands, key=lambda sc: np.mean([res["auroc"][f"val/{o}/{sc}"] for o in OOD]))
t_score = float(np.percentile(S["id_val"][best], 5))
t_area = float(np.percentile(S["id_val"]["area"], 1))
print(f"\nChosen confidence score (on OOD-val): {best}")
print(f"Thresholds (from ID val only): {best} >= {t_score:.4f}, area >= {t_area:.4f}")

def accept(s):
    return (s[best] >= t_score) & (s["area"] >= t_area)

acc_v = accept(S["id_val"]); acc_t = accept(S["id_test"])
print("\nOperating point on held-out data (pre-specified rule)")
print(f"  ID val  kept: {acc_v.mean():.3f}")
print(f"  ID test kept: {acc_t.mean():.3f}  ({acc_t.sum()} of {len(acc_t)})")
print(f"  ID test breed accuracy: all {(S['id_test']['pred'] == yt).mean():.4f} | "
      f"kept {(S['id_test']['pred'][acc_t] == yt[acc_t]).mean():.4f} | "
      f"rejected {(S['id_test']['pred'][~acc_t] == yt[~acc_t]).mean():.4f}")
op = {"id_val_kept": float(acc_v.mean()), "id_test_kept": float(acc_t.mean()),
      "id_test_acc_all": float((S["id_test"]["pred"] == yt).mean()),
      "id_test_acc_kept": float((S["id_test"]["pred"][acc_t] == yt[acc_t]).mean())}
for o in OOD:
    r = 1 - accept(S[f"{o}_test"]).mean()
    op[f"{o}_test_rejected"] = float(r)
    print(f"  {o:<12} test rejected: {r:.3f}  (n={len(S[f'{o}_test']['msp'])})")

json.dump({"chosen_score": best, "t_score": t_score, "t_area": t_area, "operating_point": op,
           "auroc": res["auroc"], "fpr95": res["fpr95"],
           "n": {k: int(len(v["msp"])) for k, v in S.items()}},
          open("ood_results.json", "w"), indent=1)
print("\nSaved ood_results.json")
