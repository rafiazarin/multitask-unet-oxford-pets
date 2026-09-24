"""Inference code for the multi-task U-Net (EfficientNet-B0 encoder).

Architecture copied from 02_analysis.ipynb so the saved checkpoint
bb_efficientnet_b0_bestloss.pth loads with strict=True.
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models as tvm
from PIL import Image

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Same order as training: sorted() puts the 12 capitalised cat breeds first (0-11),
# then the 25 lowercase dog breeds (12-36).
BREEDS = sorted([
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Egyptian_Mau",
    "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx",
    "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle", "boxer",
    "chihuahua", "english_cocker_spaniel", "english_setter", "german_shorthaired",
    "great_pyrenees", "havanese", "japanese_chin", "keeshond", "leonberger",
    "miniature_pinscher", "newfoundland", "pomeranian", "pug", "saint_bernard",
    "samoyed", "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
])
IS_CAT = [b[0].isupper() for b in BREEDS]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


def classifier_head(in_f, n_classes, p=0.3):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(in_f, 256), nn.ReLU(inplace=True),
                         nn.Dropout(p), nn.Linear(256, n_classes))


def _efficientnet_b0_stages(pretrained=False):
    f = tvm.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None).features
    return [f[0:2], f[2:3], f[3:4], f[4:6], f[6:]]


class BackboneUNet(nn.Module):
    def __init__(self, out_ch=1, n_classes=len(BREEDS), pretrained=False):
        super().__init__()
        self.stages = nn.ModuleList(_efficientnet_b0_stages(pretrained))
        with torch.no_grad():
            x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            ch, sz = [], []
            for s in self.stages:
                x = s(x)
                ch.append(x.shape[1])
                sz.append(x.shape[-1])
        assert sz == [IMG_SIZE // k for k in (2, 4, 8, 16, 32)], sz
        c0, c1, c2, c3, c4 = ch
        d3, d2, d1, d0 = 256, 128, 64, 48
        self.up4 = nn.ConvTranspose2d(c4, d3, 2, 2); self.dec4 = DoubleConv(d3 + c3, d3)
        self.up3 = nn.ConvTranspose2d(d3, d2, 2, 2); self.dec3 = DoubleConv(d2 + c2, d2)
        self.up2 = nn.ConvTranspose2d(d2, d1, 2, 2); self.dec2 = DoubleConv(d1 + c1, d1)
        self.up1 = nn.ConvTranspose2d(d1, d0, 2, 2); self.dec1 = DoubleConv(d0 + c0, d0)
        self.up0 = nn.ConvTranspose2d(d0, 32, 2, 2); self.dec0 = DoubleConv(32, 32)
        self.final = nn.Conv2d(32, out_ch, 1)
        self.classifier = classifier_head(c4, n_classes)

    def forward(self, x):
        e0 = self.stages[0](x); e1 = self.stages[1](e0); e2 = self.stages[2](e1)
        e3 = self.stages[3](e2); e4 = self.stages[4](e3)
        cls = self.classifier(e4)
        u4 = self.dec4(torch.cat([self.up4(e4), e3], 1))
        u3 = self.dec3(torch.cat([self.up3(u4), e2], 1))
        u2 = self.dec2(torch.cat([self.up2(u3), e1], 1))
        u1 = self.dec1(torch.cat([self.up1(u2), e0], 1))
        return self.final(self.dec0(self.up0(u1))), cls


def load_model(ckpt_path, device="cpu"):
    model = BackboneUNet()
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)


@torch.no_grad()
def predict(model, pil_img, thr=0.5, topk=3):
    """Returns a 224x224 uint8 mask (1 = pet) and the top-k (breed, probability) list."""
    x = preprocess(pil_img).to(next(model.parameters()).device)
    seg_logits, cls_logits = model(x)
    mask = (torch.sigmoid(seg_logits.float()) > thr).squeeze().cpu().numpy().astype("uint8")
    prob = torch.softmax(cls_logits.float(), 1).squeeze(0)
    conf, ids = prob.topk(topk)
    return mask, [(BREEDS[i], c) for c, i in zip(conf.tolist(), ids.tolist())]
