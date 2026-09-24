"""Export the verified PyTorch checkpoint to ONNX (pets_unet.onnx).

Runs on Kaggle (CPU is fine) with the training notebook's output attached as an input.
Requires: pip install onnx onnxruntime. Put model.py in the working directory, then:
python export_onnx.py

Inputs:  image       float32 [batch, 3, 224, 224], ImageNet-normalised
Outputs: seg_logits  float32 [batch, 1, 224, 224]  (sigmoid > 0.5 = pet)
         cls_logits  float32 [batch, 37]           (order = model.BREEDS)
Run verify_onnx.py afterwards to check parity on the full test set.
"""
import glob, os
import numpy as np
import torch
import onnxruntime as ort
import model as M

ckpt = glob.glob("/kaggle/input/**/bb_efficientnet_b0_bestloss.pth", recursive=True)
assert ckpt, "checkpoint not found - add the '428 training' notebook output as an input"
net = M.load_model(ckpt[0])

torch.onnx.export(
    net, torch.randn(1, 3, M.IMG_SIZE, M.IMG_SIZE), "pets_unet.onnx",
    input_names=["image"], output_names=["seg_logits", "cls_logits"],
    dynamic_axes={"image": {0: "batch"}, "seg_logits": {0: "batch"}, "cls_logits": {0: "batch"}},
    opset_version=17, dynamo=False,
)

# Quick smoke test on random inputs before the full test-set check
sess = ort.InferenceSession("pets_unet.onnx", providers=["CPUExecutionProvider"])
x = torch.randn(4, 3, M.IMG_SIZE, M.IMG_SIZE)
with torch.no_grad():
    seg_t, cls_t = net(x)
seg_o, cls_o = sess.run(None, {"image": x.numpy()})
print(f"ONNX file size: {os.path.getsize('pets_unet.onnx') / 1e6:.1f} MB")
print(f"Random-input max |diff|  seg logits {np.abs(seg_o - seg_t.numpy()).max():.2e} | "
      f"cls logits {np.abs(cls_o - cls_t.numpy()).max():.2e}")
