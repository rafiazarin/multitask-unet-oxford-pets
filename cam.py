"""Grad-CAM for the multi-task U-Net, computed inside the network so it can be exported to ONNX.

The classifier head is: global average pool -> Linear(1280, 256) -> ReLU -> Dropout -> Linear(256, 37).
For class c, the gradient of its logit with respect to the final encoder feature map A (1280 x 7 x 7) is

    d s_c / d A[k, i, j] = (1 / HW) * sum_h W2[c, h] * 1[z_h > 0] * W1[h, k]

which is the same at every spatial position (i, j). So the Grad-CAM channel weights are exactly
alpha[c, k] = (1 / HW) * sum_h W2[c, h] * 1[z_h > 0] * W1[h, k], and no backpropagation is needed.
Dropout is the identity in eval mode. verify_cam.py checks this against PyTorch autograd.
"""
import torch
import torch.nn as nn


class CamNet(nn.Module):
    """Wraps BackboneUNet. Returns (seg_logits, cls_logits, cam) where cam is [B, 37, 7, 7]."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        e0 = n.stages[0](x); e1 = n.stages[1](e0); e2 = n.stages[2](e1)
        e3 = n.stages[3](e2); e4 = n.stages[4](e3)

        head = n.classifier                  # [pool, flatten, lin1, relu, dropout, lin2]
        z = head[2](head[1](head[0](e4)))    # hidden pre-activation, [B, 256]
        cls = head[5](head[4](head[3](z)))   # identical to BackboneUNet's classifier output

        u4 = n.dec4(torch.cat([n.up4(e4), e3], 1))
        u3 = n.dec3(torch.cat([n.up3(u4), e2], 1))
        u2 = n.dec2(torch.cat([n.up2(u3), e1], 1))
        u1 = n.dec1(torch.cat([n.up1(u2), e0], 1))
        seg = n.final(n.dec0(n.up0(u1)))

        B, K, H, W = e4.shape
        gate = (z > 0).to(e4.dtype)                                        # [B, 256]
        w2g = head[5].weight.unsqueeze(0) * gate.unsqueeze(1)              # [B, 37, 256]
        alpha = torch.matmul(w2g, head[2].weight) / (H * W)                # [B, 37, 1280]
        cam = torch.relu(torch.matmul(alpha, e4.reshape(B, K, H * W)))     # [B, 37, 49]
        return seg, cls, cam.reshape(B, -1, H, W)


def gradcam_autograd(net, x, c):
    """Reference Grad-CAM with PyTorch autograd, for one image x [1, 3, 224, 224] and class c."""
    store = {}

    def hook(module, inp, out):
        out.retain_grad()
        store["A"] = out

    h = net.stages[4].register_forward_hook(hook)
    _, cls = net(x)
    h.remove()
    net.zero_grad()
    cls[0, c].backward()
    A, g = store["A"], store["A"].grad
    alpha = g.mean(dim=(2, 3), keepdim=True)
    return torch.relu((alpha * A).sum(1))[0].detach()                    # [7, 7]
