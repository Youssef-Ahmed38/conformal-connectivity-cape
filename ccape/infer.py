import torch
import torch.nn.functional as F


@torch.no_grad()
def infer(model, img, T, dev="cuda", chunk=10):
    """One image (3,H,W), any size -> p_det (H,W) from eval mode, p_mc (T,H,W) from MC-Dropout."""
    H, W = img.shape[-2:]
    x = F.pad(torch.from_numpy(img[None]).to(dev), (0, -W % 16, 0, -H % 16))
    with torch.autocast("cuda", torch.bfloat16):
        model.eval(); p_det = torch.sigmoid(model(x).float())[0, :H, :W]
        model.mc_mode()
        p_mc = torch.cat([torch.sigmoid(model(x.expand(min(chunk, T - k), -1, -1, -1)).float())
                          for k in range(0, T, chunk)])[:, :H, :W]
    model.eval()
    return p_det.cpu().numpy(), p_mc.cpu().numpy()
