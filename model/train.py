

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

#Hyperparams
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 5         
GRAD_CLIP = 1.0


THR_TRANSIT = 0.30
THR_FLARE   = 0.70

# Tempered pos_weight settings
PW_POWER = 0.5       
PW_MIN   = 1.0
PW_CAP_TRANSIT = 50.0
PW_CAP_FLARE   = 200.0

device = next(model.parameters()).device
ON_CUDA = (device.type == "cuda")

# AMP only on CUDA
USE_AMP = ON_CUDA
AMP_DTYPE = torch.float16

print("Device:", device, "| USE_AMP:", USE_AMP)

#Define T
T = int(X.shape[1])
print("T (cadences per window):", T)

#Dataset
class WindowDataset(Dataset):
    def __init__(self, X, Y, M, indices):
        self.X = X[indices]  # (N,T,1)
        self.Y = Y[indices]  # (N,T,2)
        self.M = M[indices]  # (N,T,1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float()  # (T,1)
        y = torch.from_numpy(self.Y[i]).float()  # (T,2)
        m = torch.from_numpy(self.M[i]).float()  # (T,1)

        # return (C,T)
        x = x.permute(1, 0).contiguous()  # (1,T)
        y = y.permute(1, 0).contiguous()  # (2,T)
        m = m.permute(1, 0).contiguous()  # (1,T)
        return x, y, m

train_ds = WindowDataset(X, Y, M, splits["train_idx"])
val_ds   = WindowDataset(X, Y, M, splits["val_idx"])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=ON_CUDA)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=ON_CUDA)
#Shape canonicalizer
def to_BCT(z, C, T, name="tensor"):
    if z.dim() != 3:
        raise RuntimeError(f"{name} must be 3D, got {z.dim()}D shape={tuple(z.shape)}")
    if z.shape[1] == C and z.shape[2] == T:
        return z
    if z.shape[1] == T and z.shape[2] == C:
        return z.permute(0, 2, 1).contiguous()
    raise RuntimeError(f"{name} wrong shape: expected (B,{C},{T}) or (B,{T},{C}), got {tuple(z.shape)}")


@torch.no_grad()
def compute_pos_weight_tempered(loader,
                                power=0.5,
                                pw_min=1.0,
                                cap_transit=50.0,
                                cap_flare=200.0):
    pos = torch.zeros(2, device=device)
    tot = torch.zeros(2, device=device)

    for xb, yb, mb in loader:
        yb = yb.to(device, non_blocking=ON_CUDA)
        mb = mb.to(device, non_blocking=ON_CUDA)

        yb = to_BCT(yb, 2, T, "yb")
        mb = to_BCT(mb, 1, T, "mb")
        mb2 = mb.expand(-1, 2, -1)

        tot += mb2.sum(dim=(0, 2))               
        pos += (yb * mb2).sum(dim=(0, 2))        

    neg = tot - pos

    
    pos = torch.clamp(pos, min=1.0)
    raw = neg / pos                             
    tempered = torch.pow(torch.clamp(raw, min=1.0), power)

    
    caps = torch.tensor([cap_transit, cap_flare], device=device)
    pw = torch.clamp(tempered, min=pw_min)
    pw = torch.min(pw, caps)
    return pw

pos_weight = compute_pos_weight_tempered(
    train_loader,
    power=PW_POWER,
    pw_min=PW_MIN,
    cap_transit=PW_CAP_TRANSIT,
    cap_flare=PW_CAP_FLARE
)

print("pos_weight TEMPERED [transit, flare] =", pos_weight.detach().cpu().numpy())

# reshape so it broadcasts across time dimension
pos_weight_broadcast = pos_weight.view(2, 1)  # (2,1) -> matches (B,2,T)

bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_broadcast)

def masked_bce_loss(logits, y, m):
    logits = to_BCT(logits, 2, T, "logits")
    y      = to_BCT(y,      2, T, "y")
    m      = to_BCT(m,      1, T, "m")

    m2 = m.expand(-1, 2, -1)
    loss_el = bce(logits, y) * m2
    return loss_el.sum() / torch.clamp(m2.sum(), min=1.0)


if USE_AMP:
    autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True)
    scaler = torch.amp.GradScaler(enabled=True)
else:
    autocast_ctx = lambda: nullcontext()
    scaler = None

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def dump_tensor(name, t):
    t2 = t.detach()
    finite = torch.isfinite(t2)
    msg = {
        "shape": tuple(t2.shape),
        "dtype": str(t2.dtype),
        "device": str(t2.device),
        "finite_frac": float(finite.float().mean().item()) if t2.numel() else 1.0,
    }
    if finite.any():
        msg["min"] = float(t2[finite].min().item())
        msg["max"] = float(t2[finite].max().item())
    print(name, msg)

#Train/Eval
@torch.no_grad()
def eval_loop(loader):
    model.eval()
    tot = 0.0
    n = 0
    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=ON_CUDA)
        yb = yb.to(device, non_blocking=ON_CUDA)
        mb = mb.to(device, non_blocking=ON_CUDA)

        xb = torch.nan_to_num(xb, nan=0.0)

        with autocast_ctx():
            logits = model(xb)
            loss = masked_bce_loss(logits, yb, mb)

        tot += float(loss.item())
        n += 1
    return tot / max(1, n)

def train_one_epoch(loader):
    model.train()
    tot = 0.0
    n = 0

    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=ON_CUDA)
        yb = yb.to(device, non_blocking=ON_CUDA)
        mb = mb.to(device, non_blocking=ON_CUDA)

        xb = torch.nan_to_num(xb, nan=0.0)

        optimizer.zero_grad(set_to_none=True)

        try:
            with autocast_ctx():
                logits = model(xb)
                loss = masked_bce_loss(logits, yb, mb)

            if USE_AMP:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        except Exception as e:
            print("\n=== TRAIN STEP FAILED ===")
            print("ERROR:", repr(e))
            dump_tensor("xb", xb)
            dump_tensor("yb(raw)", yb)
            dump_tensor("mb(raw)", mb)
            try:
                dump_tensor("logits", logits)
            except Exception:
                print("logits not available (failed before forward completed).")
            print("=========================\n")
            raise

        tot += float(loss.item())
        n += 1

    return tot / max(1, n)

for epoch in range(1, EPOCHS + 1):
    tr = train_one_epoch(train_loader)
    va = eval_loop(val_loader)
    print(f"Epoch {epoch:03d} | train_loss={tr:.4f} | val_loss={va:.4f}")
