import torch
import numpy as np
import matplotlib.pyplot as plt

model.eval()

# ----------------------------
# Pick ONE validation window index
# (change idx to look at different windows)
# ----------------------------
idx = 0   # index inside val_ds

# Grab data
x, y, m = val_ds[idx]        # x:(1,T), y:(2,T), m:(1,T)
x = x.unsqueeze(0).to(device)  # (1,1,T)

# Forward pass
with torch.no_grad():
    logits = model(x)                # (1,2,T)
    probs = torch.sigmoid(logits)    # (1,2,T)

# Move to CPU
x = x.cpu().numpy()[0, 0]
p_transit = probs.cpu().numpy()[0, 0]
p_flare   = probs.cpu().numpy()[0, 1]

# ----------------------------
# Plot
# ----------------------------
T = x.shape[0]
t = np.arange(T)

fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

axes[0].plot(t, x, color="black")
axes[0].set_ylabel("Flux")
axes[0].set_title("Light Curve")

axes[1].plot(t, p_transit, color="tab:blue")
axes[1].set_ylabel("P(transit)")
axes[1].set_ylim(0, 1)

axes[2].plot(t, p_flare, color="tab:red")
axes[2].set_ylabel("P(flare)")
axes[2].set_ylim(0, 1)
axes[2].set_xlabel("Cadence")

plt.tight_layout()
plt.show()
