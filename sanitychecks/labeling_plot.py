import numpy as np
import matplotlib.pyplot as plt


# Pick a random window 
-
rng = np.random.default_rng()

# Prefer a window that has *something* labeled
event_windows = np.where(
    (np.sum(Y[:, :, 0] * M[:, :, 0], axis=1) > 0) |
    (np.sum(Y[:, :, 1] * M[:, :, 0], axis=1) > 0)
)[0]

if len(event_windows) > 0:
    idx = rng.choice(event_windows)
else:
    idx = rng.integers(0, X.shape[0])

# Extract window
x = X[idx, :, 0]
y_tr = Y[idx, :, 0]
y_fl = Y[idx, :, 1]
m = M[idx, :, 0]
t0 = t_starts[idx]
kepid = kepid_per_window[idx]
host = host_per_window[idx]

# Time axis 
t = np.arange(len(x))

#Build Masks
valid = m > 0.5
bg = valid & (y_tr == 0) & (y_fl == 0)
tr = valid & (y_tr == 1) & (y_fl == 0)
fl = valid & (y_tr == 0) & (y_fl == 1)
both = valid & (y_tr == 1) & (y_fl == 1)
invalid = ~valid

#Plot
plt.figure(figsize=(12, 4))

plt.scatter(t[invalid], x[invalid], s=8, c="lightgray", label="masked", zorder=1)
plt.scatter(t[bg],      x[bg],      s=10, c="black",     label="background", zorder=2)
plt.scatter(t[tr],      x[tr],      s=18, c="orange",    label="transit",    zorder=3)
plt.scatter(t[fl],      x[fl],      s=18, c="red",       label="flare",      zorder=4)
plt.scatter(t[both],    x[both],    s=30, c="purple",    label="both",       zorder=5)

plt.axhline(0, color="gray", lw=0.5, ls="--")

plt.title(
    f"KIC {kepid} | {host}\n"
    f"Window start = {t0:.5f} (LC time units)"
)
plt.xlabel("Cadence index (grid)")
plt.ylabel("Normalized flux")

plt.legend(ncol=5, fontsize=9, loc="upper right")
plt.tight_layout()
plt.show()
