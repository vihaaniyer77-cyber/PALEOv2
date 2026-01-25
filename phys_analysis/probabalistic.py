import numpy as np
import torch
from collections import defaultdict

@torch.no_grad()
def infer_star_probability_curve(
    model,
    lc,
    device,
    p_channel: int = 0,   # 0=transit, 1=flare
):
    """
    Runs the trained TCN over all valid windows of a star and stitches overlapping
    window predictions into a single star-level probability curve.

    Returns:
      t_full  (N,) sorted unique times (same units as lc.time.value)
      p_full  (N,) averaged probabilities in [0,1]
    """
    model.eval()

    windows = window_lc(lc)  # uses your is_valid_window() filtering
    if len(windows) == 0:
        return np.array([]), np.array([])

    # accumulate multiple window predictions per absolute time grid point
    prob_accum = defaultdict(list)

    for (tw, fw, start, end) in windows:
        t_grid, f_grid, M = build_fixed_grid(tw, fw, start)

        # build model input X: (1,1,T)
        x = normalize_window_flux(f_grid, M)              # (T,)
        xb = torch.from_numpy(x[None, :, None]).float()   # (1,T,1)
        xb = xb.permute(0, 2, 1).to(device)               # (1,1,T)

        logits = model(xb)                                # (1,2,T)
        probs = torch.sigmoid(logits)[0, p_channel].detach().cpu().numpy()  # (T,)

        # stash per-cadence probability at absolute times
        for ti, pi in zip(t_grid, probs):
            prob_accum[float(ti)].append(float(pi))

    # average overlapping predictions
    t_full = np.array(sorted(prob_accum.keys()), dtype=float)
    p_full = np.array([np.mean(prob_accum[t]) for t in t_full], dtype=float)

    return t_full, p_full
