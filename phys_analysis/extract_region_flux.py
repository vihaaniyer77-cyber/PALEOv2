def extract_region_flux(lc, t_start, t_end, pad_days: float = 0.25):
    """
    Extract cadence-level (t,f) for the candidate region (optionally padded).
    Computes robust noise sigma using MAD of residuals about the median.

    Returns: (t_seg, f_seg, sigma)
      or (None, None, None) if insufficient data
    """
    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float)

    good = np.isfinite(t) & np.isfinite(f)
    t = t[good]; f = f[good]
    if t.size < 10:
        return None, None, None

    lo = float(t_start) - float(pad_days)
    hi = float(t_end)   + float(pad_days)

    m = (t >= lo) & (t <= hi)
    t_seg = t[m]
    f_seg = f[m]
    if t_seg.size < 10:
        return None, None, None

    med = np.median(f_seg)
    sig = mad_sigma(f_seg - med)
    if not (np.isfinite(sig) and sig > 0):
        return None, None, None

    return t_seg, f_seg, sig
