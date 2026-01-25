def extract_candidate_regions(t_full, p_full, p_thr: float = 0.5, min_pts: int = 2):
    #connects
    t_full = np.asarray(t_full, dtype=float)
    p_full = np.asarray(p_full, dtype=float)

    if t_full.size == 0:
        return []

    good = np.isfinite(t_full) & np.isfinite(p_full)
    t = t_full[good]
    p = p_full[good]
    if t.size == 0:
        return []

    mask = p >= float(p_thr)
    if not np.any(mask):
        return []

    idx = np.where(mask)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, splits + 1]
    ends   = np.r_[splits, idx.size - 1]

    regions = []
    for a, b in zip(starts, ends):
        seg = idx[a:b+1]
        if seg.size < int(min_pts):
            continue
        i0 = int(seg[0])
        i1 = int(seg[-1])
        regions.append({
            "t_start": float(t[i0]),
            "t_end":   float(t[i1]),
            "i0": i0,
            "i1": i1
        })

    return regions
