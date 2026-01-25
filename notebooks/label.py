import numpy as np
import pandas as pd
import requests

#knobs
CADENCE_DAYS = CADENCE_MIN / (24.0 * 60.0)
T = int(round(WINDOW_DAYS * 24.0 * 60.0 / CADENCE_MIN))  # ~245 for 5d Kepler LC

MAX_GAP_HOURS_FOR_GRID = 1.0   # grid point invalid if > this from nearest real cadence
BASELINE_DAYS = 1.0            # rolling median baseline window for dip/flare scoring

# Transit dip gate  
K_DIP = 1.5
MIN_CONSEC_TRANSIT = 2


FLARE_W_DAYS       = 2.5
FLARE_K_AMP        = 5.0
FLARE_K_RISE       = 3.0
FLARE_K_DECAY      = 2.5
FLARE_MIN_LEN      = 2
FLARE_MAX_LEN_HRS  = 10.0
FLARE_GAP_MULT     = 3.0

# Reproducible split
SPLIT_SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15



# Robust stats helpers
def mad_sigma(x):
    """Robust sigma estimate from MAD. Ignores non-finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def rolling_median_baseline(f_grid, win_pts):
    """
    Rolling median baseline on grid flux.
    Uses pandas rolling; handles NaNs.
    """
    s = pd.Series(f_grid)
    base = s.rolling(win_pts, center=True, min_periods=max(5, win_pts // 10)).median()
    base = base.bfill().ffill()
    return base.values.astype(float)


def runlength_filter(mask, min_len):
    """Keep only True-runs with length >= min_len."""
    mask = np.asarray(mask, dtype=bool)
    if min_len <= 1:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) >= min_len:
            out[i:j] = True
        i = j
    return out



# Time system helpers (BJD -> LC time)

def lc_time_offset_days(lc):
    """
    Safer heuristic:
      - Kepler BKJD times are ~[0, 1600] (certainly < 1e5)
      - JD/BJD times are ~2.4e6
    """
    tv = np.asarray(lc.time.value, dtype=float)
    med = np.nanmedian(tv[np.isfinite(tv)]) if np.any(np.isfinite(tv)) else np.nan
    if np.isfinite(med) and med < 1e5:
        return 2454833.0
    return 0.0


def bjd_to_lc_time(lc, bjd):
    """Convert BJD to the same units as lc.time.value."""
    return np.asarray(bjd, dtype=float) - lc_time_offset_days(lc)


# Exoplanet Archive (TAP sync) —accounts for multi-planet ephemerides per hostname

_EPHEM_CACHE = {}

def fetch_ephemerides_for_host(hostname):
    hn = str(hostname).strip()
    if hn == "" or hn.lower() == "nan":
        return []
    if hn in _EPHEM_CACHE:
        return _EPHEM_CACHE[hn]

    query = f"""
        SELECT pl_name, pl_orbper, pl_tranmid, pl_trandur
        FROM ps
        WHERE hostname = '{hn.replace("'", "''")}'
          AND pl_orbper IS NOT NULL
          AND pl_tranmid IS NOT NULL
          AND pl_trandur IS NOT NULL
    """.strip().replace("\n", " ")

    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    params = {"query": query, "format": "json"}

    try:
        rows = requests.get(url, params=params, timeout=30).json()
    except Exception:
        _EPHEM_CACHE[hn] = []
        return []

    planets_by_name = {}
    for row in rows:
        try:
            name = str(row["pl_name"]).strip()
            P = float(row["pl_orbper"])
            t0 = float(row["pl_tranmid"])
            dur_raw = float(row["pl_trandur"])

           #accounts for hours to days
            dur_days = dur_raw / 24.0
            if dur_raw > 48.0:
                dur_days = dur_raw  # assume already days for big cases

            if name not in planets_by_name:
                planets_by_name[name] = {
                    "name": name,
                    "P_days": P,
                    "t0_bjd": t0,
                    "dur_days": dur_days
                }
        except Exception:
            continue

    planets = list(planets_by_name.values())
    _EPHEM_CACHE[hn] = planets
    return planets


#Gridding for model input
def build_fixed_grid(tw, fw, start):
    """
    Given irregular (tw, fw) points in a window and the window start,
    build:
      t_grid (T,)
      f_grid (T,) with NaNs where outside hull OR too far from nearest real cadence
      M     (T,)  1 where finite else 0
    """
    t_grid = start + CADENCE_DAYS * np.arange(T, dtype=float)

    good = np.isfinite(tw) & np.isfinite(fw)
    tw2 = np.asarray(tw[good], dtype=float)
    fw2 = np.asarray(fw[good], dtype=float)

    if tw2.size < 2:
        f_grid = np.full(T, np.nan, dtype=float)
        M = np.zeros(T, dtype=float)
        return t_grid, f_grid, M

    order = np.argsort(tw2)
    tws = tw2[order]
    fws = fw2[order]

    f_interp = np.interp(t_grid, tws, fws)  # numeric
    outside = (t_grid < tws[0]) | (t_grid > tws[-1])
    f_interp[outside] = np.nan

    idx = np.searchsorted(tws, t_grid, side="left")
    left_idx = np.clip(idx - 1, 0, tws.size - 1)
    right_idx = np.clip(idx, 0, tws.size - 1)

    d_left = np.abs(t_grid - tws[left_idx])
    d_right = np.abs(t_grid - tws[right_idx])
    d_near = np.minimum(d_left, d_right)

    max_gap_days = hours_to_days(MAX_GAP_HOURS_FOR_GRID)
    f_grid = f_interp.copy()
    f_grid[d_near > max_gap_days] = np.nan

    M = np.isfinite(f_grid).astype(float)
    return t_grid, f_grid, M


#Transit Labeling
BASELINE_DAYS_TRANSIT = 0.3  

def label_transits_on_grid(lc, hostname, t_grid, f_grid, M, debug=False):
    planets = fetch_ephemerides_for_host(hostname)
    if len(planets) == 0:
        return np.zeros(T, dtype=float)

    win_pts = max(5, int(round(BASELINE_DAYS_TRANSIT / CADENCE_DAYS)))
    baseline = rolling_median_baseline(f_grid, win_pts)
    resid = f_grid - baseline

    valid = (M > 0.5) & np.isfinite(resid)
    sig = mad_sigma(resid[valid])
    if not np.isfinite(sig) or sig <= 0:
        return np.zeros(T, dtype=float)

    dip_ok = (resid < (-K_DIP * sig)) & (M > 0.5)

    y_union = np.zeros(T, dtype=bool)

    for p in planets:
        P = float(p["P_days"])
        t0_lc = float(bjd_to_lc_time(lc, p["t0_bjd"]))
        dur_days = float(p["dur_days"])
        if not (np.isfinite(P) and P > 0 and np.isfinite(dur_days) and dur_days > 0):
            continue

        phase = ((t_grid - t0_lc) / P) % 1.0
        dphi = np.minimum(phase, 1.0 - phase)
        half_width = 0.5 * (dur_days / P)
        phase_ok = dphi < half_width

        y = phase_ok & dip_ok

        dur_pts = int(round(dur_days / CADENCE_DAYS))
        min_consec = max(3, dur_pts // 2)
        y = runlength_filter(y, min_consec)

        if debug:
            print(
                f"[{hostname} | {p['name']}] "
                f"phase_ok={phase_ok.sum()} dip_ok={dip_ok.sum()} both={y.sum()} "
                f"dur_pts≈{dur_pts} min_consec={min_consec}"
            )
            debug = False

        y_union |= y

    return y_union.astype(float)


#Flare Labeling Methods
def flare_flags_full_lc(
    t, f, cadence_days,
    W_DAYS=FLARE_W_DAYS,
    K_AMP=FLARE_K_AMP,
    K_RISE=FLARE_K_RISE,
    K_DECAY=FLARE_K_DECAY,
    MIN_LEN=FLARE_MIN_LEN,
    MAX_LEN_HOURS=FLARE_MAX_LEN_HRS,
    GAP_MULT=FLARE_GAP_MULT
):
    t = np.asarray(t, float)
    f = np.asarray(f, float)

    # rolling-median baseline
    W = max(7, int(round(W_DAYS / cadence_days)))
    if W % 2 == 0:
        W += 1
    baseline = pd.Series(f).rolling(window=W, center=True, min_periods=max(5, W // 3)).median().to_numpy()
    resid = f - baseline

    # gap mask
    dt = np.diff(t)
    gap_ok = np.ones_like(t, dtype=bool)
    gap_ok[1:] = dt < (GAP_MULT * cadence_days)
    gap_ok[:-1] &= dt < (GAP_MULT * cadence_days)

    sigma_res = mad_sigma(resid[gap_ok])
    if not (np.isfinite(sigma_res) and sigma_res > 0):
        return np.zeros_like(t, dtype=bool)

    # 2-cadence derivative 
    dr2 = np.full_like(resid, np.nan)
    dr2[2:] = resid[2:] - resid[:-2]
    sigma_dr = mad_sigma(dr2[gap_ok])
    if not (np.isfinite(sigma_dr) and sigma_dr > 0):
        return np.zeros_like(t, dtype=bool)

    amp_thr   = K_AMP   * sigma_res
    rise_thr  = K_RISE  * sigma_dr
    decay_thr = K_DECAY * sigma_res

    hot = gap_ok & np.isfinite(resid) & (resid > amp_thr)
    idx = np.where(hot)[0]

    flare_flag = np.zeros_like(t, dtype=bool)
    if idx.size == 0:
        return flare_flag

    MAX_LEN = int(round((MAX_LEN_HOURS / 24.0) / cadence_days))
    MAX_LEN = max(MAX_LEN, MIN_LEN)

    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, splits + 1]
    ends   = np.r_[splits, idx.size - 1]

    for a, b in zip(starts, ends):
        seg = idx[a:b+1]
        L = len(seg)
        if L < MIN_LEN or L > MAX_LEN:
            continue

        pk = seg[np.argmax(resid[seg])]

        # fast rise 
        if not (np.isfinite(dr2[pk]) and dr2[pk] > rise_thr):
            continue

        # require some decay within next few cadences
        j2 = min(len(resid) - 1, pk + 4)
        post = resid[pk+1:j2+1]
        if post.size == 0 or not np.all(np.isfinite(post)):
            continue
        if not (np.nanmax(resid[pk] - post) > decay_thr):
            continue

        lo = max(0, seg[0] - 1)
        hi = min(len(resid), seg[-1] + 2)
        flare_flag[lo:hi] = True

    return flare_flag


def detect_flares_full_lc(lc):
    """
   
      - uses the event-based flare_flags_full_lc() you provided (inspired by your working cell)
      - returns (t_full, flare_flag_full) on the stitched LC cadences
    """
    # Use the same flux stream used for the dataset 
    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float)

    good = np.isfinite(t) & np.isfinite(f)
    t = t[good]; f = f[good]
    if t.size < 10:
        return t, np.zeros_like(t, dtype=bool)

    order = np.argsort(t)
    t = t[order]; f = f[order]

    dts = np.diff(t)
    cadence_days_est = np.nanmedian(dts[np.isfinite(dts)]) if dts.size else np.nan
    if not (np.isfinite(cadence_days_est) and cadence_days_est > 0):
        cadence_days_est = CADENCE_DAYS

    flare_flag = flare_flags_full_lc(t, f, cadence_days=cadence_days_est)
    return t, flare_flag


def map_flags_to_grid(t_full, flag_full, t_grid):
    """
    Nearest-neighbor map from (t_full, flag_full) to t_grid.
    If nearest real cadence is farther than MAX_GAP_HOURS_FOR_GRID, return False.
    """
    t_full = np.asarray(t_full, dtype=float)
    flag_full = np.asarray(flag_full, dtype=bool)

    if t_full.size == 0:
        return np.zeros(T, dtype=float)

    order = np.argsort(t_full)
    ts = t_full[order]
    fs = flag_full[order]

    idx = np.searchsorted(ts, t_grid, side="left")
    left = np.clip(idx - 1, 0, ts.size - 1)
    right = np.clip(idx, 0, ts.size - 1)

    d_left = np.abs(t_grid - ts[left])
    d_right = np.abs(t_grid - ts[right])

    choose_right = d_right < d_left
    nearest_idx = np.where(choose_right, right, left)
    d_near = np.minimum(d_left, d_right)

    ok = d_near <= hours_to_days(MAX_GAP_HOURS_FOR_GRID)
    out = np.zeros(T, dtype=bool)
    out[ok] = fs[nearest_idx[ok]]
    return out.astype(float)



# Normalization for X (masked, robust per-window)

def normalize_window_flux(f_grid, M):
    """
    Robust per-window normalization:
      (f - median) / sigma
    Stores NaNs as 0 (mask handles loss).
    """
    fg = np.asarray(f_grid, dtype=float).copy()
    Mv = np.asarray(M, dtype=float)
    valid = (Mv > 0.5) & np.isfinite(fg)

    if valid.sum() < 10:
        return np.nan_to_num(fg, nan=0.0).astype(np.float32)

    med = np.median(fg[valid])
    sig = mad_sigma(fg[valid] - med)
    if not np.isfinite(sig) or sig <= 0:
        sig = np.std(fg[valid]) if valid.sum() > 1 else 1.0
        if not np.isfinite(sig) or sig <= 0:
            sig = 1.0

    z = (fg - med) / sig
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


#Build Dataset
X_list, Y_list, M_list = [], [], []
t_starts_list, kepid_list, host_list = [], [], []
per_star_window_counts = {}

for _, row in df.iterrows():
    kepid = int(row["kepid"])
    hostname = str(row.get("hostname", "")).strip()
    target = f"KIC {kepid}"

    try:
        lc = stitch_kepler_longcadence(target, flux_column="pdcsap_flux")

        # Flare flags once per star 
        t_full, flare_flag_full = detect_flares_full_lc(lc)

        # Candidate windows 
        windows = window_lc(lc)
        per_star_window_counts[kepid] = len(windows)

        for (tw, fw, start, end) in windows:
            # Fixed grid + mask
            t_grid, f_grid, M0 = build_fixed_grid(tw, fw, start)

            # X (normalized, NaNs->0)
            x = normalize_window_flux(f_grid, M0)[:, None]  

            # Y transit 
            y_tr = label_transits_on_grid(lc, hostname, t_grid, f_grid, M0)  # (T,)

            # Y flare 
            y_fl = map_flags_to_grid(t_full, flare_flag_full, t_grid)        # (T,)
            y_fl = y_fl * (M0 > 0.5).astype(float)

            Yw = np.stack([y_tr, y_fl], axis=1).astype(np.float32)           # (T,2)
            Mw = M0.astype(np.float32)[:, None]                              # (T,1)

            X_list.append(x.astype(np.float32))
            Y_list.append(Yw)
            M_list.append(Mw)
            t_starts_list.append(float(start))
            kepid_list.append(kepid)
            host_list.append(hostname)

    except Exception as e:
        print(f"[SKIP] {target} failed: {e}")


#Stack Dataset Arrays
if len(X_list) == 0:
    raise RuntimeError("No windows were produced.")

X = np.stack(X_list, axis=0).astype(np.float32)  # (N,T,1)
Y = np.stack(Y_list, axis=0).astype(np.float32)  # (N,T,2)
M = np.stack(M_list, axis=0).astype(np.float32)  # (N,T,1)

t_starts = np.asarray(t_starts_list, dtype=np.float64)
kepid_per_window = np.asarray(kepid_list, dtype=np.int64)
host_per_window = np.asarray(host_list, dtype=object)

#Split by Star
unique_stars = np.unique(kepid_per_window)
rng = np.random.default_rng(SPLIT_SEED)
rng.shuffle(unique_stars)

n_stars = unique_stars.size
n_train = int(round(TRAIN_FRAC * n_stars))
n_val = int(round(VAL_FRAC * n_stars))

train_stars = unique_stars[:n_train]
val_stars = unique_stars[n_train:n_train + n_val]
test_stars = unique_stars[n_train + n_val:]

is_train = np.isin(kepid_per_window, train_stars)
is_val   = np.isin(kepid_per_window, val_stars)
is_test  = np.isin(kepid_per_window, test_stars)

train_idx = np.where(is_train)[0]
val_idx   = np.where(is_val)[0]
test_idx  = np.where(is_test)[0]

splits = {
    "train_stars": train_stars,
    "val_stars": val_stars,
    "test_stars": test_stars,
    "train_idx": train_idx,
    "val_idx": val_idx,
    "test_idx": test_idx,
}

#Diagnostics
valid_total = float(np.sum(M))
transit_rate = float(np.sum(Y[:, :, 0:1] * M) / (valid_total + 1e-12))
flare_rate   = float(np.sum(Y[:, :, 1:2] * M) / (valid_total + 1e-12))

windows_with_transit = int(np.sum((np.sum(Y[:, :, 0] * M[:, :, 0], axis=1) > 0)))
windows_with_flare   = int(np.sum((np.sum(Y[:, :, 1] * M[:, :, 0], axis=1) > 0)))

print("==== DATASET BUILD COMPLETE ====")
print(f"Stars processed (with >=1 window): {np.unique(kepid_per_window).size}")
print(f"Total windows N: {X.shape[0]} | T: {X.shape[1]}")
print(f"Avg valid fraction: {valid_total / (X.shape[0] * T):.4f}")
print(f"Transit labeled cadence rate: {transit_rate:.6e}")
print(f"Flare labeled cadence rate:   {flare_rate:.6e}")
print(f"Windows with any transit labels: {windows_with_transit}")
print(f"Windows with any flare labels:   {windows_with_flare}")
print(f"Split stars: train={train_stars.size}, val={val_stars.size}, test={test_stars.size}")
print(f"Split windows: train={train_idx.size}, val={val_idx.size}, test={test_idx.size}")
