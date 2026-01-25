import pandas as pd

def window_lc(lc):
    t = lc.time.value
    f = lc.flux.value.astype(float)

    
    order = np.argsort(t)
    t = t[order]
    f = f[order]

    win  = WINDOW_DAYS
    step = WINDOW_DAYS * (1 - OVERLAP_FRAC)

    cadence_days = CADENCE_MIN / (24 * 60)
    start0 = np.floor(t.min() / cadence_days) * cadence_days

    windows = []
    start = start0
    tmax = t.max()

    while start + win <= tmax:
        end = start + win

        i0 = np.searchsorted(t, start, side="left")
        i1 = np.searchsorted(t, end,   side="left")

        if i1 - i0 > 0:
            tw = t[i0:i1]
            fw = f[i0:i1]
            if is_valid_window(tw, fw):
                windows.append((tw, fw, start, end))

        start += step

    return windows


df = pd.read_csv("preliminary_kepler_star_list.csv")
all_windows = {}

for _, row in df.iterrows():
    kepid = int(row["kepid"])
    target = f"KIC {kepid}"

    try:
        
        lc = stitch_kepler_longcadence(target, flux_column="pdcsap_flux")

        windows = window_lc(lc)
        all_windows[kepid] = windows

        print(f"{target} | Valid windows: {len(windows)}")

    except Exception as e:
        print(f"[SKIP] {target} failed: {e}")
