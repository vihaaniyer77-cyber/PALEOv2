import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt

#Stitch
def stitch_kepler_longcadence(target, flux_column="pdcsap_flux"):

    # Search all Kepler long-cadence products directly
    sr = lk.search_lightcurve(target, mission="Kepler", cadence="long")
    lcc = sr.download_all(flux_column=flux_column)

    # Remove empty / None
    lcs = [lc for lc in (lcc or []) if lc is not None and len(lc.time) > 0]
    assert len(lcs) > 0, f"No Kepler long-cadence LCs found for {target}."

    # Stitch + clean + normalize + sort
    lc_stitched = lk.LightCurveCollection(lcs).stitch()
    lc_stitched = lc_stitched.remove_nans().normalize()

    tmp = lc_stitched.sort()
    if tmp is not None:
        lc_stitched = tmp

    # Remove duplicate timestamps
    t = lc_stitched.time.value
    _, keep_idx = np.unique(t, return_index=True)
    keep_idx = np.sort(keep_idx)
    lc_stitched = lc_stitched[keep_idx]

    return lc_stitched


#Load CSV
df = pd.read_csv("preliminary_kepler_star_list.csv", on_bad_lines="skip")

#Loop Over Stars
for i, row in df.iterrows():
    kepid = int(row["kepid"])
    name  = str(row["hostname"])
    target = f"KIC {kepid}"  

    try:
        lc = stitch_kepler_longcadence(target)

        ax = lc.plot(label=f"{name} PDCSAP stitched")
        ax.set_title(f"{name} (KIC {kepid}) — stitched long cadence")
        plt.show()

    except Exception as e:
        print(f"[SKIP] {name} (KIC {kepid}) failed: {e}")
