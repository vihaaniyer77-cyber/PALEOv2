
import pandas as pd

# 1) Pick one star from CSV 
df = pd.read_csv("preliminary_kepler_star_list.csv", on_bad_lines="skip")
row = df.iloc[0]   # idx

kepid = int(row["kepid"])
target = f"KIC {kepid}"

print("Running physics-aware inference for:", target)

# 2) Load stitched light curve
lc = stitch_kepler_longcadence(target, flux_column="pdcsap_flux")

# 3) RUN MODEL
t_prob, p_prob = infer_star_probability_curve(
    model,
    lc,
    device,
    p_channel=0   # transit channel
)

print("Prob curve points:", len(t_prob))

# 4) Extract candidate regions
regions = extract_candidate_regions(t_prob, p_prob, p_thr=0.5)
print("Candidate regions:", len(regions))

# 5) Run physics-aware check on first region
results = []
if regions:
    reg = regions[0]

    t_seg, f_seg, sig = extract_region_flux(
        lc, reg["t_start"], reg["t_end"]
    )

    ok, batman_fit, params, diag = validate_transit_physics(
        t_seg, f_seg, sig
    )

    results.append({
        "target": target,
        "t_start": reg["t_start"],
        "t_end": reg["t_end"],
        "ok": ok,
        "best_params": params,
        "t_seg": t_seg,
        "f_seg": f_seg,
        "batman_model": batman_fit,
    })

# 6) One-panel diagnostic plot
plot_diagnostic_one_panel(
    lc,
    t_prob,
    p_prob,
    result_dict=results[0] if results else None,
    p_thr=0.5
)
