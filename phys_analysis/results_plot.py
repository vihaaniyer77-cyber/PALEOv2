

import pandas as pd

df = pd.read_csv("preliminary_kepler_star_list.csv", on_bad_lines="skip")

all_results = []

for _, row in df.iterrows():
    kepid = int(row["kepid"])
    target = f"KIC {kepid}"

    print(f"\n=== {target} ===")

    try:
        # 1) Load LC
        lc = stitch_kepler_longcadence(target, flux_column="pdcsap_flux")

        # 2) Inference
        t_prob, p_prob = infer_star_probability_curve(
            model, lc, device, p_channel=0
        )

        # 3) Candidate extraction
        regions = extract_candidate_regions(t_prob, p_prob, p_thr=0.5)
        print("Candidate regions:", len(regions))

        for reg in regions:
            # 4) Physics-aware check
            t_seg, f_seg, sig = extract_region_flux(
                lc, reg["t_start"], reg["t_end"]
            )
            if t_seg is None:
                continue

            ok, batman_fit, params, diag = validate_transit_physics(
                t_seg, f_seg, sig
            )

            all_results.append({
                "kepid": kepid,
                "t_start": reg["t_start"],
                "t_end": reg["t_end"],
                "ok": ok,
                "best_params": params,
                "diagnostics": diag,
            })

            
            if ok:
                plot_diagnostic_one_panel(
                    lc,
                    t_prob,
                    p_prob,
                    result_dict={
                        "target": target,
                        "t_start": reg["t_start"],
                        "t_end": reg["t_end"],
                        "ok": ok,
                        "best_params": params,
                        "t_seg": t_seg,
                        "f_seg": f_seg,
                        "batman_model": batman_fit,
                    },
                    p_thr=0.5
                )

    except Exception as e:
        print(f"[SKIP] {target} failed: {e}")

print("\nDONE. Total regions tested:", len(all_results))
print("Accepted:", sum(r["ok"] for r in all_results))
