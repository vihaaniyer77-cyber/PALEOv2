import pandas as pd

def run_physics_aware_for_row(
    model,
    row,
    device,
    p_thr: float = 0.5,
    rmse_gain_thr: float = 1.0,
    max_regions_per_star: int = 10,
):
    """
    Runs: stitch LC -> infer p_transit(t) -> extract regions -> BATMAN validate.
    'row' is one row from preliminary_kepler_star_list.csv.

    Returns:
      (lc, t_prob, p_prob, results)
    where results is a list of dicts with accepted candidates and fit info.
    """
    kepid = int(row["kepid"])
    hostname = str(row.get("hostname", "")).strip()
    target = f"KIC {kepid}"

    lc = stitch_kepler_longcadence(target, flux_column="pdcsap_flux")

    # 1) ML inference: star-level probability curve
    t_prob, p_prob = infer_star_probability_curve(model, lc, device, p_channel=0)

    # 2) Candidate extraction
    regions = extract_candidate_regions(t_prob, p_prob, p_thr=p_thr, min_pts=2)
    regions = regions[:int(max_regions_per_star)]

    results = []
    for reg in regions:
        t_start, t_end = reg["t_start"], reg["t_end"]

        # 3) Extract flux around region
        t_seg, f_seg, sig = extract_region_flux(lc, t_start, t_end, pad_days=0.25)
        if t_seg is None:
            continue

        # 4) Physics-aware check
        ok, best_model, best_params, diag = validate_transit_physics(
            t_seg, f_seg, sig,
            rmse_gain_thr=rmse_gain_thr
        )

        results.append({
            "kepid": kepid,
            "hostname": hostname,
            "target": target,
            "t_start": float(t_start),
            "t_end": float(t_end),
            "ok": bool(ok),
            "best_params": best_params,     # plausible example params
            "diagnostics": diag,
            "t_seg": t_seg,                 # keep for plotting
            "f_seg": f_seg,
            "batman_model": best_model,     # overlay for plotting if ok
        })

    return lc, t_prob, p_prob, results


def run_physics_aware_over_csv(
    model,
    csv_path="preliminary_kepler_star_list.csv",
    device=None,
    p_thr: float = 0.5,
    rmse_gain_thr: float = 1.0,
    max_stars: int | None = None,
):
    
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    if max_stars is not None:
        df = df.head(int(max_stars))

    all_results = []
    for _, row in df.iterrows():
        try:
            lc, t_prob, p_prob, results = run_physics_aware_for_row(
                model, row, device=device,
                p_thr=p_thr, rmse_gain_thr=rmse_gain_thr
            )
            all_results.extend(results)
            print(f"[OK] KIC {int(row['kepid'])} | regions tested={len(results)} | accepted={sum(r['ok'] for r in results)}")
        except Exception as e:
            print(f"[SKIP] KIC {row.get('kepid','?')} failed: {e}")

    return all_results
