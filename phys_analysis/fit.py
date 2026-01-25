def validate_transit_physics(
    t_seg,
    f_seg,
    sigma,
    rmse_gain_thr: float = 1.0,
    # search bounds (in units BATMAN expects)
    rp_bounds=(0.01, 0.25),
    a_bounds=(5.0, 40.0),
    inc_bounds=(85.0, 90.0),
    # grid sizes
    n_rp=9,
    n_a=10,
    n_inc=7,
):
    
    t = np.asarray(t_seg, dtype=float)
    f = np.asarray(f_seg, dtype=float)

    if t.size < 10 or not (np.isfinite(sigma) and sigma > 0):
        return False, None, None, {"reason": "insufficient_data_or_bad_sigma"}

    # Null model: constant baseline
    f0 = np.median(f)
    rmse_null = float(np.sqrt(np.mean((f - f0) ** 2)) / sigma)

    # Initial t0: time of minimum flux (good local guess)
    t0 = float(t[np.argmin(f)])

    # Depth-based rp guess to center the scan)
    depth = max(0.0, float(f0 - np.min(f)))  # since LC normalized ~1
    rp_guess = np.sqrt(max(depth, 1e-8))
    rp_lo = max(float(rp_bounds[0]), 0.5 * rp_guess)
    rp_hi = min(float(rp_bounds[1]), 2.0 * rp_guess)
    if rp_hi <= rp_lo:  # fallback
        rp_lo, rp_hi = float(rp_bounds[0]), float(rp_bounds[1])

    rp_grid  = np.linspace(rp_lo, rp_hi, int(n_rp))
    a_grid   = np.linspace(float(a_bounds[0]), float(a_bounds[1]), int(n_a))
    inc_grid = np.linspace(float(inc_bounds[0]), float(inc_bounds[1]), int(n_inc))

    best_rmse = np.inf
    best_model = None
    best_params = None

    for rp in rp_grid:
        for a in a_grid:
            for inc in inc_grid:
                try:
                    m = generate_batman_model(t, t0=t0, rp=rp, a=a, inc=inc)
                    rmse = float(np.sqrt(np.mean((f - m) ** 2)) / sigma)
                except Exception:
                    continue

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = m
                    best_params = {"t0": t0, "rp": float(rp), "a": float(a), "inc": float(inc)}

    if best_model is None:
        return False, None, None, {"reason": "no_batman_model_succeeded"}

    gain = rmse_null - float(best_rmse)
    ok = bool(gain > float(rmse_gain_thr))

    diagnostics = {
        "rmse_null": rmse_null,
        "rmse_best": float(best_rmse),
        "rmse_gain": float(gain),
        "sigma": float(sigma),
        "depth_est": float(depth),
        "rp_guess": float(rp_guess),
    }

    if ok:
        return True, best_model, best_params, diagnostics
    return False, best_model, best_params, diagnostics
