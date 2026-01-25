def is_valid_window(t, f): #decides whether a candidate time window is usable
    # finite points + time points so that everything is alright for training
    good = np.isfinite(t) & np.isfinite(f) # ensures no nans|infinities -- formality
    t_good = t[good]
    f_good = f[good]
    if len(t_good) < 2: #--- need multiple points
        return False

    # reject if big gap between good points
    if np.max(np.diff(np.sort(t_good))) > hours_to_days(MAX_GAP_HOURS):
        return False

    # require enough points based on MAX_MISSING_FRAC variable
    expected = int(round(WINDOW_DAYS * 24 * 60 / CADENCE_MIN))  # expected cadences in window
    min_points = int(np.ceil((1.0 - MAX_MISSING_FRAC) * expected))

    if len(t_good) < min_points:
        return False

    return True

