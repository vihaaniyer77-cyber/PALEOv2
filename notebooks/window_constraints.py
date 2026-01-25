import numpy as np

CADENCE_MIN = 29.4244      # Kepler long cadence (minutes)
WINDOW_DAYS = 5.0
OVERLAP_FRAC = 0.50        # 50% overlap, Solution: transits dont get smothered during clippings

MAX_MISSING_FRAC = 0.10    # reject if >10% nans
MAX_GAP_HOURS = 6.0        # reject if any gap >6 hours -- ensure good training data

def hours_to_days(h):
    return h / 24.0
