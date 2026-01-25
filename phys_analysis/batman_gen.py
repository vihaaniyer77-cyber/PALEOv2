import batman

def generate_batman_model(
    t,
    t0,
    rp,
    a,
    inc,
    limb_dark="nonlinear",
    u=(0.5, 0.1, 0.1, -0.1),
):
    """
    Generate a BATMAN light curve on times t, for a single-transit local fit.

    Note: per is a dummy constant here;  NOT inferring orbital period.
    """
    t = np.asarray(t, dtype=float)

    params = batman.TransitParams()
    params.t0 = float(t0)
    params.per = 10.0       # dummy; irrelevant for local single-transit shape
    params.rp = float(rp)   # Rp/R*
    params.a  = float(a)    # a/R*
    params.inc = float(inc)
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = str(limb_dark)
    params.u = list(u)

    m = batman.TransitModel(params, t)
    return m.light_curve(params)
