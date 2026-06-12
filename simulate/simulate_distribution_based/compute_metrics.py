import numpy as np

def compute_metrics(history, N0):
    """
    history : dict with keys Hi, Ot, In, Im, De — each a 1-D numpy array of length T
    N0      : int, initial total population (scalar)
    Returns : dict of scalar metrics
    """
    T = len(history["Hi"])
    alive = (np.array(history["Hi"])
           + np.array(history["Ot"])
           + np.array(history["In"])
           + np.array(history["Im"]))   # := N(t)

    # population metrics
    N_min = alive.min() # min population ever reached
    N_max = alive.max() # max population ever reached
    N_final = alive[-1] # final pop

    # prevalence: P(t) = In(t) / N(t)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.where(alive > 0, np.array(history["In"]) / alive, 0.0)
    P_max   = P.max()                           # peak prevalence
    T_Pmax  = int(np.argmax(P))                 # day of peak prevalence
    P_avg   = P.mean()                          # time-averaged prevalence

    # persistence: S(t) = N(t) / N(0)
    S = alive / N0
    S_final = S[-1]                             # final persistence

    # mortality burded: M(t) = De(t) / N(0)
    M       = np.array(history["De"]) / N0
    M_final = M[-1]
    M_max   = M.max()
    death_days = np.where(np.array(history["De"]) > 0)[0]
    T_De    = int(death_days[0]) if len(death_days) > 0 else np.nan  # first death

    # disease invasion rate: I_new(t) = In(t+1) - In(t)
    In_arr  = np.array(history["In"])
    I_new   = np.diff(In_arr, prepend=In_arr[0])                     # new infections per day
    with np.errstate(invalid='ignore', divide='ignore'):
        I_rate = np.where(alive > 0, I_new / alive, 0.0)             # per-capita rate

    return {
        "N_min":    N_min,
        "N_final":  N_final,
        "P_max":    P_max,
        "T_Pmax":   T_Pmax,
        "P_avg":    P_avg,
        "S_final":  S_final,
        "M_final":  M_final,
        "M_max":    M_max,
        "T_De":     T_De,
        # full time-series (needed for Monte Carlo bands)
        "_P":       P,
        "_S":       S,
        "_M":       M,
        "_I_rate":  I_rate,
        "_alive":   alive,
    }