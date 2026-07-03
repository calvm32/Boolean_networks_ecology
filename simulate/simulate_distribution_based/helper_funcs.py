import matplotlib.pyplot as plt
import numpy as np
from simulate.simulate_distribution_based.rules import *

# ---------------------
# setup before each run
# ---------------------

def make_initial_state(Hi_list, fraction_infected):
    # NOTICE : each inhabitant node contains the following information:
    # [ ON/OFF, 
    #   resistance number AKA res_num, 
    #   clustering number AKA mu_i, 
    #   days left infirm, 
    #   0 for just entered hibernation OR 1 for exited hibernation at least once (to track arousal periods),
    #   days left immune
    # ]
    
    empty_pop = []

    return {
        "Hi": [
                [1, 0, rand.uniform(Hi_list[i][1], Hi_list[i][2]), 0, 0, 0]
                for i in range(len(Hi_list))
                for _ in range(Hi_list[i][0] - int(Hi_list[i][0]*fraction_infected))
              ],
        "Ot": empty_pop.copy(),
        "In": [
                [1, 0, rand.uniform(Hi_list[i][1], Hi_list[i][2]), 0, 0, 0]
                for i in range(len(Hi_list))
                for _ in range(int(Hi_list[i][0]*fraction_infected))
              ],
        "Im": empty_pop.copy(),
        "De": 0, # only need total numbers of dead
        "Re": 1,
        "Te": 0,
        "Hu": 0,
        "PD": 0,
    }

# ----------------------------------------
# computing individaul & population values
# ----------------------------------------

def step(state, parameters, t):
    agg = aggregate(state)

    env_next = update_environment(state, agg, parameters, t)
    pop_next = update_individuals(state, {**state, **env_next}, parameters, t)

    return {**state, **env_next, **pop_next}
    
def aggregate(state):
    return {
        "Hi_any": len(state["Hi"]) > 0,
        "Ot_any": len(state["Ot"]) > 0,
        "In_any": len(state["In"]) > 0,
        "Im_any": len(state["Im"]) > 0,
        "Hi_sum": len(state["Hi"]),
        "Ot_sum": len(state["Ot"]),
        "In_sum": len(state["In"]),
        "Im_sum": len(state["Im"]),
    }

def count(state):
    return {
        "Hi": len(state["Hi"]),
        "Ot": len(state["Ot"]),
        "In": len(state["In"]),
        "Im": len(state["Im"]),
        "De": state["De"],
    }

def perturb(params, keys, scale=0.15):
    new = params.copy()
    for k in keys:
        val = params[k]
        if val > 0:
            new[k] = max(0, val + np.random.normal(0, scale * val))
    return new

def compute_metrics(history, Hi_list):
    
    N0 = 0 # total pop
    for i in range(len(Hi_list)):
        N0 += Hi_list[i][0] 

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

    P_max = P.max()            # peak prevalence
    T_Pmax = int(np.argmax(P)) # day of peak prevalence
    P_avg = P.mean()           # time-averaged prevalence

    # persistence: S(t) = N(t) / N(0)
    S = alive / N0
    S_final = S[-1]                         # final persistence

    # mortality burded: M(t) = De(t) / N(0)
    M = np.array(history["De"]) / N0
    M_final = M[-1]
    M_max = M.max()
    death_days = np.where(np.array(history["De"]) > 0)[0]
    T_De = int(death_days[0]) if len(death_days) > 0 else np.nan    # first death

    # disease invasion rate: I_new(t) = In(t+1) - In(t)
    In_arr = np.array(history["In"])
    I_new = np.diff(In_arr, prepend=In_arr[0])                      # new infections per day
    with np.errstate(invalid='ignore', divide='ignore'):
        I_rate = np.where(alive > 0, I_new / alive, 0.0)            # per-capita rate

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

# ------------------------------
# plotting at the end of the run
# ------------------------------

def plot_history(history, sample=[]):
    t = range(len(history["Hi"]))

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        constrained_layout=True,
        figsize = (14,7)
    )

    # -----------------
    # individual counts
    # -----------------

    ax1.plot(t, history["Hi"], label="Hibernating (Hi)")
    ax1.plot(t, history["Ot"], label="Non-hibernating, non-infected, non-immune (Ot)")
    ax1.plot(t, history["In"], label="Infected (In)")
    ax1.plot(t, history["Im"], label="Immune (Im)")
    ax1.plot(t, history["De"], label="Deceased (De)")

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Population count")
    ax1.set_title("Bat Population Dynamics")
    ax1.legend()
    ax1.grid()

    # ------------
    # total counts
    # ------------

    total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])

    ax2.plot(t, total, label="Total tricolored bats")

    # if there's sample data, compare:
    if len(sample) != 0:
        obs_times = sample[0]; obs_Ot = sample[1]
        ax2.scatter(obs_times, obs_Ot, label="Observed total tricolored bats")

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Population count")
    ax2.set_title("Bat Population Dynamics")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.grid(axis='x')
    plt.savefig('figures/history_plot.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def plot_history_highlights(history, win_length, win_start, T_seasonal, sample=[], xlim_max=None):
    t = range(len(history["Hi"]))
    n_days = len(t)

    # default cutoff: last observed time point, capped by how long the sim actually ran
    if xlim_max is None:
        if len(sample) != 0:
            xlim_max = min(max(sample[0]), n_days - 1)
        else:
            xlim_max = n_days - 1

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        constrained_layout=True,
        figsize=(14, 7)
    )

    ax1.plot(t, history["Hi"], label="Hibernating (Hi)")
    ax1.plot(t, history["Ot"], label="Non-hibernating, non-infected, non-immune (Ot)")
    ax1.plot(t, history["In"], label="Infected (In)")
    ax1.plot(t, history["Im"], label="Immune (Im)")
    ax1.plot(t, history["De"], label="Deceased (De)")

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Population count")
    ax1.set_title("Bat Population Dynamics")
    ax1.legend()
    ax1.grid()

    total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])
    ax2.plot(t, total, label="Total tricolored bats")

    if len(sample) != 0:
        obs_times = sample[0]; obs_Ot = sample[1]
        ax2.scatter(obs_times, obs_Ot, label="Observed total tricolored bats")

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Population count")
    ax2.set_title("Bat Population Dynamics")
    ax2.legend()
    ax2.grid()

    # highlight winter, using xlim_max as the cutoff instead of the full sim length
    days_per_year = 365
    cutoff = xlim_max + 1
    n_years = int(np.ceil(cutoff / days_per_year))

    win_length_padded = win_length + T_seasonal

    highlighter(n_years, cutoff, days_per_year, win_start, win_length_padded, ax1, ax2, 0.2)

    ax1.set_xlim(0, xlim_max)
    ax2.set_xlim(0, xlim_max)

    plt.tight_layout()
    plt.grid(axis='x')
    plt.savefig('figures/history_plot_highlighted.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def plot_error(history, win_length, win_start, T_seasonal, sample=[], xlim_max=None):
    t = range(len(history["Hi"]))
    n_days = len(t)

    if xlim_max is None:
        if len(sample) != 0:
            xlim_max = min(max(sample[0]), n_days - 1)
        else:
            xlim_max = n_days - 1

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        constrained_layout=True,
        figsize=(14, 7)
    )

    ax1.plot(t, history["Hi"], label="Hibernating (Hi)")
    ax1.plot(t, history["Ot"], label="Non-hibernating, non-infected, non-immune (Ot)")
    # ax1.plot(t, history["In"], label="Infected (In)")
    # ax1.plot(t, history["Im"], label="Immune (Im)")
    # ax1.plot(t, history["De"], label="Deceased (De)")

    total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])
    ax1.plot(t, total, label="Total tricolored bats", color='black', linewidth=2)

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Population count")
    ax1.set_title("Bat Population Dynamics")
    ax1.legend()
    ax1.grid()

    if len(sample) != 0:
        obs_times, obs_Hi = sample[0], sample[1]

        fitted_Hi = np.array(history["Hi"])
        diff = [obs_Hi[i] - fitted_Hi[t_val] for i, t_val in enumerate(obs_times)]

        ax2.plot(obs_times, diff, marker='o', linestyle='-', label="Observed − Fitted (Hi)")
        ax2.axhline(0, color='black', linewidth=1, linestyle='--')

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Observed − Fitted (Hi)")
    ax2.set_title("Residual Error in Hibernating Population (Hi)")
    ax2.legend()
    ax2.grid()

    ax1.set_xlim(0, xlim_max)
    ax2.set_xlim(0, xlim_max)

    plt.tight_layout()
    plt.grid(axis='x')
    plt.savefig('figures/history_error_plot.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def highlighter(n_years, cutoff, days_per_year, win_start, win_length, ax1, ax2, alpha):
    # cutoff = the day AFTER the last day you want plotted (i.e. xlim_max + 1)

    for year in range(-1, n_years):
        year_start = year * days_per_year

        start = year_start + win_start
        end = start + win_length

        year_end = year_start + days_per_year

        # skip this year's span entirely if it starts beyond the cutoff
        if start >= cutoff:
            continue

        # wraps before Jan 1
        if start < 0:
            ax1.axvspan(0, min(end, cutoff), alpha=alpha)
            ax2.axvspan(0, min(end, cutoff), alpha=alpha)

        # wraps after Dec 31
        elif end > year_end:
            ax1.axvspan(start, min(year_end, cutoff), alpha=alpha)
            ax2.axvspan(start, min(year_end, cutoff), alpha=alpha)

            wrap_end = end - year_end
            next_year_start = year_end

            if next_year_start < cutoff:
                ax1.axvspan(next_year_start, min(next_year_start + wrap_end, cutoff), alpha=alpha)
                ax2.axvspan(next_year_start, min(next_year_start + wrap_end, cutoff), alpha=alpha)

        # no wrapping
        else:
            ax1.axvspan(start, min(end, cutoff), alpha=alpha)
            ax2.axvspan(start, min(end, cutoff), alpha=alpha)