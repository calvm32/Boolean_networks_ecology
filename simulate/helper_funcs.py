import matplotlib.pyplot as plt
import numpy as np
from simulate.rules import *

import os

OUTPUT_DIR = os.environ.get('SIM_OUTPUT_DIR', '.')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'results_and_data')

# publication-grade theme style
from simulate.style_config import *
apply_publication_style()

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

# ---------------------
# setup before each run
# ---------------------

def make_initial_state(Hi_list, num_infected):
    # NOTICE : each inhabitant node contains the following information:
    # [ ON/OFF, 
    #   resistance number AKA res_num, 
    #   clustering number AKA mu_i, 
    #   days left infirm, 
    #   0 for just entered hibernation OR 1 for exited hibernation at least once (to track arousal periods),
    #   days left immune,
    #   1 if that bat was an "index case", i.e. first infected bats introduced. used for calculating R0
    # ]
    
    empty_pop = []

    return{
        "Hi": [
                [1, 0, rand.uniform(Hi_list[i][1], Hi_list[i][2]), 0, 0, 0, 0]
                for i in range(len(Hi_list))
                for _ in range(Hi_list[i][0] - num_infected)
              ],
        "Ot": empty_pop.copy(),
        "In": [
                [1, 0, rand.uniform(Hi_list[i][1], Hi_list[i][2]), 0, 0, 0, 1]
                for i in range(len(Hi_list))
                for _ in range(num_infected)
              ],
        "Im": empty_pop.copy(),
        "De": 0, # only need total numbers of dead
        "Re": 1,
        "Te": 0,
        "Hu": 0,
        "PD": 0,
        "SC": 0, # SECONDARY cases, used to calculate R0
    }

# -------------------------------------------------------
# metrics used for measuring data, e.g. in Sobol analysis
# -------------------------------------------------------

def compute_metrics(history, Hi_list, num_infected):
    
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

    # ----------------------------------
    # 1. prevalence: P(t) = In(t) / N(t)
    # ----------------------------------
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.where(alive > 0, np.array(history["In"]) / alive, 0.0)

    P_max = P.max()             # peak prevalence
    T_Pmax = int(np.argmax(P))  # day of peak prevalence
    P_avg = P.mean()            # time-averaged prevalence

    # ----------------------------------
    # 2. persistence: S(t) = N(t) / N(0)
    # ----------------------------------
    S = alive / N0
    S_final = S[-1]             # final persistence

    # ----------------------------------------
    # 3. mortality burded: M(t) = De(t) / N(0)
    # ----------------------------------------
    M = np.array(history["De"]) / N0
    M_final = M[-1]
    M_max = M.max()
    death_days = np.where(np.array(history["De"]) > 0)[0]
    T_De = int(death_days[0]) if len(death_days) > 0 else np.nan    # first death

    # ----------------------------------------------------
    # 4. disease invasion rate: I_new(t) = In(t+1) - In(t)
    # ----------------------------------------------------
    In_arr = np.array(history["In"])
    I_new = np.diff(In_arr, prepend=In_arr[0])                      # new infections per day
    with np.errstate(invalid='ignore', divide='ignore'):
        I_rate = np.where(alive > 0, I_new / alive, 0.0)            # per-capita rate

    # -------------------------------------------------------
    # 5. R0 reproduction number: secondary inf. by single bat
    # -------------------------------------------------------
    exact_R0 = history.get("SC", 0) / num_infected if num_infected > 0 else 0
    
    return {
        "N_min": N_min,
        "N_final": N_final,
        "P_max": P_max,
        "T_Pmax": T_Pmax,
        "P_avg": P_avg,
        "S_final": S_final,
        "M_final": M_final,
        "M_max": M_max,
        "T_De": T_De,
        "R0_empirical": exact_R0,

        # full time-series (needed for Monte Carlo bands)
        "_P": P,
        "_S": S,
        "_M": M,
        "_I_rate": I_rate,
        "_alive": alive,
    }

# ------------------------------
# plotting at the end of the run
# ------------------------------

def plot_history(history, sample=[]):
    t = range(len(history["Hi"]))

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        constrained_layout=True,
        figsize=(14, 6)
    )

    ax1.plot(t, history["Hi"], label=r"Hibernating (Hi)", color=COLORS["Hi"], linewidth=1.8)
    ax1.plot(t, history["Ot"], label=r"Susceptible (Ot)", color=COLORS["Ot"], linewidth=1.8)
    ax1.plot(t, history["In"], label=r"Infected (In)", color=COLORS["In"], linewidth=1.8)
    ax1.plot(t, history["Im"], label=r"Immune (Im)", color=COLORS["Im"], linewidth=1.8)
    ax1.plot(t, history["De"], label=r"Deceased (De)", color=COLORS["De"], linewidth=1.5, linestyle=':')

    ax1.set_xlabel("Time Step (Days)")
    ax1.set_ylabel("Population Count")
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_title("Epidemiological Cohort Dynamics", fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax1.grid(True)

    total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])
    ax2.plot(t, total, label="Simulated Total", color=COLORS["Total"], linewidth=2.2)

    if len(sample) != 0:
        obs_times = sample[0]
        obs_Ot = sample[1]
        ax2.scatter(obs_times, obs_Ot, label="Observed Field Data", color=COLORS["In"], s=30, zorder=5, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel("Time Step (Days)")
    ax2.set_ylabel("Population Count")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_title("Total Population Trajectory", fontweight='bold')
    ax2.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax2.grid(True)

    plt.savefig(os.path.join(FIGURES_DIR, 'history_plot.pdf'), bbox_inches='tight')
    plt.show()


def plot_history_highlights(history, win_length, win_start, T_seasonal, sample=[], xlim_max=None):
    t = range(len(history["Hi"]))
    n_days = len(t)

    if xlim_max is None:
        xlim_max = min(max(sample[0]), n_days - 1) if len(sample) != 0 else n_days - 1

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        constrained_layout=True,
        figsize=(14, 6)
    )

    ax1.plot(t, history["Hi"], label=r"Hibernating (Hi)", color=COLORS["Hi"], linewidth=1.8)
    ax1.plot(t, history["Ot"], label=r"Susceptible (Ot)", color=COLORS["Ot"], linewidth=1.8)
    ax1.plot(t, history["In"], label=r"Infected (In)", color=COLORS["In"], linewidth=1.8)
    ax1.plot(t, history["Im"], label=r"Immune (Im)", color=COLORS["Im"], linewidth=1.8)
    ax1.plot(t, history["De"], label=r"Deceased (De)", color=COLORS["De"], linewidth=1.5, linestyle=':')

    ax1.set_xlabel("Time Step (Days)")
    ax1.set_ylabel("Population Count")
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_title("Cohort Dynamics (Winter Highlighted)", fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax1.grid(True)

    total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])
    ax2.plot(t, total, label="Simulated Total", color=COLORS["Total"], linewidth=2.2)

    if len(sample) != 0:
        obs_times = sample[0]
        obs_Ot = sample[1]
        ax2.scatter(obs_times, obs_Ot, label="Observed Field Data", color=COLORS["In"], s=30, zorder=5, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel("Time Step (Days)")
    ax2.set_ylabel("Population Count")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_title("Total Population (Winter Highlighted)", fontweight='bold')
    ax2.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax2.grid(True)

    days_per_year = 365
    cutoff = xlim_max + 1
    n_years = int(np.ceil(cutoff / days_per_year))
    win_length_padded = win_length + T_seasonal

    highlighter(n_years, cutoff, days_per_year, win_start, win_length_padded, ax1, ax2, 0.35)

    ax1.set_xlim(0, xlim_max)
    ax2.set_xlim(0, xlim_max)

    plt.savefig(os.path.join(FIGURES_DIR, 'history_plot_highlighted.pdf'), bbox_inches='tight')
    plt.show()


def plot_residual_error(history, win_length, win_start, T_seasonal, sample=[], xlim_max=None):
    t = range(len(history["Hi"]))
    n_days = len(t)

    if xlim_max is None:
        xlim_max = min(max(sample[0]), n_days - 1) if len(sample) != 0 else n_days - 1

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        constrained_layout=True,
        figsize=(14, 6)
    )

    ax1.plot(t, history["Hi"], label=r"Hibernating (Hi)", color=COLORS["Hi"], linewidth=1.8)
    ax1.plot(t, history["Ot"], label=r"Susceptible (Ot)", color=COLORS["Ot"], linewidth=1.8)
    total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])
    ax1.plot(t, total, label="Total Population", color=COLORS["Total"], linewidth=2.2)

    ax1.set_xlabel("Time Step (Days)")
    ax1.set_ylabel("Population Count")
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_title("Reference Population Trajectories", fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax1.grid(True)

    if len(sample) != 0:
        obs_times, obs_Hi = sample[0], sample[1]
        fitted_Hi = np.array(history["Hi"])
        diff = [obs_Hi[i] - fitted_Hi[t_val] for i, t_val in enumerate(obs_times)]

        ax2.plot(obs_times, diff, marker='o', color=COLORS["In"], markersize=4, linestyle='-', label="Residual (Observed - Simulated)")
        ax2.axhline(0, color='black', linewidth=1, linestyle='--')

    ax2.set_xlabel("Time Step (Days)")
    ax2.set_ylabel(r"Residual Error (Hi count)")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_title("Model Residuals in Hibernating Cohort", fontweight='bold')
    ax2.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax2.grid(True)

    days_per_year = 365
    cutoff = xlim_max + 1
    n_years = int(np.ceil(cutoff / days_per_year))
    win_length_padded = win_length + T_seasonal

    highlighter(n_years, cutoff, days_per_year, win_start, win_length_padded, ax1, ax2, 0.35)

    ax1.set_xlim(0, xlim_max)
    ax2.set_xlim(0, xlim_max)

    plt.savefig(os.path.join(FIGURES_DIR, 'history_error_plot.pdf'), bbox_inches='tight')
    plt.show()


def plot_wMAPE(history, win_length, win_start, T_seasonal, sample=[], xlim_max=None):
    plot_residual_error(history, win_length, win_start, T_seasonal, sample=sample, xlim_max=xlim_max)


def highlighter(n_years, cutoff, days_per_year, win_start, win_length, ax1, ax2, alpha):
    for year in range(-1, n_years):
        year_start = year * days_per_year
        start = year_start + win_start
        end = start + win_length
        year_end = year_start + days_per_year

        if start >= cutoff:
            continue

        if start < 0:
            ax1.axvspan(0, min(end, cutoff), color=WINTER_COLOR, alpha=alpha, ec=None)
            ax2.axvspan(0, min(end, cutoff), color=WINTER_COLOR, alpha=alpha, ec=None)
        elif end > year_end:
            ax1.axvspan(start, min(year_end, cutoff), color=WINTER_COLOR, alpha=alpha, ec=None)
            ax2.axvspan(start, min(year_end, cutoff), color=WINTER_COLOR, alpha=alpha, ec=None)
            wrap_end = end - year_end
            next_year_start = year_end
            if next_year_start < cutoff:
                ax1.axvspan(next_year_start, min(next_year_start + wrap_end, cutoff), color=WINTER_COLOR, alpha=alpha, ec=None)
                ax2.axvspan(next_year_start, min(next_year_start + wrap_end, cutoff), color=WINTER_COLOR, alpha=alpha, ec=None)
        else:
            ax1.axvspan(start, min(end, cutoff), color=WINTER_COLOR, alpha=alpha, ec=None)
            ax2.axvspan(start, min(end, cutoff), color=WINTER_COLOR, alpha=alpha, ec=None)