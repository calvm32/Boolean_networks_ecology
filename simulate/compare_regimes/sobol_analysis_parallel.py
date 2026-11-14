import os

# One thread per worker process: avoids BLAS oversubscription when running many processes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")  # headless cluster nodes

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
import pandas as pd
import random as rand
import numpy as np

from simulate.helper_funcs import *
from simulate.rules import *
from simulate.simulate import *
from simulate.working_params import *

# publication-grade theme style
from simulate.style_config import *

# re-import after the star-imports so nothing above can shadow these
import os
import argparse
import math
import sys
from multiprocessing import Pool
import matplotlib.pyplot as plt

OUTPUT_DIR = os.environ.get('SIM_OUTPUT_DIR', '.')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'results_and_data')
CHUNK_DIR = os.path.join(FIGURES_DIR, 'sobol_chunks')
PARAM_FILE = os.path.join(FIGURES_DIR, 'sobol_param_values.npy')

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

avg_over = 10

NUM_SAMPLES = 2 ** int(os.environ.get('SOBOL_LOG2_N', 14))   # SOBOL_LOG2_N=6 gives a quick dry run
CHUNK_SIZE = 128        # parameter sets per chunk file (~ CHUNK_SIZE*avg_over simulations)
BASE_SEED = 12345       # change to get a different (but still reproducible) set of stochastic runs

T_DAYS = 365 * 20       # total days simulated

# order must match PROBLEM["names"] / "bounds"
PARAM_KEYS = ["inf_alpha", "inf_beta", "delta", "T_inf", "res_max", "res_gain", "k_imm", "theta_imm"]

PROBLEM = {
    "num_vars": 8,
    # Formatted cleanly to match LaTeX parameter standards
    "names": [
        r"$\alpha_{\mathrm{inf}}$", r"$\beta_{\mathrm{inf}}$", r"$\delta$",
        r"$T_{\mathrm{inf}}$", r"$r_{\mathrm{max}}$", r"$r_{\mathrm{gain}}$", r"$k_{\mathrm{imm}}$",
        r"$\theta_{\mathrm{imm}}$"
    ],
    "bounds": [
        [1, 5],
        [2, 10],
        [0.005, 0.05],
        [10, 40],
        [0, 0.8],
        [0, 1],
        [0.1, 10],
        [0, 30],
    ],
}

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

# -------------------------
# set up initial population
# -------------------------

num_infected = 5

# first select number of bats belonging to each species
tricolor_num = 100
tricolor_cluster_sizeMIN = 1
tricolor_cluster_sizeMAX = 2

bigbrown_num = 0
bigbrown_cluster_sizeMIN = 1
bigbrown_cluster_sizeMAX = 9

# hibernating non-infected bats of each species
Hi_list = [[tricolor_num, tricolor_cluster_sizeMIN, tricolor_cluster_sizeMAX],
           [bigbrown_num, bigbrown_cluster_sizeMIN, bigbrown_cluster_sizeMAX]]

# NOTICE : the remaining populations (Ot, Im) all start with 0 inhabitants
# NOTICE : resistance starts at 0 for every bat

# ---------------------------
# system-governing parameters
# ---------------------------

# INFECTION PATHWAYS
inf_alpha, inf_beta = 5, 2                  # infected variables for beta distribution
                                            # chance a hibernating bat gets infected (given that PD is on) on any given day
                                            # low: alpha = 1, beta = 10
                                            # moderate: alpha = 2, beta = 5
                                            # high: alpha = 5, beta = 2

delta = 0.05                                # P. destructans decay rate, considered in [0.005, 0.03]

# DEATH OR RECOVERY PATHWAYS
T_inf = 30                                  # approximate time in dayseach bat spends infirm before recovering or dying,
                                            # considered in [10, 40]

# BOUT and SEASONAL HIBERNATING PATHWAYS
T_TBD = 4.1                                 # CONSTANT FOR TRICOLORED # length of torpor bout in days,
                                            # considered in [3.9, 4.3] for tricolored bats
T_AD = 88.5/1440                            # CONSTANT FOR TRICOLORED # length of arousal bout in days,
                                            # considered in [1.74166, 5.63333] for tricolored bats

# -----------------
# types of immunity
# -----------------

res_max = 0.2                               # hereditary resistance of newborn, corresp. w/ rand.normalvariate(0, X)
k_imm, theta_imm = 1, 1                     # number of days spent in recovery before re-infection is possible
                                            # corresp. w/ Gamma(k_imm, theta_imm)
res_gain = 0.02                             # resistance AFTER recovery

# ---------------------------------
# latitudinally-averaged parameters
# ---------------------------------

T_seasonal, win_length, win_start, lambda_sum, lambda_win = latitude1_NorthMidwest() # or latitude2_SouthMidwest()
NAME_APPEND = "lat1NMW"


def sample_params():
    return {
        "inf_alpha": inf_alpha,
        "inf_beta": inf_beta,
        "delta": delta,
        "T_inf": T_inf,
        "T_TBD": T_TBD,
        "T_AD": T_AD,
        "T_seasonal": T_seasonal,
        "win_length": win_length,
        "win_start": win_start,
        "lambda_win": lambda_win,
        "lambda_sum": lambda_sum,
        "res_gain": res_gain,
        "res_max": res_max,
        "k_imm": k_imm,
        "theta_imm": theta_imm,
    }

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

"""
Parallel, resumable Sobol sensitivity analysis.

Workflow (all subcommands are safe to re-run):

    python sobol_parallel.py prepare            # once: generate + save the Sobol sample matrix
    python sobol_parallel.py run --workers 64   # one big node, OR
    sbatch run_sobol.sbatch                     # SLURM array (each task calls `run`)
    python sobol_parallel.py status             # how many chunks are done
    python sobol_parallel.py analyze            # once all chunks exist: indices, CSV, figures

The N*(D+2) parameter sets are cut into chunks of CHUNK_SIZE. Each chunk is
written to its own .npz file when finished, so a killed/timed-out job only loses
the chunks it was in the middle of, and re-running skips everything already done.
Every simulation gets its own deterministic seed, so results do not depend on
how many workers/tasks you use.
"""

COMPARTMENTS = ["Hi", "Ot", "In", "Im", "De"]

_PARAM_VALUES = None   # set in each worker by _init_worker


def evaluate(row, i):
    """Run avg_over stochastic replicates for sample row `i`; return [P_max, S_final, M_final, R0]."""
    parameters = sample_params()                       # fresh dict every sample
    for key, val in zip(PARAM_KEYS, row):
        parameters[key] = max(1.0, float(val)) if key == "inf_alpha" else float(val)

    history_avg = {k: np.zeros(T_DAYS) for k in COMPARTMENTS}   # fresh arrays every sample
    history = None
    for j in range(avg_over):
        seed = BASE_SEED + i * avg_over + j            # unique + reproducible regardless of worker layout
        rand.seed(seed)
        np.random.seed(seed)
        history = simulate(make_initial_state(Hi_list, num_infected), T_DAYS, parameters, False)
        for key in history_avg:
            history_avg[key] += np.asarray(history[key])

    for key in history_avg:
        history_avg[key] /= avg_over

    history_avg["SC"] = history["SC"]

    m = compute_metrics(history_avg, Hi_list, num_infected)
    return [m["P_max"], m["S_final"], m["M_final"], m["R0_empirical"]]


def chunk_path(chunk_id):
    return os.path.join(CHUNK_DIR, f"chunk_{chunk_id:05d}.npz")


def _init_worker(param_values):
    global _PARAM_VALUES
    _PARAM_VALUES = param_values


def run_chunk(chunk_id):
    out = chunk_path(chunk_id)
    if os.path.exists(out):
        return chunk_id
    lo = chunk_id * CHUNK_SIZE
    hi = min(lo + CHUNK_SIZE, len(_PARAM_VALUES))
    Y = np.zeros((hi - lo, 4))
    for i in range(lo, hi):
        Y[i - lo] = evaluate(_PARAM_VALUES[i], i)
    tmp = out[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, Y=Y, lo=lo, hi=hi)
    os.replace(tmp, out)                               # atomic: a chunk file is either complete or absent
    return chunk_id


def n_chunks_for(n):
    return math.ceil(n / CHUNK_SIZE)


def default_workers():
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        return int(env)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ==========================================================================================================================
# subcommands
# ==========================================================================================================================

def cmd_prepare(args):
    os.makedirs(CHUNK_DIR, exist_ok=True)
    if os.path.exists(PARAM_FILE) and not args.force:
        pv = np.load(PARAM_FILE)
        print(f"{PARAM_FILE} already exists ({len(pv)} rows); keeping it. Use --force to regenerate "
              f"(this also invalidates existing chunks).")
        return pv
    pv = sobol_sample.sample(PROBLEM, NUM_SAMPLES, calc_second_order=False)
    tmp = PARAM_FILE + ".tmp.npy"
    np.save(tmp, pv)
    os.replace(tmp, PARAM_FILE)
    print(f"Saved {len(pv)} parameter sets ({n_chunks_for(len(pv))} chunks of {CHUNK_SIZE}) to {PARAM_FILE}")
    return pv


def cmd_run(args):
    if not os.path.exists(PARAM_FILE):
        if args.n_tasks > 1:
            sys.exit("No sample file yet. Run `python sobol_parallel.py prepare` once before launching the array.")
        cmd_prepare(argparse.Namespace(force=False))
    pv = np.load(PARAM_FILE)
    n_chunks = n_chunks_for(len(pv))
    mine = [c for c in range(n_chunks) if c % args.n_tasks == args.task_id]
    todo = [c for c in mine if not os.path.exists(chunk_path(c))]
    os.makedirs(CHUNK_DIR, exist_ok=True)

    workers = max(1, min(args.workers, len(todo))) if todo else 1
    print(f"task {args.task_id}/{args.n_tasks}: {len(todo)} chunks to do "
          f"({len(mine) - len(todo)} already done), {workers} workers", flush=True)
    if not todo:
        return

    with Pool(workers, initializer=_init_worker, initargs=(pv,)) as pool:
        for k, cid in enumerate(pool.imap_unordered(run_chunk, todo), 1):
            print(f"  [{k}/{len(todo)}] chunk {cid} done", flush=True)


def cmd_status(args):
    pv = np.load(PARAM_FILE)
    n_chunks = n_chunks_for(len(pv))
    done = sum(os.path.exists(chunk_path(c)) for c in range(n_chunks))
    print(f"{done}/{n_chunks} chunks complete ({100 * done / n_chunks:.1f}%)")


def cmd_analyze(args):
    apply_publication_style()
    pv = np.load(PARAM_FILE)
    n = len(pv)
    Y = np.full((n, 4), np.nan)
    missing = []
    for c in range(n_chunks_for(n)):
        p = chunk_path(c)
        if not os.path.exists(p):
            missing.append(c)
            continue
        d = np.load(p)
        Y[int(d["lo"]):int(d["hi"])] = d["Y"]
    if missing:
        sys.exit(f"{len(missing)} chunks still missing (e.g. {missing[:10]}). Re-run `run`/sbatch, then analyze.")
    if np.isnan(Y).any():
        print(f"WARNING: {int(np.isnan(Y).sum())} NaN outputs; Sobol indices will be NaN for those metrics.")

    problem = PROBLEM
    Y_Pmax, Y_Sfinal, Y_Mfinal, Y_R0 = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]

    Si_P = sobol.analyze(problem, Y_Pmax, calc_second_order=False, print_to_console=False)
    Si_S = sobol.analyze(problem, Y_Sfinal, calc_second_order=False, print_to_console=False)
    Si_M = sobol.analyze(problem, Y_Mfinal, calc_second_order=False, print_to_console=False)
    Si_R0 = sobol.analyze(problem, Y_R0, calc_second_order=False, print_to_console=False)

    sobol_data = {
        "Parameter": [nm.replace('$', '').replace('\\', '') for nm in problem["names"]],
        "R0_S1": Si_R0["S1"], "R0_S1_conf": Si_R0["S1_conf"],
        "R0_ST": Si_R0["ST"], "R0_ST_conf": Si_R0["ST_conf"],
        "Pmax_S1": Si_P["S1"], "Pmax_S1_conf": Si_P["S1_conf"],
        "Pmax_ST": Si_P["ST"], "Pmax_ST_conf": Si_P["ST_conf"],
        "Sfinal_S1": Si_S["S1"], "Sfinal_S1_conf": Si_S["S1_conf"],
        "Sfinal_ST": Si_S["ST"], "Sfinal_ST_conf": Si_S["ST_conf"],
        "Mfinal_S1": Si_M["S1"], "Mfinal_S1_conf": Si_M["S1_conf"],
        "Mfinal_ST": Si_M["ST"], "Mfinal_ST_conf": Si_M["ST_conf"],
    }

    df_sobol = pd.DataFrame(sobol_data)
    csv_path = os.path.join(FIGURES_DIR, "sobol_analysis_data.csv")
    df_sobol.to_csv(csv_path, index=False)

    # raw model outputs too, so you can re-analyze without re-simulating
    pd.DataFrame(Y, columns=["P_max", "S_final", "M_final", "R0_empirical"]).to_csv(
        os.path.join(FIGURES_DIR, "sobol_raw_outputs.csv"), index=False)

    def plot_single_sobol(Si, problem, title, filename):
        names = problem["names"]
        x = np.arange(len(names))
        width = 0.38

        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        ax.bar(x - width/2, Si["S1"], width, label=r"First-order ($S_1$)",
               color=COLORS["Sobol_S1"], yerr=Si["S1_conf"], capsize=3, error_kw={"lw": 0.8, "color": "#333333"})
        ax.bar(x + width/2, Si["ST"], width, label=r"Total-order ($S_T$)",
               color=COLORS["Sobol_ST"], yerr=Si["ST_conf"], capsize=3, error_kw={"lw": 0.8, "color": "#333333"})

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Sobol Sensitivity Index")
        ax.set_title(title, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.5)

        out_path = os.path.join(FIGURES_DIR, filename)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    plot_single_sobol(Si_R0, problem, r"Sensitivity: Empirical $R_0$", f'sobol_R0_{NAME_APPEND}.pdf')
    plot_single_sobol(Si_P, problem, r"Sensitivity: Peak Prevalence ($P_{\max}$)", f'sobol_Pmax_{NAME_APPEND}.pdf')
    plot_single_sobol(Si_S, problem, r"Sensitivity: Final Persistence ($S_{\text{final}}$)", f'sobol_Sfinal_{NAME_APPEND}.pdf')
    plot_single_sobol(Si_M, problem, r"Sensitivity: Final Mortality ($M_{\text{final}}$)", f'sobol_Mfinal_{NAME_APPEND}.pdf')
    print(f"Wrote CSVs and figures to {FIGURES_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="generate and save the Sobol sample matrix (run once)")
    p.add_argument("--force", action="store_true", help="regenerate even if it exists")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("run", help="simulate chunks in parallel (resumable)")
    p.add_argument("--workers", type=int, default=default_workers(),
                   help="processes to use (default: SLURM_CPUS_PER_TASK or all available cores)")
    p.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
                   help="this task's index (default: SLURM_ARRAY_TASK_ID or 0)")
    p.add_argument("--n-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)),
                   help="total number of tasks splitting the work (default: SLURM_ARRAY_TASK_COUNT or 1)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("status", help="show how many chunks are finished")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("analyze", help="combine chunks, compute Sobol indices, write CSV + figures")
    p.set_defaults(fn=cmd_analyze)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()