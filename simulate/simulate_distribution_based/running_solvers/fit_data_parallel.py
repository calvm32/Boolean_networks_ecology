import random as rand
import numpy as np
import copy
from mpi4py import MPI

from simulate.simulate_distribution_based.helper_funcs import *
from simulate.simulate_distribution_based.rules import *
from simulate.simulate_distribution_based.simulate import *
from simulate.data import *

# --------------------
# set up control group
# --------------------

data = happy_jack_data()

START_YEAR = data[0]["year"]
SAMPLE_DAY = data[0]["day"]

obs_times = []
obs_Hi = []
obs_In = []

for d in data:
    t = d["day"] + 365*(d["year"]-START_YEAR)
    
    obs_times.append(t)
    obs_Hi.append(d["Tri_Hi"] + d["Misc_Hi"])
    obs_In.append(d["In"])

def sample_params():                                                                                                                                     
    return {                                                                                                                                             
        "inf_alpha": inf_alpha,                                                                                               
        "inf_beta": inf_beta,                                               
        "delta": delta,                                                     
        "T_inf": T_inf,                                                     
        "T_TBD": T_TBD,                                                     
        "T_AD": T_AD,                                                       
        "T_seasonal": rand.uniform(20,60),                                                                                                               
        "win_length": rand.uniform(80,180),                                                                                                              
        "win_start": rand.uniform(230,280),                                                                                                              
        "lambda_win": lambda_win,                                           
        "lambda_sum": rand.uniform(0.00015, 0.00025),                                                                                                    
        "res_gain": res_gain,                                                                                                              
        "res_max": res_max,                                                                                                    
        "k_imm": k_imm,   
        "theta_imm": theta_imm,                                                                                             
    }    

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

# -------------------------
# set up initial population
# -------------------------

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

fraction_infected = 0   # choose in [0, 1]

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
T_TBD = 4.1                                 # CONFIDENT # length of torpor bout in days, 
                                            # considered in [3.9, 4.3] for tricolored bats
T_AD = 88.5/1440                            # CONFIDENT # length of arousal bout in days, 
                                            # considered in [1.74166, 5.63333] for tricolored bats
T_seasonal = 40                             # CONFIDENT # approx. transition time in days between hibernating and not
                                            # considered in 10-40 maybe?
win_length = 95                             # CONFIDENT # length of winter season in days in Nebraska mines
                                            # considered in 5-7 months, depending on transition period T_seasonal
win_start = 297                             # CONFIDENT # approximate day in calendar year that Te : 1 -> 0

# BAT IN/OUT FLUX
lambda_win = 0                              # CONFIDENT # population growth value during winter, 
                                            # considered in [0, 0.01] 
lambda_sum = 0.00013942579094               # CONFIDENT # population growth value during summer,
                                            # considered in [0.01, 0.1] 

# -----------------
# types of immunity
# -----------------

res_max = 0.2                               # hereditary resistance of newborn, corresp. w/ rand.normalvariate(0, X)
k_imm, theta_imm = 1, 1                     # number of days spent in recovery before re-infection is possible
                                            # corresp. w/ Gamma(k_imm, theta_imm)
res_gain = 0.02                             # resistance AFTER recovery

# ----------
# initialize
# ----------

time = 3650             # total days

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================
    
def loss(parameters, runs=2):
    losses = []

    for _ in range(runs):
        sim = simulate(make_initial_state(Hi_list, fraction_infected), steps=max(obs_times)+1, parameters=parameters, Print=False)

        error = 0.0
        for i, t in enumerate(obs_times):
            pred_Hi = sim["Hi"][t]
            #pred_In = sim["In"][t]

            error += (pred_Hi - obs_Hi[i])**2
            #error += (pred_In - obs_In[i])**2

        losses.append(error / len(obs_times))

    return np.mean(losses)

# ----------
# run things
# ----------

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parameters = sample_params()

    best = None
    best_loss = float("inf")
    local_best = None
    local_best_loss = float("inf")

    n_iter = 10000
    local_iters = n_iter // size

    for i in range(local_iters):
        params = sample_params()
        L = abs(loss(params))

        if L < best_loss:
            best_loss = L
            best = params
            print("New best:", best_loss, best)
        
        if L < local_best_loss:
            print(f"checked {i}")
            local_best_loss = L
            local_best = params

        #print(f"[rank {rank}] iteration {i}, loss={L}")

    # gather results
    all_results = comm.gather((local_best_loss, local_best), root=0)

    if rank == 0:
        best_loss = float("inf")
        best_params = None

        for L, p in all_results:
            if L < best_loss:
                best_loss = L
                best_params = p

        print("\nGLOBAL BEST:")
        print(best_loss, best_params)

    best_sim = simulate(make_initial_state(Hi_list, fraction_infected), steps = 4500, parameters=best, Print=False)
    plot_history_highlights(best_sim, win_length, win_start, T_seasonal, sample=[obs_times, obs_Hi])

if __name__ == "__main__":
    main()

"""

New best: 379.54166666666663    {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 53.03145102211801, 'win_length': 171.8073545805218, 'win_start': 278.92671603576855, 'lambda_win': 0, 'lambda_sum': 0.00017711957391318124, 'res_gain': 0.02, 'res_max': 0.2, 'k_imm': 1, 'theta_imm': 1}
New best: 146.33333333333334    {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 41.21787049830048, 'win_length': 190.3327411105248, 'win_start': 269.08553986430616, 'lambda_win': 0, 'lambda_sum': 0.00015441617617432672, 'res_gain': 0.02, 'res_max': 0.2, 'k_imm': 1, 'theta_imm': 1}
New best: 132.375               {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 43.460387151848266, 'win_length': 197.57926135060328, 'win_start': 261.508501689169, 'lambda_win': 0, 'lambda_sum': 0.0001676431683071884, 'res_gain': 0.02, 'res_max': 0.2, 'k_imm': 1, 'theta_imm': 1}
New best: 127.54166             {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'lambda_win': 0, 'res_gain': 0.02, 'res_max': 0.2, 'k_imm': 1, 'theta_imm': 1, 'T_seasonal': np.float64(47.40178558756894), 'win_length': np.float64(170.03850624815655), 'win_start': np.float64(287.8314551381096), 'lambda_sum': np.float64(0.0001605245540170072)}

"""
