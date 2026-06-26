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
T_TBD = 4.1                                 # length of torpor bout in days, 
                                            # considered in [3.9, 4.3] for tricolored bats
T_AD = 88.5/1440                            # length of arousal bout in days, 
                                            # considered in [1.74166, 5.63333] for tricolored bats
T_seasonal = 35                             # approx. transition time in days between hibernating and not
                                            # considered in 10-40 maybe?
win_length = 95                             # length of winter season in days in Nebraska mines
                                            # considered in 5-7 months, depending on transition period T_seasonal
win_start = 264                             # approximate day in calendar year that Te : 1 -> 0

# BAT IN/OUT FLUX
lambda_win = 0                              # population growth value during winter, 
                                            # considered in [0, 0.01] 
lambda_sum = 0.00015317467856               # population growth value during summer,
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
        sim = simulate(make_initial_state(Hi_list, fraction_infected), steps=max(obs_times)+1, parameters=parameters)

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

def simulate(initial_state, steps, parameters):
    state = initial_state
    win_length = parameters["win_length"]

    history = {
        "Hi": np.empty(steps,dtype=np.int32),
        "Ot": np.empty(steps,dtype=np.int32),
        "In": np.empty(steps,dtype=np.int32),
        "Im": np.empty(steps,dtype=np.int32),
        "De": np.empty(steps,dtype=np.int32),
    }

    for t in range(steps):

        # Seasonal tempcycle
        if (t % 365) <= win_length: # win_length
            state["Te"] = 0   
        else:
            state["Te"] = 1 # summer
        counts = count(state)

        history["Hi"][t] = (counts["Hi"])
        history["Ot"][t] = (counts["Ot"])
        history["In"][t] = (counts["In"])
        history["Im"][t] = (counts["Im"])
        history["De"][t] = (counts["De"])

        state = step(state, parameters)

        # if t % 50 == 0:
        #     print(f"done w/ simulation at step {t}")

    return history

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

    best_sim = simulate(make_initial_state(Hi_list, fraction_infected), steps = 4500, parameters=best)
    plot_history_highlights(best_sim, win_length, win_start, sample=[obs_times, obs_Hi])

if __name__ == "__main__":
    main()

"""
New best: 408.29166666666663    {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 40, 'win_length': 104.55315805032492, 'lambda_win': 0, 'lambda_sum': 0.00030810579178920205, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}
New best: 294.625               {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 40, 'win_length': 93.66088582529426, 'lambda_win': 0, 'lambda_sum': 0.00016730065634549607, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}
New best: 290.41666666666663    {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 42.84137132196838, 'win_length': 92.9025546096547, 'lambda_win': 0,'lambda_sum': 0.00018596411797403993, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}          
New best: 253.83333333333331    {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 41.84816384106251, 'win_length': 97.62007352307558, 'lambda_win': 0, 'lambda_sum': 0.00020760238859780686, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02} 
New best: 245.20833333333331    {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 33.27959466619289, 'win_length': 103.45290874030258, 'lambda_win': 5.956138472403809e-06, 'lambda_sum': 0.00021796636177355251, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}   
New best: 1429.2916666666667    {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 40, 'win_length': 81.23002673731209, 'win_start': 255.9791250706794, 'lambda_win': 0, 'lambda_sum': 0.0004228730195226812, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}
New best: 932.3333333333334     {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 40, 'win_length': 80.00036540268336, 'win_start': 250.03157138705092, 'lambda_win': 0, 'lambda_sum': 0.00028885954892277863, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}
New best: 816.625               {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 20.114361051619568, 'win_length': 81.17831315823864, 'win_start': 251.1083278511041, 'lambda_win': 0, 'lambda_sum': 0.00019076783002288473, 'T_im': 0, 'birth_resistance_mcax': 0, 'recover_resistance_max': 0.02}
New best: 121.54166666666667    {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 24.457813421532066, 'win_length': 95.81641676014776, 'win_start': 235.0505687167425, 'lambda_win': 0, 'lambda_sum': 0.0001557314442035817, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}
New best: 137.08333333333334    {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 28.063329031264438, 'win_length': 96.98455618792036, 'win_start': 269.301568887714, 'lambda_win': 0, 'lambda_sum': 0.00015453903825809258, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}
New best: 116.08333333333334    {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 36.0488257140253, 'win_length': 92.28247844125634, 'win_start': 256.9756753221267, 'lambda_win': 0, 'lambda_sum': 0.00012091541782081852, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}
New best: 109.0                 {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 35.346254558328134, 'win_length': 94.61334026606372, 'win_start': 263.7285881550976, 'lambda_win': 0, 'lambda_sum': 0.00015806109432201025, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}
New best: 89.125                {'inf_alpha': 5, 'inf_beta': 2, 'delta': 0.05, 'T_inf': 30, 'T_TBD': 4.1, 'T_AD': 0.06145833333333333, 'T_seasonal': 35, 'win_length': 95, 'win_start': 264, 'lambda_win': 0, 'lambda_sum': 0.00015317467855989502, 'T_im': 0, 'res_max': 0, 'recover_resistance_max': 0.02}


"""
