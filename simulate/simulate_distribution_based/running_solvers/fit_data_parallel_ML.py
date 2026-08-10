import random as rand
import numpy as np
import copy
from mpi4py import MPI
from scipy.special import gammaln

from simulate.simulate_distribution_based.helper_funcs import *
from simulate.simulate_distribution_based.rules import *
from simulate.simulate_distribution_based.simulate import *
from simulate.data import *

"""
NOTICE: this code was generated using an LLM

Uses PSO (particle swarm optimization) hyperparameter searching
"""

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

num_infected = 0 # DO NOT CHANGE
for i in range(len(Hi_list)):
    num_infected += int(Hi_list[i][0]*fraction_infected) # DO NOT CHANGE

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
T_seasonal = 59                             # CONFIDENT # approx. transition time in days between hibernating and not
                                            # considered in 10-40 maybe?
win_length = 161                            # CONFIDENT # length of winter season in days in Nebraska mines
                                            # considered in 5-7 months, depending on transition period T_seasonal
win_start = 289                             # CONFIDENT # approximate day in calendar year that Te : 1 -> 0

# BAT IN/OUT FLUX
lambda_win = 0                              # CONFIDENT # population growth value during winter, 
                                            # considered in [0, 0.01] 
lambda_sum = 0.00028895065208267            # CONFIDENT # population growth value during summer,
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

disp_r = 10             # dispersion parameter # ONLY USED FOR DATA FITTING


# Static parameters that do not get optimized
FIXED_PARAMS = {
    "inf_alpha": inf_alpha,
    "inf_beta": inf_beta,
    "delta": delta,
    "T_inf": T_inf,
    "T_TBD": T_TBD,
    "T_AD": T_AD,
    "res_max": res_max,
    "k_imm": k_imm,
    "theta_imm": theta_imm,
    "res_gain": res_gain,
}

# Parameters to Optimize (Search Space)
BOUNDS = {
    "T_seasonal": (40.0, 80.0),
    "win_length": (120.0, 240.0),
    "win_start": (200.0, 350.0),
    "lambda_win": (0, 0.0001),
    "lambda_sum": (0.0001, 0.0004),
    "disp_r": (0, 100),
}

PARAM_KEYS = list(BOUNDS.keys())
NUM_DIMS = len(PARAM_KEYS)

def array_to_params(arr):
    # Converts a numpy array from the ML optimizer back into the simulation parameter dictionary
    params = copy.deepcopy(FIXED_PARAMS)
    for i, key in enumerate(PARAM_KEYS):
        params[key] = arr[i]
    return params

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================
    
def loss(parameters, runs=5):
    losses = []
    r = parameters.get("disp_r")

    for _ in range(runs):
        sim = simulate(make_initial_state(Hi_list, num_infected), steps=max(obs_times)+1, parameters=parameters, Print=False)

        nll = 0.0
        for i, t in enumerate(obs_times):
            pred = sim["Hi"][t]
            obs = obs_Hi[i]

            pred = max(pred, 1e-9) # remove log(0) domain errs if predicts extinction
            p = r / (r + pred) # for variance = pred + (pred^2 / r)

            # log likelihood based on NB2
            ll = (gammaln(obs + r) - gammaln(obs + 1) - gammaln(r) 
                  + r * np.log(p) + obs * np.log(1 - p))
            
            nll -= ll # subtracted b/c we're trying to minimize

        losses.append(nll)

    return np.mean(losses)

# ---------------------------
# MPI-Parallelized PSO Engine
# ---------------------------

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # PSO Hyperparameters
    num_particles = max(size * 4, 40) # Ensure we have enough particles to saturate all cores
    max_iterations = 1000
    w = 0.7298   # Inertia weight
    c1 = 1.49618 # Cognitive coefficient (Personal Best)
    c2 = 1.49618 # Social coefficient (Global Best)

    # Initialize Swarm Variables (Root only)
    if rank == 0:
        positions = np.zeros((num_particles, NUM_DIMS))
        velocities = np.zeros((num_particles, NUM_DIMS))
        
        # Randomly initialize positions and velocities within bounds
        for i, key in enumerate(PARAM_KEYS):
            lower, upper = BOUNDS[key]
            positions[:, i] = np.random.uniform(lower, upper, num_particles)
            velocities[:, i] = np.random.uniform(-0.1*(upper-lower), 0.1*(upper-lower), num_particles)
            
        pbests = np.copy(positions)
        pbest_scores = np.full(num_particles, np.inf)
        
        gbest = np.zeros(NUM_DIMS)
        gbest_score = np.inf
        
        print(f"Starting MPI Parallel PSO Optimization with {num_particles} particles on {size} nodes over {max_iterations} iterations...\n")
    else:
        positions = None

    # Optimization Loop
    for it in range(max_iterations):
        # Linearly decay inertia weight from 0.9 down to 0.4
        w = 0.9 - ((0.9 - 0.4) * (it / max_iterations))
        
        # Broadcast the current particle positions to all compute nodes
        positions = comm.bcast(positions, root=0)
        
        # Evaluate the loss function in parallel
        # Each node handles a slice of the particles, stepping by `size`
        local_results = []
        for i in range(rank, num_particles, size):
            params = array_to_params(positions[i])
            particle_loss = loss(params)
            local_results.append((i, particle_loss))
            
        # Gather results back to the root node
        gathered_results = comm.gather(local_results, root=0)
        
        # Update the Swarm memory and velocities (Root only)
        if rank == 0:
            # Flatten the gathered results
            for res_list in gathered_results:
                for i, score in res_list:
                    # Update Personal Best
                    if score < pbest_scores[i]:
                        pbest_scores[i] = score
                        pbests[i] = np.copy(positions[i])
                        
                    # Update Global Best
                    if score < gbest_score:
                        gbest_score = score
                        gbest = np.copy(positions[i])
            
            print(f"Iteration {it+1:3d}/{max_iterations} | Best Loss: {gbest_score:.4f} | Best Params: {array_to_params(gbest)}")
            
            # Update Velocities and Positions
            for i in range(num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                
                # PSO Velocity update formula
                velocities[i] = (w * velocities[i] + 
                                 c1 * r1 * (pbests[i] - positions[i]) + 
                                 c2 * r2 * (gbest - positions[i]))
                
                # Position update
                positions[i] += velocities[i]
                
                # Enforce bounds
                for j, key in enumerate(PARAM_KEYS):
                    lower, upper = BOUNDS[key]
                    if positions[i][j] < lower:
                        positions[i][j] = lower
                        velocities[i][j] *= -0.5 # bounce back slightly
                    elif positions[i][j] > upper:
                        positions[i][j] = upper
                        velocities[i][j] *= -0.5
                        
                # Mutation Operator
                # 5% chance to randomly teleport a particle to a new spot
                # to prevent the swarm from getting stuck in a local minimum.
                mutation_rate = 0.05 
                if np.random.rand() < mutation_rate:
                    for j, key in enumerate(PARAM_KEYS):
                        lower, upper = BOUNDS[key]
                        positions[i][j] = np.random.uniform(lower, upper)
                        # Give it a fresh random velocity to explore the new area
                        velocities[i][j] = np.random.uniform(-0.1*(upper-lower), 0.1*(upper-lower))
                        
    # -----------------------
    # Finish and Plot Results
    # -----------------------
    
    if rank == 0:
        best_final_params = array_to_params(gbest)
        
        print("\n=============================================")
        print("OPTIMIZATION COMPLETE")
        print(f"GLOBAL BEST LOSS: {gbest_score}")
        print(f"GLOBAL BEST PARAMS: {best_final_params}")
        print("=============================================\n")

        # Run one final simulation with the best parameters and plot it
        best_sim = simulate(make_initial_state(Hi_list, num_infected), steps=4500, parameters=best_final_params, Print=False)
        plot_history_highlights(best_sim, 
                                best_final_params['win_length'], 
                                best_final_params['win_start'], 
                                best_final_params['T_seasonal'], 
                                sample=[obs_times, obs_Hi])

if __name__ == "__main__":
    main()