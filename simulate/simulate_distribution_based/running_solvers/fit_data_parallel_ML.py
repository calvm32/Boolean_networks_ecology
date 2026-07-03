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


# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

# -------------------------
# set up initial population
# -------------------------

tricolor_num = 100
tricolor_cluster_sizeMIN = 1
tricolor_cluster_sizeMAX = 2

bigbrown_num = 0
bigbrown_cluster_sizeMIN = 1
bigbrown_cluster_sizeMAX = 9

Hi_list = [[tricolor_num, tricolor_cluster_sizeMIN, tricolor_cluster_sizeMAX], 
           [bigbrown_num, bigbrown_cluster_sizeMIN, bigbrown_cluster_sizeMAX]] 

fraction_infected = 0  

# -----------------------------------
# system-governing parameters (FIXED)
# -----------------------------------

inf_alpha, inf_beta = 5, 2                  
delta = 0.05                                
T_inf = 30                                  
T_TBD = 4.1                                 
T_AD = 88.5/1440                            
lambda_win = 0                              
res_max = 0.2                               
k_imm, theta_imm = 1, 1                     
res_gain = 0.02                             

# Static parameters that do not get optimized
FIXED_PARAMS = {
    "inf_alpha": inf_alpha,
    "inf_beta": inf_beta,
    "delta": delta,
    "T_inf": T_inf,
    "T_TBD": T_TBD,
    "T_AD": T_AD,
    "lambda_win": lambda_win,
    "res_gain": res_gain,
    "res_max": res_max,
    "k_imm": k_imm,
    "theta_imm": theta_imm,
}

# ------------------------------------
# Parameters to Optimize (Search Space)
# ------------------------------------

BOUNDS = {
    "T_seasonal": (20.0, 60.0),
    "win_length": (80.0, 200.0),
    "win_start": (230.0, 290.0),
    "lambda_sum": (0.00010, 0.00030)
}
PARAM_KEYS = list(BOUNDS.keys())
NUM_DIMS = len(PARAM_KEYS)

def array_to_params(arr):
    """Converts a numpy array from the ML optimizer back into the simulation parameter dictionary."""
    params = copy.deepcopy(FIXED_PARAMS)
    for i, key in enumerate(PARAM_KEYS):
        params[key] = arr[i]
    return params

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
            error += (pred_Hi - obs_Hi[i])**2

        losses.append(error / len(obs_times))

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
        
        # 1. Broadcast the current particle positions to all compute nodes
        positions = comm.bcast(positions, root=0)
        
        # 2. Evaluate the loss function in parallel
        # Each node handles a slice of the particles, stepping by `size`
        local_results = []
        for i in range(rank, num_particles, size):
            params = array_to_params(positions[i])
            particle_loss = loss(params)
            local_results.append((i, particle_loss))
            
        # 3. Gather results back to the root node
        gathered_results = comm.gather(local_results, root=0)
        
        # 4. Update the Swarm memory and velocities (Root only)
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
                        
                # === NEW: Mutation Operator ("Craziness") ===
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
        best_sim = simulate(make_initial_state(Hi_list, fraction_infected), steps=4500, parameters=best_final_params, Print=False)
        plot_history_highlights(best_sim, 
                                best_final_params['win_length'], 
                                best_final_params['win_start'], 
                                best_final_params['T_seasonal'], 
                                sample=[obs_times, obs_Hi])

if __name__ == "__main__":
    main()