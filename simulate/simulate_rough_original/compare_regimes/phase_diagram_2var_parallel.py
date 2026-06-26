import random as rand
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpi4py import MPI

from simulate.simulate_rough_original.helper_funcs import *
from simulate.simulate_rough_original.rules import *

# ------------------------------------------
# hibernacula-INDEPENDENT initial conditions
# ------------------------------------------

# probabilities
p_infected = 0.005                          # chance a hibernating bat gets infected (given that PD is on) on any given day
p_dead = 0.005                              # chance that an infected bat dies on any given day
p_recover = 1-(1-(1-p_dead)**30)**(1/30)    # CONFIDENT # chance of recovering and going back into hibernation on any given day
p_awake = 0.08                              # OKAY # chance of a waking bat arousing a hibernating bat from torpor on any given day
p_hibernate = 0.5                           # CONFIDENT # chance of a bat switching between hibernating and not (given that Te switches) on any given day
p_netchange = 0.000215                      # CONFIDENT # chance of new bat due to immigration/birth per day

# -----------------
# types of immunity
# -----------------

T_im = 0                         # number of days spent in recovery before re-infection is possible
res_max = 0.02                 # hereditary resistance of newborn, corresp. w/ rand.normalvariate(0, X)
recover_resistance_max = 0.02               # resistance after recovery, corresp. w/ rand.normalvariate(0, X)res_max = 0.02                 # corresp. w/ rand.normalvariate(0, X)

# ----------------------------------------
# hibernacula-DEPENDENT initial conditions
# ----------------------------------------

contact_rate = 10       # population-dependent rate of contact btwn healthy bat and WNS infected bat on any given day

# population counts
Hi_num = 100            # hibernating bats
NHO_num = 0              # non-hibernating non-infected bats
In_num = 1              # non-hibernating infected bats
Ot_num = 0              # other bats
Im_num = 0              # recovered bats

# resource limits
water = 1000            # OKAY # number of bats it would take to deplete water completely
food = 1000             # OKAY # number of bats it would take to deplete food completely

time = 3650             # total days
win_length = 120            # CONFIDENT # length of winter season in Nebraska mines

# ------------------
# bifurcation values
# ------------------

times_list = [180, 365, 3*365, 10*365, 20*365, 40*365]
num_params = 30

# ------------------
# parameters to test
# ------------------

# dead vs inf
param_change = ["p_dead", "p_infected"]
parameters_list = [np.linspace(0.001,0.5,num_params), np.linspace(0.001,0.5,num_params)]
title = "deadvinf"

# inf v immune
# param_change = ["p_infected", "T_im"]
# parameters_list = [np.linspace(0.001,0.1,num_params), np.linspace(0,130,num_params)]
# title = "infvimm"

# inf v contact rate
# param_change = ["p_infected", "contact_rate"]
# parameters_list = [np.linspace(0.001,0.1,num_params), np.linspace(0,130,num_params)]
# title = "infvcon"

# --------------
# actual testing
# --------------

totals_list = np.empty((num_params,num_params), dtype=object)

for i in range(num_params):
    for j in range(num_params):
        totals_list[i, j] = []

# ----------
# initialize
# ----------

res_num = 0             # starting resistance for bats in the hibernaculum

def make_initial_state():
    return {
        "Hi": [[1, res_num] for _ in range(Hi_num)],
        "NHO": [[1, res_num] for _ in range(NHO_num)],
        "Ot": [[1, res_num] for _ in range(Ot_num)],
        "In": [[1, res_num] for _ in range(In_num)],
        "De": [[0, res_num] for _ in range(Hi_num + NHO_num + In_num)],
        "Im": [[0, res_num] for _ in range(Im_num)],
        "Wa": 1,
        "Fo": 1,
        "Te": 0,
        "Hu": 0,
        "PD": 0,
    }

def simulate(initial_state, steps, parameters):
    state = initial_state
    win_length = parameters["win_length"]

    history = {
        "Hi":[],
        "NHO":[],
        "Ot":[],
        "In":[],
        "De":[],
        "Im":[],
    }

    for t in range(steps):

        # Seasonal tempcycle
        if (t % 365) <= win_length: # win_length
            state["Te"] = 0   
        else:
            state["Te"] = 1 # summer
        counts = count(state)

        history["Hi"].append(counts["Hi"])
        history["NHO"].append(counts["NHO"])
        history["Ot"].append(counts["Ot"])
        history["In"].append(counts["In"])
        history["De"].append(counts["De"])
        history["Im"].append(counts["Im"])

        state = step(state, parameters)

    return history


def main():

    # setup mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    inhabitant_nodes = ["Hi", "NHO", "In", "Ot", "De", "Im"]
    resource_nodes = ["Wa", "Fo"]
    environment_nodes = ["Te", "Hu", "El", "Po", "Su", "Ba", "PD"]

    nodes = inhabitant_nodes + resource_nodes + environment_nodes

    parameters = {
        "p_infected": p_infected,
        "p_dead": p_dead,
        "p_awake": p_awake,
        "p_recover": p_recover,
        "p_hibernate": p_hibernate,
        "p_netchange": p_netchange,
        "water": water,
        "food": food,
        "water0": water,
        "food0": food,
        "win_length": win_length,
        "T_im": T_im,
        "contact_rate": contact_rate,
        "res_max": res_max,
        "recover_resistance_max": recover_resistance_max,
    }
    

    # ----------------------
    # distribute jobs (root)
    # ----------------------
    if rank == 0:
        jobs = [(i, j) for i in range(num_params) for j in range(num_params)]
        chunks = np.array_split(jobs, size)
    else:
        chunks = None

    # send jobs to all ranks
    local_jobs = comm.scatter(chunks, root=0)

    # ---------------
    # compute locally
    # ---------------
    local_results = []

    for (i, j) in local_jobs:
        parameters_local = parameters.copy()
        parameters_local[param_change[0]] = parameters_list[0][i]
        parameters_local[param_change[1]] = parameters_list[1][j]

        history = simulate(make_initial_state(), steps=times_list[-1], parameters=parameters_local)

        total = (
            np.array(history["Hi"]) +
            np.array(history["NHO"]) +
            np.array(history["In"]) +
            np.array(history["Im"])
        )

        local_results.append((i, j, total))

        if (i % 10 == 0) and (j % 10 == 0):
            print(f"[rank {rank}] ({i},{j})")

    # ------------------------
    # gather results + rebuild
    # ------------------------
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        for proc_results in all_results:
            for i, j, total in proc_results:
                totals_list[i][j] = total

    if rank == 0:
        rows = 2
        cols = len(times_list)// rows
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 10), subplot_kw={'projection': '3d'}, constrained_layout=True)
        axes = axes.ravel()

        # for 3d plotting
        X, Y = np.meshgrid(np.ravel(parameters_list[0]), np.ravel(parameters_list[1])) # create meshgrid for X and Y
        Z = np.zeros(X.shape) # initialize Z values (total population) for the 3D plot

        for i, time in enumerate(times_list):
            Z.fill(0) # clear Z before updating with new values

            for j in range(totals_list.shape[0]):
                for k in range(totals_list.shape[1]):
                    Z[j, k] = totals_list[j][k][time - 1]

            # plot the surface for the current time slice
            ax = axes[i]
            ax.plot_surface(X, Y, Z, cmap='viridis')

            ax.set_xlabel(f"{param_change[0]}")
            ax.set_ylabel(f"{param_change[1]}")
            ax.set_zlabel(f"Population at day {time}")
            ax.set_title(f"Population at day {time}", pad=15)

        for ax in axes:
            ax.zaxis.labelpad = 10
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10
        
        fig.set_constrained_layout_pads(
            w_pad=0.05,   # width padding
            h_pad=0.1,    # height padding
            hspace=0.2,
            wspace=0.2
        )

        plt.savefig(f"3D_bifurcations_{title}.png", dpi=200)
        plt.show()
    
if __name__ == "__main__":
    main()