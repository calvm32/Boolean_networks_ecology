import random as rand
import numpy as np
from helper_funcs import *
from rules import *
import copy
from mpi4py import MPI

# --------------------------------------
# set up control group (happy jack mine)
# --------------------------------------

# happy jack data
data = [
    {"year": 2014, "NHi_NIn": 62, "Ot": 0, "In": 0},
    {"year": 2015, "NHi_NIn": 52, "Ot": 0, "In": 0},
    {"year": 2016, "NHi_NIn": 72, "Ot": 0, "In": 0},
    {"year": 2017, "NHi_NIn": 101, "Ot": 0, "In": 0},
    {"year": 2018, "NHi_NIn": 73, "Ot": 0, "In": 0},
    {"year": 2019, "NHi_NIn": 96, "Ot": 0, "In": 0},
    {"year": 2021, "NHi_NIn": 108, "Ot": 2, "In": 0},
    {"year": 2022, "NHi_NIn": 128, "Ot": 0, "In": 0},
    {"year": 2023, "NHi_NIn": 128, "Ot": 5, "In": 0},
    {"year": 2024, "NHi_NIn": 95, "Ot": 0, "In": 0},
    {"year": 2025, "NHi_NIn": 86, "Ot": 0, "In": 0},
]

START_YEAR = 2014
SAMPLE_DAY = 140

obs_times = []
obs_NHi_NIn = []
obs_Ot = []
obs_In = []

for d in data:
    t = SAMPLE_DAY + 365 * (d["year"] - START_YEAR)
    
    obs_times.append(t)
    obs_NHi_NIn.append(d["NHi_NIn"])
    obs_Ot.append(d["Ot"])
    obs_In.append(d["In"])

def sample_params():
    return {
        "p_infected": 0, #rand.uniform(0.001, 0.05),
        "p_dead": 0, # not needed for control
        "p_awake": rand.uniform(0.001, 0.3),
        "p_recover": 0, # not needed for control
        "p_hibernate": rand.uniform(0.45, 0.6),
        "p_influx": rand.uniform(0.0001, 0.0003),
        "water": 5000,
        "food": 5000,
        "winter": 120
    }

# ------------------------------------------
# hibernacula-INDEPENDENT initial conditions
# ------------------------------------------

# probabilities
p_infected = 0.01         # chance a hibernating bat gets infected (given that WNS is on) on any given day
p_dead = 0.005            # chance that an infected bat dies on any given day
p_awake = 0.02            # chance of a waking bat arousing a hibernating bat from torpor on any given day
p_recover = 0.01          # chance of recovering and going back into hibernation on any given day
p_hibernate = 0.5         # chance of a bat switching between hibernating and not (given that Te switches) on any given day
p_influx = 0.001          # chance of new bat due to immigration/birth per day

# ----------------------------------------
# hibernacula-DEPENDENT initial conditions
# ----------------------------------------

# population counts
Hi_num = 62            # hibernating bats
NHi_NIn_num = 0         # non-hibernating non-infected bats
In_num = 0              # non-hibernating infected bats
Ot_num = 0              # other bats

# resource limits
water = 5000            # number of bats it would take to deplete water completely
food = 5000             # number of bats it would take to deplete food completely

time = 1000              # total days
winter = 120            # length of winter season

# ----------
# initialize
# ----------

state0 = {
    "Hi": [1]*Hi_num,
    "NHi_NIn": [1]*NHi_NIn_num,
    "Ot": [1]*Ot_num,
    "In": [1]*In_num,
    "De": [0]*(Hi_num + NHi_NIn_num + In_num),

    "Wa": 1,
    "Fo": 1,
    "Te": 0,
    "Hu": 0,
    "WNS": 0,
}
    
def loss(parameters, runs=4):
    losses = []

    for _ in range(runs):
        sim = simulate(copy.deepcopy(state0), steps=max(obs_times)+1, parameters=parameters)

        error = 0.0
        for i, t in enumerate(obs_times):
            pred_N = sim["NHi_NIn"][t]
            pred_O = sim["Ot"][t]
            pred_I = sim["In"][t]

            error += (pred_N - obs_NHi_NIn[i])**2
            error += (pred_O - obs_Ot[i])**2
            error += (pred_I - obs_In[i])**2

        losses.append(error / len(obs_times))

    return np.mean(losses)

# ----------
# run things
# ----------

def simulate(initial_state, steps, parameters):
    state = initial_state
    winter = parameters["winter"]

    history = {
        "Hi":[],
        "NHi_NIn":[],
        "Ot":[],
        "In":[],
        "De":[],
    }

    for t in range(steps):

        # Seasonal tempcycle
        if (t % 365) <= winter: # winter
            state["Te"] = 0   
        else:
            state["Te"] = 1 # summer
        counts = count(state)

        history["Hi"].append(counts["Hi"])
        history["NHi_NIn"].append(counts["NHi_NIn"])
        history["Ot"].append(counts["Ot"])
        history["In"].append(counts["In"])
        history["De"].append(counts["De"])

        state = step(state, parameters)

    return history


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    inhabitant_nodes = ["Hi", "NHi_NIn", "In", "Ot", "De"]
    resource_nodes = ["Wa", "Fo"]
    environment_nodes = ["Te", "Hu", "El", "Po", "Su", "Ba", "WNS"]

    nodes = inhabitant_nodes + resource_nodes + environment_nodes

    parameters = {
        "p_infected": p_infected,
        "p_dead": p_dead,
        "p_awake": p_awake,
        "p_recover": p_recover,
        "p_hibernate": p_hibernate,
        "p_influx": p_influx,
        "water": water,
        "food": food,
        "winter": winter
    }

    if In_num != 0:
        state0["WNS"] = 1

    local_best = None
    local_best_loss = float("inf")

    # split work over nodes
    n_iter = 1000
    local_iters = n_iter // size

    for i in range(local_iters):
        params = sample_params()
        L = loss(params)

        if L < local_best_loss:
            local_best_loss = L
            local_best = params

        print(f"[rank {rank}] iteration {i}, loss={L}")

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


if __name__ == "__main__":
    main()


"""
New best: 2183.6363636363635 {'p_infected': 0.007998399524002234, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.21995833929973632, 'p_influx': 0.0005224715374178368, 'water': 941.0402072309407, 'food': 5000, 'winter': 120}
New best: 3743.8863636363635 {'p_infected': 0.041521217775843514, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.3986495789950476, 'p_influx': 0.000515467966854555, 'water': 990.266884779426, 'food': 5000, 'winter': 120}
New best: 1235.3636363636365 {'p_infected': 0.01524217865079747, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.2697086101327978, 'p_influx': 0.0004065503241910787, 'water': 735.3878583939961, 'food': 5000, 'winter': 120}
New best: 362.0909090909091 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5629538071675663, 'p_influx': 0.0002439501436381033, 'water': 1829.7787389194987, 'food': 5000, 'winter': 120}
New best: 358.22727272727275 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.48303984638268127, 'p_influx': 0.00020677835763457532, 'water': 5000, 'food': 5000, 'winter': 120}
New best: 349.79545454545456 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5573477611867865, 'p_influx': 0.0002470489971729022, 'water': 5000, 'food': 5000, 'winter': 120}
New best: 340.4318181818182 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5375541529135586, 'p_influx': 0.0002226736397319496, 'water': 5000, 'food': 5000, 'winter': 120}
New best: 353.59090909090907 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5217376853281397, 'p_influx': 0.0002371693157187258, 'water': 5000, 'food': 5000, 'winter': 127.79382882757909}
New best: 351.3636363636364 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.14737923993374222, 'p_recover': 0, 'p_hibernate': 0.5398474209465732, 'p_influx': 0.00024587321732499006, 'water': 5000, 'food': 5000, 'winter': 120}

RESULTS:
    - p_influx = 0.0002
    - p_hibernate = 0.56
    - water = much higher than population
    - 

"""