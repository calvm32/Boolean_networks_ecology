import random as rand
import numpy as np
import copy
from mpi4py import MPI

from simulate.simulate_rough_original.helper_funcs import *
from simulate.simulate_rough_original.rules import *
from simulate.data import *

# --------------------
# set up control group
# --------------------

data = happy_jack_data()

START_YEAR = data[0]["year"]
SAMPLE_DAY = data[0]["day"]

obs_times = []
obs_NHO = []
obs_Ot = []
obs_In = []

for d in data:
    t = d["day"] + 365*(d["year"]-START_YEAR)
    
    obs_times.append(t)
    obs_NHO.append(d["NHO"])
    obs_Ot.append(d["Ot"])
    obs_In.append(d["In"])

def sample_params():
    water = rand.uniform(100,5000)
    food = rand.uniform(100,5000)
    return {
        "p_infected": 0, #rand.uniform(0.01, 0.8),
        "p_dead": 0, #rand.uniform(0.01, 0.8),
        "p_awake": rand.uniform(0.01, 0.2),
        "p_recover": 0, #rand.uniform(0.01, 0.8),
        "p_hibernate": rand.uniform(0.01, 0.8),
        "p_netchange": rand.uniform(0.0001, 0.0003),
        "water": water,
        "food": water,
        "water0": water,
        "food0": food,
        "win_length": win_length,
        "immunity_period": immunity_period,
        "contact_rate": contact_rate,
        "birth_resistance_max": birth_resistance_max,
        "recover_resistance_max": recover_resistance_max,
    }

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
contact_rate = 10                           # population-dependent rate of contact btwn health bat and WNS infected bat or surface

# -----------------
# types of immunity
# -----------------

immunity_period = 0                         # number of days spent in recovery before re-infection is possible
birth_resistance_max = 0.02                 # hereditary resistance of newborn, corresp. w/ rand.normalvariate(0, X)
recover_resistance_max = 0.02               # resistance after recovery, corresp. w/ rand.normalvariate(0, X)birth_resistance_max = 0.02                # corresp. w/ rand.normalvariate(0, X)

# ----------------------------------------
# hibernacula-DEPENDENT initial conditions
# ----------------------------------------

# population counts
Hi_num = 100            # hibernating bats
NHO_num = 0         # non-hibernating non-infected bats
In_num = 1              # non-hibernating infected bats
Ot_num = 0              # other bats
Im_num = 0              # recovered bats

# resource limits
water = 1000            # OKAY # number of bats it would take to deplete water completely
food = 1000             # OKAY # number of bats it would take to deplete food completely

time = 3650             # total days
winter = 120            # CONFIDENT # length of win_length season in Nebraska mines

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
    
def loss(parameters, runs=2):
    losses = []

    for _ in range(runs):
        sim = simulate(make_initial_state(), steps=max(obs_times)+1, parameters=parameters)

        error = 0.0
        for i, t in enumerate(obs_times):
            pred_N = sim["NHO"][t]
            pred_O = sim["Ot"][t]
            pred_I = sim["In"][t]

            error += (pred_N - obs_NHO[i])**2
            error += (pred_O - obs_Ot[i])**2
            error += (pred_I - obs_In[i])**2

        losses.append(error / len(obs_times))

    return np.mean(losses)

# ----------
# run things
# ----------

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
        "immunity_period": immunity_period,
        "contact_rate": contact_rate,
        "birth_resistance_max": birth_resistance_max,
        "recover_resistance_max": recover_resistance_max,
    }

    best = None
    best_loss = float("inf")
    local_best = None
    local_best_loss = float("inf")

    n_iter = 10000
    local_iters = n_iter // size

    for i in range(local_iters):
        params = sample_params()
        L = loss(params)

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

    best_sim = simulate(make_initial_state(), steps = 4500, parameters=best)
    plot_history_highlights(best_sim, win_length, sample=[obs_times, obs_NHO])


if __name__ == "__main__":
    main()


"""
New best: 2183.6363636363635 {'p_infected': 0.007998399524002234, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.21995833929973632, 'p_netchange': 0.0005224715374178368, 'water': 941.0402072309407, 'food': 5000, 'win_length': 120}
New best: 3743.8863636363635 {'p_infected': 0.041521217775843514, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.3986495789950476, 'p_netchange': 0.000515467966854555, 'water': 990.266884779426, 'food': 5000, 'win_length': 120}
New best: 1235.3636363636365 {'p_infected': 0.01524217865079747, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.2697086101327978, 'p_netchange': 0.0004065503241910787, 'water': 735.3878583939961, 'food': 5000, 'win_length': 120}
New best: 362.0909090909091 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5629538071675663, 'p_netchange': 0.0002439501436381033, 'water': 1829.7787389194987, 'food': 5000, 'win_length': 120}
New best: 358.22727272727275 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.48303984638268127, 'p_netchange': 0.00020677835763457532, 'water': 5000, 'food': 5000, 'win_length': 120}
New best: 349.79545454545456 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5573477611867865, 'p_netchange': 0.0002470489971729022, 'water': 5000, 'food': 5000, 'win_length': 120}
New best: 340.4318181818182 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5375541529135586, 'p_netchange': 0.0002226736397319496, 'water': 5000, 'food': 5000, 'win_length': 120}
New best: 353.59090909090907 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5217376853281397, 'p_netchange': 0.0002371693157187258, 'water': 5000, 'food': 5000, 'win_length': 127.79382882757909}
New best: 351.3636363636364 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.14737923993374222, 'p_recover': 0, 'p_hibernate': 0.5398474209465732, 'p_netchange': 0.00024587321732499006, 'water': 5000, 'food': 5000, 'win_length': 120}
GLOBAL BEST: 324.8863636363636 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.08177036053584033, 'p_recover': 0, 'p_hibernate': 0.506130033345668, 'p_netchange': 0.00022945884090207235, 'water': 5000, 'food': 5000, 'win_length': 120}
GLOBAL BEST: 302.3636363636364 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.26956807970377467, 'p_recover': 0, 'p_hibernate': 0.5448954517680774, 'p_netchange': 0.00021523603923560332, 'water': 927.0382284770973, 'food': 414.8102016331979, 'win_length': 120}
GLOBAL BEST: 324.70454545454544 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.08629881107887209, 'p_recover': 0, 'p_hibernate': 0.4521499873392658, 'p_netchange': 0.00021204118330130886, 'water': 191.12299861491675, 'food': 58.85014554319323, 'win_length': 120}
GLOBAL BEST: 355.5909090909091 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.053821586907585324, 'p_recover': 0, 'p_hibernate': 0.5757344511791069, 'p_netchange': 0.00021559487804763611, 'water': 616.2438584507863, 'food': 60.8566143812073, 'win_length': 120}
GLOBAL BEST: 344.77272727272725 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.14891365897212455, 'p_recover': 0, 'p_hibernate': 0.4777174885328155, 'p_netchange': 0.0002050202186951778, 'water': 412.01349345275577, 'food': 898.6018997126824, 'win_length': 120}
New best: 317.9545454545455 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.04345960635454034, 'p_recover': 0, 'p_hibernate': 0.5332425535631224, 'p_netchange': 0.00024123902192027023, 'water': 869.4131416026416, 'food': 920.9508227333656, 'win_length': 120}
New best: 321.1363636363636 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.11996019118621024, 'p_recover': 0, 'p_hibernate': 0.5358468255532175, 'p_netchange': 0.00022540489216466345, 'water': 246.9063201140136, 'food': 386.8256630780159, 'win_length': 120}
New best: 308.72727272727275 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.08, 'p_recover': 0, 'p_hibernate': 0.5, 'p_netchange': 0.0002, 'water': 350.34867097205273, 'food': 324.50109337947646, 'win_length': 120}
New best: 308.04545454545456 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.08, 'p_recover': 0, 'p_hibernate': 0.5, 'p_netchange': 0.0002, 'water': 371.46665433373187, 'food': 164.4683879543965, 'win_length': 120}
GLOBAL BEST: 286.0909090909091 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.08, 'p_recover': 0 , 'p_hibernate': 0.5, 'p_netchange': 0.0002, 'water': 406.0607758097961, 'food': 389.59646170871036, 'win_length': 120}
GLOBAL BEST: 255.58333333333331 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.08, 'p_recover': 0, 'p_hibernate': 0.5, 'p_netchange': 0.00021506633607029088, 'water': 291.10905941 12792, 'food': 168.22746809535204, 'win_length': 120}    

RESULTS:
    - p_awake = 0.08
    - p_netchange = 0.00021
    - p_hibernate = 0.5
    - food and water genuinely seem to have no effect here, 
      but that makes sense given the data we're testing against

"""