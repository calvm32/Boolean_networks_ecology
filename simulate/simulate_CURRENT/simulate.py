import random as rand
import numpy as np

from simulate.simulate_CURRENT.helper_funcs import *
from simulate.simulate_CURRENT.rules import *

def simulate(initial_state, steps, parameters, Print=100):
    state = initial_state
    win_length = parameters["win_length"]
    win_start = parameters["win_start"]

    history = {
        "Hi": np.empty(steps,dtype=np.int32),
        "Ot": np.empty(steps,dtype=np.int32),
        "In": np.empty(steps,dtype=np.int32),
        "Im": np.empty(steps,dtype=np.int32),
        "De": np.empty(steps,dtype=np.int32),
        "SC": 0,
    }

    for t in range(steps):

        counts = count(state)

        history["Hi"][t] = counts["Hi"]
        history["Ot"][t] = counts["Ot"]
        history["In"][t] = counts["In"]
        history["Im"][t] = counts["Im"]
        history["De"][t] = counts["De"]
        history["SC"] = counts["SC"]

        state = step(state, parameters, t)

        if Print != False and (t % Print == 0):
            print(f"done w/ simulation at step {t}")

    return history

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
        "SC": state["SC"],
    }

def perturb(params, keys, scale=0.15):
    new = params.copy()
    for k in keys:
        val = params[k]
        if val > 0:
            new[k] = max(0, val + np.random.normal(0, scale * val))
    return new