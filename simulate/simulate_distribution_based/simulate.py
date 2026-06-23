import random as rand
import numpy as np

from simulate.simulate_distribution_based.helper_funcs import *
from simulate.simulate_distribution_based.rules import *

def simulate(initial_state, steps, parameters):
    state = initial_state
    win_length = parameters["win_length"]
    win_start = parameters["win_start"]

    history = {
        "Hi": np.empty(steps,dtype=np.int32),
        "Ot": np.empty(steps,dtype=np.int32),
        "In": np.empty(steps,dtype=np.int32),
        "Im": np.empty(steps,dtype=np.int32),
        "De": np.empty(steps,dtype=np.int32),
    }

    for t in range(steps):

        # Seasonal tempcycle
        if ((t + win_start) % 365) <= win_length: # win_length
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

        if t % 100 == 0:
            print(f"done w/ simulation at step {t}")

    return history
