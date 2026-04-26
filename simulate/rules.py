import random as rand

def update_environment(state, agg, parameters):
    Wa, Fo, Te, Hu, WNS = state["Wa"], state["Fo"], state["Te"], state["Hu"], state["WNS"]
    NHi_NIn, Ot, Hi, In = agg["NHi_NIn_any"], agg["Ot_any"], agg["Hi_any"], agg["In_any"]
    water, food = parameters["water"], parameters["food"]

    Wa_next = Wa
    Fo_next = Fo
    Hu_next = Hu
    WNS_next = WNS

    # -----------
    # apply rules
    # -----------
    if NHi_NIn + In + Ot >= water: # rule 1 + 2
        Wa_next = 0
    if NHi_NIn + In + Ot >= food: # rule 1 + 2
        Fo_next = 0
    if Wa == 1 and Te == 1: Hu_next = 0 # rule 4
    if Wa == 0: Fo_next = 0 # rule 5
    if Te == 1: Wa_next = 1; Fo_next = 1 # rule 7
    if Wa == 1 and Hu == 1: WNS_next = 0
    if In >= 1: WNS_next = 1

    return {
        "Wa": int(Wa_next),
        "Fo": int(Fo_next),
        "Hu": int(Hu_next),
        "WNS": int(WNS_next),
        "Te": int(Te)
    }

def update_individuals(state, env, parameters):
    Wa, Fo, Te, Hu, WNS = env["Wa"], env["Fo"], env["Te"], env["Hu"], env["WNS"]
    p_awake, p_dead, p_hibernate, p_infected, p_recover, p_netchange = parameters["p_awake"], parameters["p_dead"], parameters["p_hibernate"], parameters["p_infected"], parameters["p_recover"], parameters["p_netchange"]
    food, water, recovery_period = parameters["food"], parameters["water"], parameters["recovery_period"]

    Hi_next  = []
    NHi_NIn_next = []
    Ot_next  = []
    In_next  = []
    Re_next = []
    
    # Ensure De is long enough to cover all inhabitants
    total_inhabitants = (
        len(state["Hi"]) +
        len(state["NHi_NIn"]) +
        len(state["Ot"]) +
        len(state["In"]) +
        len(state["Re"])
    )

    De_next  = state["De"][:]
    if len(De_next) < total_inhabitants:
        De_next += [0]*(total_inhabitants - len(De_next))

    # ---------
    # Update Hi
    # ---------
    for i in range(len(state["Hi"])):
        Hi = state["Hi"][i]

        if WNS and Hi and rand.uniform(0, 1) <= p_infected: # rule 9
            In_next.append(1)
        elif not Te and Hi and rand.uniform(0, 1) <= p_awake: # rule 1
            NHi_NIn_next.append(1)
        elif Te and Hi and rand.uniform(0,1) <= p_hibernate: # rules 6/7
            NHi_NIn_next.append(1) 
        else:
            Hi_next.append(Hi)

    # --------------
    # Update NHi_NIn
    # --------------
    for i in range(len(state["NHi_NIn"])):
        NHi_NIn = state["NHi_NIn"][i]

        if not Wa and not Fo: # rule 3
            De_next[len(state["Hi"]) + i] = 1
        elif not Te and NHi_NIn and rand.uniform(0,1) <= p_hibernate: # rules 6/7
            Hi_next.append(1) 
        else:
            NHi_NIn_next.append(NHi_NIn)

    # ---------
    # Update Ot
    # ---------
    for i in range(len(state["Ot"])):
        Ot = state["Ot"][i]
        idx = len(state["Hi"]) + len(state["NHi_NIn"]) + i

        if not Wa and not Fo: # rule 3
            De_next[idx] = 1
        else:
            Ot_next.append(Ot)

    # ---------
    # Update In
    # ---------
    for i in range(len(state["In"])):
        In = state["In"][i]
        idx = len(state["Hi"]) + len(state["NHi_NIn"]) + len(state["Ot"]) + i

        if In and rand.uniform(0, 1) <= p_dead: # rule 9
            De_next[idx] = 1
        elif not Wa and not Fo: # rule 3
            De_next[idx] = 1
        elif In and rand.uniform(0,1) <= p_recover: # rule 13
            Re_next.append(0) # start recovery counter
        else:
            In_next.append(In)

    # ---------
    # Update Re
    # ---------
    for i in range(len(state["Re"])):
        age = state["Re"][i]

        if age >= recovery_period:
            Hi_next.append(1) # return to hibernation
        else:
            Re_next.append(age + 1)

    # --------------------
    # Influx (summer only)
    # --------------------
    if Te == 1:
        n_influx = len(NHi_NIn_next) + len(Ot_next) # active bats reproduce
        
        births = 0
        for _ in range(n_influx):
            if rand.uniform(0, 1) <= p_netchange:
                births += 1

        NHi_NIn_next.extend([1]*births)
        De_next.extend([0]*births) # extend De so indexing doesn't break

    return {
        "Hi": Hi_next,
        "NHi_NIn": NHi_NIn_next,
        "Ot": Ot_next,
        "In": In_next,
        "De": De_next,
        "Re": Re_next
    }