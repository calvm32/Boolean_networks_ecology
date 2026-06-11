import random as rand

def update_environment(state, agg, parameters):
    Wa, Fo, Te, Hu, PD = state["Wa"], state["Fo"], state["Te"], state["Hu"], state["PD"]
    NHO_any, Ot_any, Hi_any, In_any = agg["NHO_any"], agg["Ot_any"], agg["Hi_any"], agg["In_any"]
    NHO_sum, Ot_sum, Hi_sum, In_sum = agg["NHO_sum"], agg["Ot_sum"], agg["Hi_sum"], agg["In_sum"]
    water, food, water0, food0 = parameters["water"], parameters["food"], parameters["water0"], parameters["food0"]

    Wa_next = Wa
    Fo_next = Fo
    Hu_next = Hu
    PD_next = PD

    # -----------
    # apply rules
    # -----------
    if Te == 0 and NHO_sum + In_sum + Ot_sum >= water: # rule 1 + 2
        Wa_next = 0
    if Te == 0 and NHO_sum + In_sum + Ot_sum >= food: # rule 1 + 2
        Fo_next = 0
    if Wa == 1 and Te == 1: Hu_next = 1 # rule 4
    if Wa == 0: Fo_next = 0 # rule 5
    if Te == 1: Wa_next = 1; Fo_next = 1 # rule 7
    if Wa == 1 and Hu == 1: PD_next = 1 # ????
    if In_sum >= 1:
        PD_next = 1
    if Te == 1:
        # PD_next = 0 # PD dies in summer 
        water = water0 # food levels are restored
        food = food0

    return {
        "Wa": int(Wa_next),
        "Fo": int(Fo_next),
        "Hu": int(Hu_next),
        "PD": int(PD_next),
        "Te": int(Te)
    }

def update_individuals(state, env, parameters):
    Wa, Fo, Te, Hu, PD = env["Wa"], env["Fo"], env["Te"], env["Hu"], env["PD"]
    p_awake, p_dead, p_hibernate, p_infected, p_recover, p_netchange = parameters["p_awake"], parameters["p_dead"], parameters["p_hibernate"], parameters["p_infected"], parameters["p_recover"], parameters["p_netchange"]
    food, water, immunity_period, contact_rate, birth_res_max, recover_res_max = parameters["food"], parameters["water"], parameters["immunity_period"], parameters["contact_rate"], parameters["birth_resistance_max"], parameters["recover_resistance_max"]

    # go over OLD STATE
    Hi_old = state["Hi"]
    NHO_old = state["NHO"]
    Ot_old = state["Ot"]
    In_old = state["In"]
    Im_old = state["Im"]

    # readd everyone who stays the same
    Hi_next  = []
    NHO_next = []
    Ot_next  = []
    In_next  = []
    Im_next = []
    
    # ensure De is long enough to cover all inhabitants
    total_inhabitants = (
        len(state["Hi"]) +
        len(state["NHO"]) +
        len(state["Ot"]) +
        len(state["In"]) +
        len(state["Im"])
    )

    infected = len(state["In"])
    if total_inhabitants == 0:
        SIR_infection_rate = 0
    else:
        SIR_infection_rate = (p_infected * contact_rate * infected / total_inhabitants)

    SIR_infection_rate = min(1, SIR_infection_rate)

    De_next = state["De"][:]
    if len(De_next) < total_inhabitants:
        De_next += [0]*(total_inhabitants - len(De_next))

    # ---------
    # Update Hi
    # ---------
    for i in range(len(Hi_old)):
        Hi = state["Hi"][i][0]
        res_num = state["Hi"][i][1]

        effective_rate = SIR_infection_rate * (1 - res_num)
        r_env = rand.uniform(0,1)
        r_bat = rand.uniform(0,1)

        if (Hi and Te == 0 and r_bat < effective_rate) or (PD and Hi and Te == 0 and r_env < p_infected): # rule 9
            In_next.append([1, res_num])
        elif not Te and Hi and r_bat <= p_awake: # rule 1
            NHO_next.append([1, res_num])
        elif Te and Hi and r_bat <= p_hibernate: # rules 6/7
            NHO_next.append([1, res_num]) 
        else:
            Hi_next.append([Hi, res_num]) # keep current Hi

    # ----------
    # Update NHO
    # ----------
    for i in range(len(NHO_old)):
        NHO = state["NHO"][i][0]
        res_num = state["NHO"][i][1]
        r = rand.uniform(0, 1)

        if not Te and not Wa and not Fo: # rule 3
            De_next[len(state["Hi"]) + i] = [1, res_num]
        elif not Te and NHO and r <= p_hibernate: # rules 6/7
            Hi_next.append([1, res_num]) 
        else:
            NHO_next.append([NHO, res_num])

    # ---------
    # Update Ot
    # ---------
    for i in range(len(Ot_old)):
        Ot = state["Ot"][i][0]
        res_num = state["Ot"][i][1]

        idx = len(state["Hi"]) + len(state["NHO"]) + i

        if not Te and not Wa and not Fo: # rule 3
            De_next[idx] = [1, res_num]
        else:
            Ot_next.append([Ot, res_num])

    # ---------
    # Update In
    # ---------
    for i in range(len(In_old)):
        In = state["In"][i][0]
        res_num = state["In"][i][1]

        idx = len(state["Hi"]) + len(state["NHO"]) + len(state["Ot"]) + i
        r = rand.uniform(0, 1)

        if (In and r <= p_dead) or (not Te and not Wa and not Fo): # rule 9 or 3
            De_next[idx] = [1, res_num]
        elif In and r <= p_recover: # rule 13
            Im_next.append([0, res_num]) # start recovery counter
        else:
            In_next.append([In, res_num])

    # ---------
    # Update Im
    # ---------
    for i in range(len(Im_old)):
        age = state["Im"][i][0]
        res_num = state["Im"][i][1]

        new_res_num = res_num + rand.normalvariate(0, recover_res_max)
        new_res_num = max(0, min(1, new_res_num))

        if age >= immunity_period:
            Hi_next.append([1, new_res_num]) # return to hibernation
        else:
            Im_next.append([age + 1, res_num])

    # --------------------
    # Influx (summer only)
    # --------------------
    if Te == 1:
        parents = state["NHO"] + state["Im"]
        for parent in parents:
            if rand.uniform(0,1) <= p_netchange:
                parent_res_num = parent[1]
                child_res_num = parent_res_num + rand.normalvariate(0, birth_res_max)
                child_res_num = max(0, min(1, child_res_num))

                NHO_next.append([1, child_res_num])
                De_next.append([0, child_res_num]) # extend De so indexing doesn't break
        
    return {
        "Hi": Hi_next,
        "NHO": NHO_next,
        "Ot": Ot_next,
        "In": In_next,
        "De": De_next,
        "Im": Im_next
    }