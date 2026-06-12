import random as rand
import numpy as np

def update_environment(state, agg, parameters):
    Te, Hu, PD, Re = state["Te"], state["Hu"], state["PD"], state["Re"]
    Ot_any, Hi_any, In_any, Im_any = agg["Ot_any"], agg["Hi_any"], agg["In_any"], agg["Im_any"]
    Ot_sum, Hi_sum, In_sum, Im_sum = agg["Ot_sum"], agg["Hi_sum"], agg["In_sum"], agg["Im_sum"]
    delta = parameters["delta"]

    Re_next = Re
    Hu_next = Hu
    PD_next = PD

    # -----------
    # apply rules
    # -----------
    if Re == 1 and Te == 1: Hu_next = 1 # rule 4
    if Te == 1: Re_next = 1  # rule 7

    PD_next = (1-delta)*PD + (In_sum)/(Ot_sum + Hi_sum + In_sum + Im_sum)

    return {
        "Hu": int(Hu_next),
        "PD": int(PD_next),
        "Re": int(Re_next),
        "Te": int(Te)
    }

def update_individuals(state, env, parameters):
    Te, Hu, PD, Re = state["Te"], state["Hu"], state["PD"], state["Re"]
    inf_alpha, inf_beta, delta, awake_a, awake_b = parameters["inf_alpha"], parameters["inf_beta"], parameters["delta"], parameters["awake_a"], parameters["awake_b"]
    T_inf, T_TBD, T_AD, T_seasonal, T_win = parameters["T_inf"], parameters["T_TBD"], parameters["T_AD"], parameters["T_seasonal"], parameters["T_win"]
    lambda_win, lambda_sum = parameters["lambda_win"], parameters["lambda_sum"]
    immunity_period, birth_res_max, recover_res_max = parameters["immunity_period"], parameters["birth_resistance_max"], parameters["recover_resistance_max"]

    # go over OLD STATE
    Hi_old = state["Hi"]
    Ot_old = state["Ot"]
    In_old = state["In"]
    Im_old = state["Im"]
    De = state["De"]

    # re-add everyone who stays the same
    Hi_next  = []
    Ot_next = []
    In_next  = []
    Im_next = []
    
    # configure totals for weighting parameters
    total_inhabitants = (
        len(state["Hi"]) +
        len(state["Ot"]) +
        len(state["In"]) +
        len(state["Im"])
    )

    weight = len(state["In"]) / total_inhabitants

    # environmental PD exposure
    p_env = PD/(1+PD)

    # seasonal hibernation
    p_seasonal = 1/T_seasonal

    # ---------
    # Update Hi
    # ---------
    for i in range(len(Hi_old)):
        Hi = state["Hi"][i][0]
        res_num = state["Hi"][i][1]
        cluster_num = state["Hi"][i][2]

        # bat-to-bat contact WNS exposure
        r = rand.uniform(0,1)
        lambda_i = weight*cluster_num
        K_i = np.random.poisson(lambda_i)

        p_i = np.random.beta(parameters["inf_alpha"], parameters["inf_beta"])
        p_bat = (1 - (1-p_i)**K_i) * (1-res_num)
        p_bat = min(1, p_bat)

        p_infected = 1 - (1-p_bat)*(1-p_env)

        # if infected, determine how long until recovery or death
        mu_mean = 1/T_inf
        k = 2.0
        theta = mu_mean / k

        mu_i = np.random.gamma(shape=k, scale=theta)

        # bout hibernation
        p_bouts = 1/T_TBD

        if (Hi and Te == 0 and r < p_infected): # rule 9
            In_next.append([1, res_num, cluster_num, mu_i, 0])
        elif not Te and Hi and r <= p_bouts: # rule 1
            Ot_next.append([1, res_num, cluster_num, 0, 1]) # mark [...,1] to signify next hibernating rule is p_bouts and not p_seasonal
        elif Te and Hi and r <= p_seasonal: # rules 6/7
            Ot_next.append([1, res_num, cluster_num, 0, 0]) 
        else:
            Hi_next.append([Hi, res_num, cluster_num, 0, 0])

    # ----------
    # Update Ot
    # ----------
    for i in range(len(Ot_old)):
        Ot = state["Ot"][i][0]
        res_num = state["Ot"][i][1]
        cluster_num = state["Ot"][i][2]
        r = rand.uniform(0, 1)

        # bout hibernation
        check = state["Ot"][i][4] # check if hibernated before, i.e. if the next hibernating rule is p_bouts and not p_seasonal
        p_bouts = 1/T_AD

        if not Te and not Re: # rule 3
            De += 1
        elif Te and Ot and r and check <= p_bouts: # rule 1
            Ot_next.append([1, res_num, cluster_num, 0, 0]) 
        elif not Te and Ot and r <= p_seasonal: # rules 6/7
            Ot_next.append([1, res_num, cluster_num, 0, 0]) 
        else:
            Ot_next.append([Ot, res_num, cluster_num, 0, 0])

    # ---------
    # Update In
    # ---------
    for i in range(len(In_old)):
        In = state["In"][i][0]
        res_num = state["In"][i][1]
        cluster_num = state["In"][i][2]
        mu_i = state["In"][i][3]

        idx = len(state["Hi"]) + len(state["Ot"]) + i
        r = rand.uniform(0, 1)

        # probability of death
        p_dead = (1 - np.exp(-mu_i))*(1 + Re)
        p_recover =  1/T_inf

        if (In and r <= p_dead) or (not Te and not Re): # rule 9 or 3
            De += 1
        elif In and r <= p_recover: # rule 13
            Im_next.append([0, res_num, cluster_num, 0, 0]) # start recovery counter
        else:
            In_next.append([In, res_num, cluster_num, 0, 0])

    # ---------
    # Update Im
    # ---------
    for i in range(len(Im_old)):
        age = state["Im"][i][0]
        res_num = state["Im"][i][1]
        cluster_num = state["Im"][i][2]

        new_res_num = res_num + rand.normalvariate(0, recover_res_max)
        new_res_num = max(0, min(1, new_res_num))

        if age >= immunity_period:
            Hi_next.append([1, new_res_num, cluster_num, 0, 0]) # return to hibernation
        else:
            Im_next.append([age + 1, res_num, cluster_num, 0, 0])

    # --------------------
    # Calculate net change
    # --------------------
    parents = state["Ot"] + state["Im"]
    for parent in parents:
        r = rand.uniform(0,1)
        if Te:
            p_netchange = 1 - np.exp(- lambda_sum)
            if r <= p_netchange:
                parent_res_num = parent[1]
                cluster_num = parent[2]
                child_res_num = parent_res_num + rand.normalvariate(0, birth_res_max)
                child_res_num = max(0, min(1, child_res_num))

                Ot_next.append([1, child_res_num, cluster_num, 0, 0])
        elif not Te:
            p_netchange = 1 - np.exp(- lambda_win)
            if r <= p_netchange:
                parent_res_num = parent[1]
                cluster_num = parent[2]
                child_res_num = parent_res_num + rand.normalvariate(0, birth_res_max)
                child_res_num = max(0, min(1, child_res_num))

                Ot_next.append([1, child_res_num, cluster_num, 0, 0])
        
    return {
        "Hi": Hi_next,
        "Ot": Ot_next,
        "In": In_next,
        "Im": Im_next,
        "De": De,
    }