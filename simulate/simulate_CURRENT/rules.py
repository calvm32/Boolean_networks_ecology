import random as rand
import numpy as np

def update_environment(state, agg, parameters, t):
    Te, Hu, PD, Re = state["Te"], state["Hu"], state["PD"], state["Re"]
    Ot_any, Hi_any, In_any, Im_any = agg["Ot_any"], agg["Hi_any"], agg["In_any"], agg["Im_any"]
    Ot_sum, Hi_sum, In_sum, Im_sum = agg["Ot_sum"], agg["Hi_sum"], agg["In_sum"], agg["Im_sum"]
    delta, win_start, win_length = parameters["delta"], parameters["win_start"], parameters["win_length"]

    Re_next = Re
    Hu_next = Hu
    PD_next = PD

    # -----------
    # apply rules
    # -----------
    if Re == 1 and Te == 1: Hu_next = 1
    if Te == 1: Re_next = 1 

    total = Ot_sum + Hi_sum + In_sum + Im_sum

    PD_next = (1-delta)*PD + In_sum / total if total else (1-delta)*PD

    # update Te
    day = t % 365
    winter_day = (day - win_start) % 365 # shift calendar
    Te_next = 0 if winter_day < win_length else 1

    return {
        "Hu": Hu_next,
        "PD": PD_next,
        "Re": Re_next,
        "Te": Te_next
    }

def update_individuals(state, env, parameters, t):
    Te, Hu, PD, Re = state["Te"], state["Hu"], state["PD"], state["Re"]
    inf_alpha, inf_beta, delta = parameters["inf_alpha"], parameters["inf_beta"], parameters["delta"]
    T_inf, T_TBD, T_AD, T_seasonal, win_length, win_start = parameters["T_inf"], parameters["T_TBD"], parameters["T_AD"], parameters["T_seasonal"], parameters["win_length"], parameters["win_start"]
    lambda_win, lambda_sum = parameters["lambda_win"], parameters["lambda_sum"]
    res_gain, res_max, k_imm, theta_imm = parameters["res_gain"], parameters["res_max"], parameters["k_imm"], parameters["theta_imm"]
    secondary_cases = state["SC"]

    # go over OLD STATE
    Hi_old = state["Hi"]
    Ot_old = state["Ot"]
    In_old = state["In"]
    Im_old = state["Im"]
    De = state["De"]

    # re-add everyone who stays the same
    Hi_next = []
    Ot_next = []
    In_next = []
    Im_next = []
    
    # configure totals for weighting parameters
    total_inhabitants = (
        len(state["Hi"]) +
        len(state["Ot"]) +
        len(state["In"]) +
        len(state["Im"])
    )

    weight = len(state["In"]) / total_inhabitants if total_inhabitants else 0

    # environmental PD exposure
    p_env = PD/(1+PD)

    # net change
    if Te:
        p_netchange = 1 - np.exp(- lambda_sum)
    else:
        p_netchange = 1 - np.exp(- lambda_win)

    # seasonal exit hibernation
    if Te == 1:
        d = (t - win_start - win_length) % 365 # 0 when 
    else:
        d = (t - win_start) % 365
    
    if T_seasonal - d <= 0: # avoid division by 0
        p_seasonal = 1
    else:
        p_seasonal = 1/(T_seasonal-d)

    active_inf = len(In_old)
    active_IC = sum(1 for bat in In_old if bat[6] == 1)

    # ---------
    # Update Hi
    # ---------
    for i in range(len(Hi_old)):
        Hi, res_num, cluster_num, _, _, _, _ = state["Hi"][i]
        r = rand.uniform(0,1)

        # bat-to-bat contact WNS exposure
        lambda_i = weight*cluster_num
        K_i = np.random.poisson(lambda_i)

        p_i = np.random.beta(parameters["inf_alpha"], parameters["inf_beta"])
        p_bat = (1 - (1-p_i)**K_i) * (1-res_num)
        p_bat = min(1, p_bat)

        p_infected = 1 - (1-p_bat)*(1-p_env)

        # if infected, determine how long until recovery or death
        mu_mean = 1/T_inf
        k_dead = 2.0
        theta_dead = mu_mean / k_dead

        mu_i = np.random.gamma(shape=k_dead, scale=theta_dead)

        # bout exit hibernation
        p_bouts = 1/T_TBD if T_AD >= 1 else 0

        if (Hi and Te == 0 and r < p_infected):

            # handle absolutes to prevent np.log(0) errors
            if p_bat == 1 and p_env == 1:
                prob_from_bat = 0.5
            elif p_bat == 1:
                prob_from_bat = 1.0
            elif p_env == 1:
                prob_from_bat = 0.0
            else:
                # separate bat vs. env transmission cases. use "continuous competing risks" approximation 
                # to determine likelihood of p_bat being the transmissio mechanism 
                # GIVEN that the transmission already occurred
                prob_from_bat = np.log(1 - p_bat) / (np.log(1 - p_bat) + np.log(1 - p_env))

                # IF infection came from a bat, attribute it to an index case based on proportion
                if rand.uniform(0,1) < prob_from_bat and active_inf > 0:
                    prob_infector_is_index = active_IC / active_inf
                    if rand.uniform(0,1) < prob_infector_is_index:
                        secondary_cases += 1
                        
            In_next.append([1, res_num, cluster_num, mu_i, 0, 0, 0])

        elif not Te and Hi and r <= p_bouts:
            Ot_next.append([1, res_num, cluster_num, 0, 1, 0, 0]) # mark [...,1] to signify next hibernating rule is p_bouts and not p_seasonal
        elif Te and Hi and r <= p_seasonal:
            Ot_next.append([1, res_num, cluster_num, 0, 0, 0, 0]) 
        else:
            Hi_next.append([Hi, res_num, cluster_num, 0, 0, 0, 0])

    # ----------
    # Update Ot
    # ----------
    for i in range(len(Ot_old)):
        Ot, res_num, cluster_num, _, check, _, _ = state["Ot"][i]
        r = rand.uniform(0, 1)

        # bout hibernation
        # check if hibernated before, i.e. if the next hibernating rule is p_bouts and not p_seasonal
        p_bouts = 1/T_AD

        if Ot and not Te and not Re:
            De += 1
        elif not Te and Ot and check and r <= p_bouts:
            Hi_next.append([1, res_num, cluster_num, 0, 0, 0, 0]) 
        elif not Te and Ot and r <= p_seasonal:
            Hi_next.append([1, res_num, cluster_num, 0, 0, 0, 0]) 
        else:
            Ot_next.append([Ot, res_num, cluster_num, 0, 0, 0, 0])

    # ---------
    # Update In
    # ---------
    for i in range(len(In_old)):
        In, res_num, cluster_num, mu_i, _, _, is_IC = state["In"][i]
        r1 = rand.uniform(0, 1)
        r2 = rand.uniform(0, 1)

        # probability of death
        p_dead = (1 - np.exp(-mu_i))*(1 + Re)
        p_recover =  1/T_inf

        T_im = int(np.random.gamma(shape=k_imm, scale=theta_imm))

        if In and not Te and not Re:
            De += 1
        elif In and r1 <= p_dead:
            De += 1
        elif In and r2 <= p_recover:
            Im_next.append([In, res_num, cluster_num, 0, 0, T_im, 0]) # start recovery counter
        else:
            In_next.append([In, res_num, cluster_num, 0, 0, 0, is_IC]) # track index cases (IC)

    # ---------
    # Update Im
    # ---------
    for i in range(len(Im_old)):
        In, res_num, cluster_num, _, _, T_im_i, _ = state["Im"][i]

        if T_im_i <= 0:
            # immunity improves with infection
            new_res_num = res_num + res_gain*(1-res_num)
            new_res_num = max(0, min(1, new_res_num))

            Hi_next.append([In, new_res_num, cluster_num, 0, 0, 0, 0]) # return to hibernation
        else:
            Im_next.append([In, res_num, cluster_num, 0, 0, T_im_i - 1, 0])

    # --------------------
    # Calculate net change
    # --------------------
    parents = state["Ot"] + state["Im"]
    for parent in parents:
        r = rand.uniform(0,1)
        _, parent_res_num, cluster_num, _, _, _, _ = parent

        if Te and r <= p_netchange:
            child_res_num = parent_res_num + rand.normalvariate(0, res_max)
            child_res_num = max(0, min(1, child_res_num))

            Ot_next.append([1, child_res_num, cluster_num, 0, 0, 0, 0])
        elif not Te and r <= p_netchange:
            child_res_num = parent_res_num + rand.normalvariate(0, res_max)
            child_res_num = max(0, min(1, child_res_num))

            Ot_next.append([1, child_res_num, cluster_num, 0, 0, 0, 0])
        
    return {
        "Hi": Hi_next,
        "Ot": Ot_next,
        "In": In_next,
        "Im": Im_next,
        "De": De,
        "SC": secondary_cases
    }