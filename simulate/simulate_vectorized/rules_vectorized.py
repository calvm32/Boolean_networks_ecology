import random as rand
import numpy as np

def update_environment(state, agg, parameters):

    # ---------------------------
    # Retrieve parameters & nodes
    # ---------------------------

    Te, Hu, PD, Re = state["Te"], state["Hu"], state["PD"], state["Re"]
    Ot_any, Hi_any, In_any, Im_any = agg["Ot_any"], agg["Hi_any"], agg["In_any"], agg["Im_any"]
    Ot_sum, Hi_sum, In_sum, Im_sum = agg["Ot_sum"], agg["Hi_sum"], agg["In_sum"], agg["Im_sum"]
    delta = parameters["delta"]

    Re_next = Re
    Hu_next = Hu
    PD_next = PD

    # --------------------------
    # Update environmental nodes
    # --------------------------

    if Re == 1 and Te == 1: Hu_next = 1 # rule 1
    if Te == 1: Re_next = 1  # rule 2

    PD_next = (1-delta)*PD + (In_sum)/(Ot_sum + Hi_sum + In_sum + Im_sum)

    return {
        "Re": Re_next,
        "Hu": Hu_next,
        "PD": PD_next,
        "Te": Te
    }

def update_individuals(state, env, parameters):

    # ---------------------------
    # Retrieve parameters & nodes
    # ---------------------------

    Te, Hu, PD, Re = state["Te"], state["Hu"], state["PD"], state["Re"]
    inf_alpha, inf_beta, delta = parameters["inf_alpha"], parameters["inf_beta"], parameters["delta"]
    T_inf, T_TBD, T_AD, T_seasonal, win_length = parameters["T_inf"], parameters["T_TBD"], parameters["T_AD"], parameters["T_seasonal"], parameters["win_length"]
    lambda_win, lambda_sum = parameters["lambda_win"], parameters["lambda_sum"]
    T_im, birth_res_max, recover_res_max = parameters["T_im"], parameters["res_max"], parameters["recover_resistance_max"]

    Hi, Ot, In, Im, De = state["Hi"], state["Ot"], state["In"], state["Im"], state["De"]
    parents = np.vstack((Ot, Im)) # reproducing population

    # -------------
    # unpack arrays
    # -------------

    Hi_active = Hi[:, 0] 
    Hi_res = Hi[:, 1] 
    Hi_cluster = Hi[:, 2] 
    
    Ot_active = Ot[:, 0] 
    Ot_res = Ot[:, 1] 
    Ot_cluster = Ot[:, 2] 
    Ot_check = Ot[:, 4] 
    
    In_active = In[:, 0] 
    In_res = In[:, 1] 
    In_cluster = In[:, 2] 
    In_mu = In[:, 3] 
    
    Im_age = Im[:, 0] 
    Im_res = Im[:, 1] 
    Im_cluster= Im[:, 2]

    # compute population counts
    nHi, nOt, nIn, nIm = len(Hi), len(Ot), len(In), len(Im)
    total = nHi + nOt + nIn + nIm

    # ---------
    # Update Hi
    # --------- 
    rHi = np.random.random(nHi) 

    # rule 3: Hi -> In
    infection_weight = nIn/total if total else 0
    lambda_i = infection_weight*Hi_cluster
    K_i = np.random.poisson(lambda_i) 
    p_i = np.random.beta(inf_alpha, inf_beta, size=nHi) 
    p_bat = np.minimum( 1.0, (1 - (1 - p_i) ** K_i) * (1 - Hi_res) ) 
    p_env = PD/(1+PD)
    p_infected = 1 - (1 - p_bat) * (1 - p_env)

    infect_mask = (Hi_active == 1) & (Te == 0) & (rHi < p_infected) 
    In_from_Hi = Hi[infect_mask].copy() 
    In_from_Hi[:, 3] = np.random.gamma( shape=2.0, scale=1.0, size=infect_mask.sum() ) 

    # rule 4: Hi -> Ot
    # from periodic bouts
    p_bouts = 1/T_TBD
    awake_mask = (Hi_active == 1) & (Te == 0) & (~infect_mask) & (rHi < p_bouts) 
    Ot_bouts_from_Hi = Hi[awake_mask].copy() 
    Ot_bouts_from_Hi[:, 4] = 1 

    # from seasonal cycles
    p_seasonal = 1/T_seasonal
    season_awake_mask = (Hi_active == 1) & (Te == 1) & (~infect_mask) & (~awake_mask) & (rHi < p_seasonal) 
    Ot_season_from_Hi = Hi[season_awake_mask].copy() 

    # else
    stay_Hi_mask = ~(infect_mask | awake_mask | season_awake_mask) 
    Hi_stay = Hi[stay_Hi_mask].copy() 
    Hi_stay[:, 3:] = 0 

    # ---------
    # Update Ot
    # --------- 
    rOt = np.random.random(nOt) 
    
    # rule 5: Ot -> De
    dead_mask = (~Te) & (Ot_active == 1) & (Re == 0) 
    De_from_Ot = Ot[dead_mask].copy() 
    De += De_from_Ot.shape[0]

    # rule 6: Ot -> Hi
    # from periodic bouts
    p_bouts = 1/T_AD
    torpor_mask = (Ot_active == 1) & (Te == 0) & (~dead_mask) & (rOt < p_bouts) 
    Hi_bouts_from_Ot = Ot[torpor_mask].copy() 
    Hi_bouts_from_Ot[:, 4] = 0 

    # from seasonal cycles
    season_torpor_mask = (Ot_active == 1) & (Te == 0) & (~dead_mask) & (~torpor_mask) & (rOt < p_seasonal) 
    Ot_season_from_Hi = Hi[season_torpor_mask].copy() 

    # else
    stay_Ot_mask = ~(dead_mask | torpor_mask | season_torpor_mask) 
    Ot_stay = Ot[stay_Ot_mask].copy() 
    Ot_stay[:, 3:] = 0 

    # ---------
    # Update In
    # --------- 
    rIn = np.random.random(nIn) 
    
    # rule 7: In -> De
    dead_mask1 = (~Te) & (In_active == 1) & (Re == 0) 
    De_from_In1 = In[dead_mask1].copy() 
    De += De_from_In1.shape[0]

    # rule 8: In -> De
    p_dead = (1 - np.exp(-In_mu)) * (1 + Re) 
    dead_mask2 = (In_active == 1) & (~dead_mask1) & (rIn < p_dead)  
    De_from_In2 = In[dead_mask2].copy() 
    De += De_from_In2.shape[0]

    # rule 9: In -> Im
    p_recover = 1 / T_inf 
    recover_mask = (In_active == 1) & (~dead_mask1) & (dead_mask2) & (rIn < p_dead)     
    Im_from_In = In[recover_mask].copy()
    Im_from_In[:, 0] = 0 # start immunity counter at 0

    # else
    stay_In_mask = ~(dead_mask1 | dead_mask2 | recover_mask) 
    In_stay = In[stay_In_mask].copy() 
    In_stay[:, 3:] = 0

    # ---------
    # Update Im
    # --------- 
    
    # rule 10: Im -> Hi
    Im_res_new = np.clip( Im_res + np.random.normal(0, parameters["recover_resistance_max"], nIm), 0, 1 ) 
    end_immunity_mask = Im_age >= parameters["T_im"] 
    
    Hi_from_Im = Im[end_immunity_mask].copy() 
    Hi_from_Im[:, 0] = 1 
    Hi_from_Im[:, 1] = Im_res_new[end_immunity_mask] 

    # else
    stay_Im_mask = ~(end_immunity_mask) 
    Im_stay = Im[stay_Im_mask].copy() 
    Im_stay[:, 0] += 1 

    # ----------
    # Net change
    # ----------

    # rule 11: net change
    # net change constant based on season
    if Te:
        p_netchange = 1 - np.exp(- lambda_sum)
    else:
        p_netchange = 1 - np.exp(- lambda_win)


    rP = np.random.random(len(parents)) 
    parents = np.vstack((Ot, Im)) 
    Ot_births = parents[rP < p_netchange].copy() 
    Ot_births[:, 0] = 1 
    Ot_births[:, 1] = np.clip( Ot_births[:, 1] + np.random.normal(0, parameters["res_max"], len(Ot_births)), 0, 1 )
    
    # -------------------------
    # Rebuild arrays and return
    # -------------------------

    Hi_next = np.vstack([Hi_stay, Hi_from_Im])
    Ot_next = np.vstack([Ot_stay, Ot_bouts_from_Hi, Ot_season_from_Hi, Ot_births])
    In_next = np.vstack([In_stay, In_from_Hi])
    Im_next = np.vstack([Im_stay, Im_from_In])
        
    return {
        "Hi": Hi_next,
        "Ot": Ot_next,
        "In": In_next,
        "Im": Im_next,
        "De": De,
    }