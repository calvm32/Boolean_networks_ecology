# -------------------------
# set up initial population
# -------------------------

# first select number of bats belonging to each species
tricolor_num = 100
tricolor_cluster_sizeMIN = 1
tricolor_cluster_sizeMAX = 2

bigbrown_num = 0
bigbrown_cluster_sizeMIN = 1
bigbrown_cluster_sizeMAX = 9

# hibernating non-infected bats of each species
Hi_list = [[tricolor_num, tricolor_cluster_sizeMIN, tricolor_cluster_sizeMAX], 
           [bigbrown_num, bigbrown_cluster_sizeMIN, bigbrown_cluster_sizeMAX]] 

fraction_infected = 0   # choose in [0, 1]

# NOTICE : the remaining populations (Ot, Im) all start with 0 inhabitants
# NOTICE : resistance starts at 0 for every bat

# ---------------------------
# system-governing parameters
# ---------------------------

# INFECTION PATHWAYS
inf_alpha, inf_beta = 5, 2                  # infected variables for beta distribution
                                            # chance a hibernating bat gets infected (given that PD is on) on any given day
                                            # low: alpha = 1, beta = 10
                                            # moderate: alpha = 2, beta = 5
                                            # high: alpha = 5, beta = 2

delta = 0.05                                # P. destructans decay rate, considered in [0.005, 0.03]
# DEATH OR RECOVERY PATHWAYS
T_inf = 30                                  # approximate time in dayseach bat spends infirm before recovering or dying, 
                                            # considered in [10, 40]

# BOUT and SEASONAL HIBERNATING PATHWAYS
T_TBD = 4.1                                 # CONFIDENT # length of torpor bout in days, 
                                            # considered in [3.9, 4.3] for tricolored bats
T_AD = 88.5/1440                            # CONFIDENT # length of arousal bout in days, 
                                            # considered in [1.74166, 5.63333] for tricolored bats
T_seasonal = 59                             # CONFIDENT # approx. transition time in days between hibernating and not
                                            # considered in 10-40 maybe?
win_length = 161                            # CONFIDENT # length of winter season in days in Nebraska mines
                                            # considered in 5-7 months, depending on transition period T_seasonal
win_start = 290                             # CONFIDENT # approximate day in calendar year that Te : 1 -> 0

# BAT IN/OUT FLUX
lambda_win = 0                              # CONFIDENT # population growth value during winter, 
                                            # considered in [0, 0.01] 
lambda_sum = 0.00028895065208               # CONFIDENT # population growth value during summer,
                                            # considered in [0.01, 0.1] 

# -----------------
# types of immunity
# -----------------

res_max = 0.2                               # hereditary resistance of newborn, corresp. w/ rand.normalvariate(0, X)
k_imm, theta_imm = 1, 1                     # number of days spent in recovery before re-infection is possible
                                            # corresp. w/ Gamma(k_imm, theta_imm)
res_gain = 0.02                             # resistance AFTER recovery


# -----------------
# -----------------

"""

New best: 379.54166666666663    'T_seasonal': 53.03145102211801, 'win_length': 171.8073545805218, 'win_start': 278.92671603576855, 'lambda_win': 0, 'lambda_sum': 0.00017711957391318124, 'res_gain': 0.02, 'res_max': 0.2, 'k_imm': 1, 'theta_imm': 1}
New best: 146.33333333333334    'T_seasonal': 41.21787049830048, 'win_length': 190.3327411105248, 'win_start': 269.08553986430616, 'lambda_win': 0, 'lambda_sum': 0.00015441617617432672, 'res_gain': 0.02, 'res_max': 0.2, 'k_imm': 1, 'theta_imm': 1}
New best: 132.375               'T_seasonal': 43.460387151848266, 'win_length': 197.57926135060328, 'win_start': 261.508501689169, 'lambda_win': 0, 'lambda_sum': 0.0001676431683071884, 'res_gain': 0.02, 'res_max': 0.2, 'k_imm': 1, 'theta_imm': 1}
New best: 127.54166             'T_seasonal': np.float64(47.40178558756894), 'win_length': np.float64(170.03850624815655), 'win_start': np.float64(287.8314551381096), 'lambda_sum': np.float64(0.0001605245540170072)}
New best: 116.375               'T_seasonal': np.float64(59.21406004055691), 'win_length': np.float64(161.27930678762138), 'win_start': np.float64(289.9954597398685), 'lambda_sum': np.float64(0.0002889506520826776)}
New best: 131.08333333333331    'T_seasonal': 58.75370628407185, 'win_length': 155.30960681369473, 'win_start': 293.137360917269, 'lambda_win': 0, 'lambda_sum': 0.000271915055457734, 'res_gain': 0.02, 'res_max': 0.2, 'k_imm': 1, 'theta_imm': 1}
New best: 151.70833333333331    'T_seasonal': 59.21406004055691, 'win_length': 161.27930678762138, 'win_start': 289.9954597398685, 'lambda_win': 0, 'lambda_sum': 0.0002889506520826776, 'res_gain': 0.02, 'res_max': 0.2, 'k_imm': 1, 'theta_imm': 1}

"""
