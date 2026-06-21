# workign parameters

# ------------------------------------------
# hibernacula-INDEPENDENT initial conditions
# ------------------------------------------

# probabilities
p_infected = 0.01                           # chance a hibernating bat gets infected (given that PD is on) on any given day
p_dead = 0.01                               # chance that an infected bat dies on any given day
p_recover = 1-(1-(1-p_dead)**30)**(1/30)    # CONFIDENT # chance of recovering and going back into hibernation on any given day
p_awake = 0.08                              # OKAY # chance of a waking bat arousing a hibernating bat from torpor on any given day
p_hibernate = 0.5                           # CONFIDENT # chance of a bat switching between hibernating and not (given that Te switches) on any given day
p_netchange = 0.000215                      # CONFIDENT # chance of new bat due to immigration/birth per day
res_num = 0                                 # CONFIDENT # starting resistance for bats in the hibernaculum

immunity_period = 0     # DEPRECATED DO NOT USE # number of days spent in recovery before re-infection is possible
contact_rate = 10       # population-dependent rate of contact btwn health bat and PD infected bat or surface

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
win_length = 120            # CONFIDENT # length of winter season in Nebraska mines
