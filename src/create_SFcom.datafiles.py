import sys
import pandas as pd
import numpy as np


# Some hardwired parameters:
# Col_Max_ell = [40,60,80]
Col_Max_ell = [80]

# Col_N_space = [40,100,200]
Col_N_space = [150]

N_time = 2

N_Fourier = 2

field_spin = 0.0

mass_ratio = 0.01

particle_charge = 1.0

Col_e_orbit = [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 0.99]

# Col_p_orbit = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0, 50.0]
Col_p_orbit = np.zeros(25)
for nn in range(0, 25, 1):
    Col_p_orbit[nn] = 40.0 + (nn + 1) * (80.0 - 40.0) / 25.0

chi_p_ini = 0.0

phi_p_ini = 0.0

rho_size_domains = 50.0

q_transition = 1.5

s_transition = 1.5


# Takes the name of the Parameter File to be created:
filename = sys.argv[1]


# Creating Parameter File
parfile = open(filename, "w")

parfile.write(
    "Max_ell,N_space,N_time,N_Fourier,field_spin,mass_ratio,particle_charge,e_orbit,p_orbit,chi_p_ini,phi_p_ini,rho_size_domains,q_transition,s_transition\n"
)

parfile.close()

for ell_max in Col_Max_ell:
    for n_col in Col_N_space:
        for e_orb in Col_e_orbit:
            for p_orb in Col_p_orbit:
                text = f"{ell_max:2d},{n_col:3d},{N_time:1d},{N_Fourier:1d},{field_spin:.2f},{mass_ratio:.2f},\
                        {particle_charge:.1f},{e_orb:.4f},{p_orb:.1f},{chi_p_ini:.1f},{phi_p_ini:.1f},\
                        {rho_size_domains:.1f},{q_transition:.2f},{s_transition:.2f}\n"
                parfile = open(filename, "a")

                parfile.write(text)
                parfile.close()
