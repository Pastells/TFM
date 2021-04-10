import pandas as pd
import numpy as np
from Schwarzschild import *
from hyperboloidal_compactification_tanh import *


class Computational_Grid:

    # Initializer / Instance Attributes
    def __init__(self, DF, run):

        # Basic Parameters
        self.run = run
        self.N_Regular_Domains = DF.N_Regular_Domains[self.run]
        self.N_space = DF.N_space[self.run]
        self.ell_max = DF.Max_ell[self.run]

        self.field_spin = DF.field_spin[self.run]
        self.sigma_spin = 1.0 - (self.field_spin) ** 2

        self.e_orbit = DF.e_orbit[self.run]
        self.p_orbit = DF.p_orbit[self.run]

        # Domain Boundaries
        self.rho = self.DomainBoundaries(self.N_Regular_Domains)

        # Assigning values to the Domain Boundaries
        self.rho.rho_IDS = DF.rho_IDS[self.run]  # Before this was called rho_size_domains

        self.rho.peri = r_schwarzschild_to_r_tortoise((self.p_orbit) / (1.0 + self.e_orbit))
        self.rho.apo = r_schwarzschild_to_r_tortoise((self.p_orbit) / (1.0 - self.e_orbit))

        # self.rho.H_RD_R[0] = self.rho.peri
        # self.rho.H_RD_L[0] = self.rho.peri - self.rho.rho_IDS
        # for nd in range(1, self.N_Regular_Domains, 1):
        #     self.rho.H_RD_R[nd] = self.rho.H_RD_L[nd-1]
        #     self.rho.H_RD_L[nd] = self.rho.H_RD_L[nd-1] - (2**nd)*self.rho.rho_IDS

        self.rho.H_RD_R[self.N_Regular_Domains - 1] = self.rho.peri
        self.rho.H_RD_L[self.N_Regular_Domains - 1] = self.rho.peri - self.rho.rho_IDS
        for nd in range(self.N_Regular_Domains - 2, -1, -1):
            self.rho.H_RD_R[nd] = self.rho.H_RD_L[nd + 1]
            self.rho.H_RD_L[nd] = self.rho.H_RD_L[nd + 1] - (2 ** (self.N_Regular_Domains - nd - 1)) * self.rho.rho_IDS

        self.rho.H_S = self.rho.H_RD_L[0]
        self.rho.H_C = self.rho.H_S - (2 ** (self.N_Regular_Domains - 1)) * self.rho.rho_IDS
        self.rho.horizon = self.rho.H_C - (2 ** (self.N_Regular_Domains - 1)) * self.rho.rho_IDS

        self.rho.I_RD_L[0] = self.rho.apo
        self.rho.I_RD_R[0] = self.rho.apo + self.rho.rho_IDS
        for nd in range(1, self.N_Regular_Domains, 1):
            self.rho.I_RD_L[nd] = self.rho.I_RD_R[nd - 1]
            self.rho.I_RD_R[nd] = self.rho.I_RD_R[nd - 1] + (2 ** nd) * self.rho.rho_IDS

        self.rho.I_S = self.rho.I_RD_R[self.N_Regular_Domains - 1]
        self.rho.I_C = self.rho.I_S + (2 ** (self.N_Regular_Domains - 1)) * self.rho.rho_IDS
        self.rho.infinity = self.rho.I_C + (2 ** (self.N_Regular_Domains - 1)) * self.rho.rho_IDS

        self.rho.H_L = self.rho.horizon
        self.rho.H_R = self.rho.H_C
        self.rho.H_tr_L = self.rho.H_C
        self.rho.H_tr_R = self.rho.H_S
        self.rho.H_pp_L = self.rho.peri
        self.rho.H_pp_R = self.rho.apo

        self.rho.I_pp_L = self.rho.peri
        self.rho.I_pp_R = self.rho.apo
        self.rho.I_tr_L = self.rho.I_S
        self.rho.I_tr_R = self.rho.I_C
        self.rho.I_L = self.rho.I_C
        self.rho.I_R = self.rho.infinity

        # Transition Function/Compactification Parameters
        self.TF = self.TransitionFunction()
        self.TF.q = DF.q_transition[self.run]
        self.TF.s = DF.s_transition[self.run]

        # Spectral Grid and First-Order Derivative Matrix
        self.X = np.zeros(self.N_space + 1)
        self.D1 = np.zeros((self.N_space + 1, self.N_space + 1))

        # Physical Domains using the Coordinate 'rho'
        self.rho_H = np.zeros(self.N_space + 1)
        self.rho_H_tr = np.zeros(self.N_space + 1)
        self.rho_H_RD = np.zeros((self.N_Regular_Domains, self.N_space + 1))
        self.rho_H_pp = np.zeros(self.N_space + 1)
        self.rho_I_pp = np.zeros(self.N_space + 1)
        self.rho_I_RD = np.zeros((self.N_Regular_Domains, self.N_space + 1))
        self.rho_I_tr = np.zeros(self.N_space + 1)
        self.rho_I = np.zeros(self.N_space + 1)

        # Physical Domains using the Coordinate 'r*' (rs: Tortoise Coordinate)
        self.rs_H = np.zeros(self.N_space + 1)
        self.rs_H_tr = np.zeros(self.N_space + 1)
        self.rs_H_RD = np.zeros((self.N_Regular_Domains, self.N_space + 1))
        self.rs_H_pp = np.zeros(self.N_space + 1)
        self.rs_I_pp = np.zeros(self.N_space + 1)
        self.rs_I_RD = np.zeros((self.N_Regular_Domains, self.N_space + 1))
        self.rs_I_tr = np.zeros(self.N_space + 1)
        self.rs_I = np.zeros(self.N_space + 1)

        # Physical Domains using the Coordinate 'r' (Schwarzschild Coordinate)
        self.r_H = np.zeros(self.N_space + 1)
        self.r_H_tr = np.zeros(self.N_space + 1)
        self.r_H_RD = np.zeros((self.N_Regular_Domains, self.N_space + 1))
        self.r_H_pp = np.zeros(self.N_space + 1)
        self.r_I_pp = np.zeros(self.N_space + 1)
        self.r_I_RD = np.zeros((self.N_Regular_Domains, self.N_space + 1))
        self.r_I_tr = np.zeros(self.N_space + 1)
        self.r_I = np.zeros(self.N_space + 1)

        # Transition/Compactification Function 'H' at the different Domains
        self.H_H = np.zeros(self.N_space + 1)
        self.H_H_tr = np.zeros(self.N_space + 1)
        self.H_H_RD = np.zeros((self.N_Regular_Domains, self.N_space + 1))
        self.H_H_pp = np.zeros(self.N_space + 1)
        self.H_I_pp = np.zeros(self.N_space + 1)
        self.H_I_RD = np.zeros((self.N_Regular_Domains, self.N_space + 1))
        self.H_I_tr = np.zeros(self.N_space + 1)
        self.H_I = np.zeros(self.N_space + 1)

        # First-Order Derivative of the Transition/Compactification Function 'H', 'DH', at the different Domains
        self.DH_H = np.zeros(self.N_space + 1)
        self.DH_H_tr = np.zeros(self.N_space + 1)
        self.DH_H_RD = np.zeros((self.N_Regular_Domains, self.N_space + 1))
        self.DH_H_pp = np.zeros(self.N_space + 1)
        self.DH_I_pp = np.zeros(self.N_space + 1)
        self.DH_I_RD = np.zeros((self.N_Regular_Domains, self.N_space + 1))
        self.DH_I_tr = np.zeros(self.N_space + 1)
        self.DH_I = np.zeros(self.N_space + 1)

        # Regge Wheeler Potential at the different Domains for all values of the Harmonic Number 'ell'
        self.V_RW_H = np.zeros((self.ell_max + 1, self.N_space + 1))
        self.V_RW_H_tr = np.zeros((self.ell_max + 1, self.N_space + 1))
        self.V_RW_H_RD = np.zeros((self.N_Regular_Domains, self.ell_max + 1, self.N_space + 1))
        self.V_RW_H_pp = np.zeros((self.ell_max + 1, self.N_space + 1))
        self.V_RW_I_pp = np.zeros((self.ell_max + 1, self.N_space + 1))
        self.V_RW_I_RD = np.zeros((self.N_Regular_Domains, self.ell_max + 1, self.N_space + 1))
        self.V_RW_I_tr = np.zeros((self.ell_max + 1, self.N_space + 1))
        self.V_RW_I = np.zeros((self.ell_max + 1, self.N_space + 1))

        # Computation of the Spectral Chebyshev-Lobatto Grid for the PseudoSpectral Collocation Method
        if np.remainder(self.N_space, 2) == 0:
            self.N_points_half = (self.N_space) // 2
            self.flag_even = "y"
        else:
            self.N_points_half = (self.N_space - 1) // 2
            self.flag_even = "n"

        self.X[0] = -1.0
        self.X[self.N_space] = 1.0

        if self.flag_even == "y":
            for k in range(1, self.N_points_half, 1):
                self.X[k] = -np.cos(k * np.pi / (self.N_space))
                self.X[self.N_space - k] = -self.X[k]
            self.X[self.N_points_half] = 0.0
        else:
            for k in range(1, self.N_points_half + 1, 1):
                self.X[k] = -np.cos(k * np.pi / (self.N_space))
                self.X[self.N_space - k] = -self.X[k]

        # Coefficients c_k [c_0=c_N=2, c_i=1 for i=1...N-1]
        # Matrix CC(k,l) = c_k/c_l
        CC = np.ones((self.N_space + 1, self.N_space + 1))

        CC[0, :] = 2.0 * (CC[0, :])
        CC[self.N_space, :] = 2.0 * (CC[self.N_space, :])
        CC[:, 0] = 0.5 * (CC[:, 0])
        CC[:, self.N_space] = 0.5 * (CC[:, self.N_space])

        # Computation of the non-diagonal elements of the 1st-order Differentiation Matrix
        for k in range(0, self.N_space + 1, 1):
            for l in range(k + 1, self.N_space + 1, 1):
                if np.remainder(k + l, 2) == 0:
                    signo = 1.0
                else:
                    signo = -1.0

                DX = signo / (self.X[k] - self.X[l])
                self.D1[k, l] = CC[k, l] * DX
                self.D1[l, k] = -CC[l, k] * DX

        # Computation of the diagonal elements of the 1st-order Differentiation Matrix
        for k in range(0, self.N_space + 1, 1):
            self.D1[k, k] = 0.0

            for l in range(0, self.N_space + 1, 1):
                if l != k:
                    self.D1[k, k] = self.D1[k, k] - self.D1[k, l]

        # Physical domain (rho coordinate). We are considering SIX domains.  Boundary coordinates:
        #  - rho.horizon : rho location of the BH horizon
        #  - rho.H_Compactified : End of the Transition Region towards the Horizon/Beginning of the Compactified Region
        #  - rho.H_Standard : End of the Standard Region towards the Horizon/Beginning of the Transition (compactified) Region
        #  - rho.H_RD_L: Left Boundary of the Regular Domains in the Horizon Region
        #  - rho.H_RD_R: Right Boundary of the Regular Domains in the Horizon Region
        #  - rho.peri : rho location of the orbital Pericenter
        #  - rho.apo : rho location of the orbital Apocenter
        #  - rho.I_RD_L: Left Boundary of the Regular Domains in the Infinity Region
        #  - rho.I_RD_R: Right Boundary of the Regular Domains in the Infinity Region
        #  - rho.I_Standard : End of the Standard Region towards Spatial Infinity/Beginning of the Transition (compactified) Region
        #  - rho.I_Compactified : End of the Transtion Region towards Spatial Infinity/Beginning of the Compactified Region
        #  - rho.infinity : rho location of spatial infinity
        #
        # SubDomains in the rho coordinate:
        #  - rho_H    : rho-Grid that goes from the end of the Transition Region to the Horizon
        #               using the Hyperboloidal Compactification
        #  - rho_H_tr : rho-Grid that covers the transition region to the left of the Particle
        #               (Going to the Horizon) from the non-compactified to the compactified regions
        #  - rho_H_RD : rho-Grids to the left of the Particle that coincide with the r_tortoise
        #               Grid, i.e. the non-compactified Region
        #  - rho_H_pp : rho-Grid covering the Particle radial trajectory, also using the r_tortoise
        #               coordinate [part of the Horizon subDomain]
        #  - rho_I_pp : rho-Grid covering the Particle radial trajectory, also using the r_tortoise
        #               coordinate [part of Spatial Infinity Domain]
        #  - rho_I_RD : rho-Grids to the right of the Particle that coincide with the tortoise Grid,
        #               i.e. the non-compactified Region
        #  - rho_I_tr : rho-Grid that covers the transition region to the right of the Particle
        #               (Going to Spatial Infinity) from the non-compactified to the compactified regions
        #  - rho_I    : rho-Grid that goes from the end of the Transition Region to Spatial Infinity
        #               using the Hyperboloidal Compactification: rho_I(1:N+1)
        #
        # Physical Grid - Compactified Domain reaching the Horizon:
        self.rho_H = 0.5 * ((self.rho.horizon + self.rho.H_R) + (self.rho.H_R - self.rho.horizon) * self.X)

        # Physical Grid - Transition Region Domain going towards the Horizon (it also uses the Compactification Coordinate):
        self.rho_H_tr = 0.5 * ((self.rho.H_tr_L + self.rho.H_tr_R) + (self.rho.H_tr_R - self.rho.H_tr_L) * self.X)

        # Physical Grid - Domains to the Left of the Particle but in vacuum (Going towards the Horizon. They use the Tortoise Coordinate):
        for nd in range(0, self.N_Regular_Domains, 1):
            self.rho_H_RD[nd, :] = 0.5 * (
                (self.rho.H_RD_L[nd] + self.rho.H_RD_R[nd]) + (self.rho.H_RD_R[nd] - self.rho.H_RD_L[nd]) * self.X
            )

        # Physical Grid - Domain covering the particle radial trajectory (part of Horizon domain. It uses the Tortoise Coordinate):
        self.rho_H_pp = 0.5 * ((self.rho.peri + self.rho.apo) + (self.rho.apo - self.rho.peri) * self.X)

        # Physical Grid - Domain covering the particle radial trajectory (part of Spatial Infinity domain. It uses the Tortoise Coordinate):
        self.rho_I_pp = 0.5 * ((self.rho.peri + self.rho.apo) + (self.rho.apo - self.rho.peri) * self.X)

        # Physical Grid - Domains to the Right of the Particle (Going towards Spatial Infinity. They uses the Tortoise Coordinate):
        for nd in range(0, self.N_Regular_Domains, 1):
            self.rho_I_RD[nd, :] = 0.5 * (
                (self.rho.I_RD_L[nd] + self.rho.I_RD_R[nd]) + (self.rho.I_RD_R[nd] - self.rho.I_RD_L[nd]) * self.X
            )

        # Physical Grid - Transition Region Domain going towards Spatial Infinity (it also uses the Compactification Coordinate):
        self.rho_I_tr = 0.5 * ((self.rho.I_tr_L + self.rho.I_tr_R) + (self.rho.I_tr_R - self.rho.I_tr_L) * self.X)

        # Physical Grid - Compactified Domain reaching Spatial Infinity (actually Null Infinity):
        self.rho_I = 0.5 * ((self.rho.I_L + self.rho.infinity) + (self.rho.infinity - self.rho.I_L) * self.X)

        # Physical domain (tortoise coordinate). We are considering SIX domains.
        # SubDomains in the tortoise coordinate:
        #  - rs_H    : rho-Grid that goes from the end of the Transition Region to the Horizon
        #              using the Hyperboloidal Compactification
        #  - rs_H_tr : rho-Grid that covers the transition region to the left of the Particle
        #              (Going to the Horizon) from the non-compactified to the compactified regions
        #  - rs_H_RD : rho-Grids to the left of the Particle that coincide with the r_tortoise
        #              Grid, i.e. the non-compactified Region
        #  - rs_H_pp : rho-Grid covering the Particle radial trajectory, also using the r_tortoise
        #              coordinate [part of the Horizon subDomain]
        #  - rs_I_pp : rho-Grid covering the Particle radial trajectory, also using the r_tortoise
        #              coordinate [part of Spatial Infinity Domain]
        #  - rs_I_RD : rho-Grids to the right of the Particle that coincide with the tortoise Grid,
        #              i.e. the non-compactified Region
        #  - rs_I_tr : rho-Grid that covers the transition region to the right of the Particle
        #              (Going to Spatial Infinity) from the non-compactified to the compactified regions
        #  - rs_I    : rho-Grid that goes from the end of the Transition Region to Spatial Infinity
        #              using the Hyperboloidal Compactification: rho_I(1:N+1)
        #
        # Physical Grid - Compactified Domain reaching the Horizon:
        self.rs_H[0] = -np.inf

        for k in range(1, self.N_space, 1):

            Omega_H = 1.0 - self.rho_H[k] / self.rho.horizon

            self.rs_H[k] = self.rho_H[k] / Omega_H

        Omega_H = 1.0 - self.rho.H_C / self.rho.horizon
        self.rs_H[self.N_space] = self.rho_H[self.N_space] / Omega_H

        # Physical Grid - Transition Region Domain going towards the Horizon [the ONLY one, in the Horizon region, that requires the computation of the transition function]:
        self.rs_H_tr[0] = self.rs_H[self.N_space]

        for k in range(1, self.N_space, 1):

            width = self.rho.H_C - self.rho.H_S

            sigma = 0.5 * (np.pi) * (self.rho_H_tr[k] - self.rho.H_S) / width

            f0 = f_transition(sigma, self.TF)

            Omega_H_tr = 1.0 - f0 * self.rho_H_tr[k] / self.rho.horizon

            self.rs_H_tr[k] = self.rho_H_tr[k] / Omega_H_tr

        self.rs_H_tr[self.N_space] = self.rho_H_tr[self.N_space]

        # Physical Grids - Non-Compactified Domains to the Left of the Particle but in vacuum (Going towards the Horizon): rho and r_star coincide
        self.rs_H_RD = self.rho_H_RD

        # Physical Grid - Non-Compactified Domain covering the particle radial trajectory (part of Horizon domain): rho and r_star coincide
        self.rs_H_pp = self.rho_H_pp

        # Physical Grid - Non-Compactified Domain covering the particle radial trajectory (part of Spatial Infinity domain): rho and r_star coincide
        self.rs_I_pp = self.rho_I_pp

        # Physical Grids - Non-Compactified Domains to the Right of the Particle (Going towards Spatial Infinity): rho and r_star coincide
        self.rs_I_RD = self.rho_I_RD

        # Physical Grid - Transition Region Domain going towards Spatial Infinity [the ONLY one, in the Infinity region, that requires the computation of the transition function]:
        self.rs_I_tr[0] = self.rho_I_tr[0]

        for k in range(1, self.N_space, 1):

            width = self.rho.I_C - self.rho.I_S

            sigma = 0.5 * (np.pi) * (self.rho_I_tr[k] - self.rho.I_S) / width

            f0 = f_transition(sigma, self.TF)

            Omega_I_tr = 1.0 - f0 * self.rho_I_tr[k] / self.rho.infinity

            self.rs_I_tr[k] = self.rho_I_tr[k] / Omega_I_tr

        Omega_I_C = 1.0 - self.rho.I_C / self.rho.infinity
        self.rs_I_tr[self.N_space] = self.rho.I_C / Omega_I_C

        # Physical Grid - Compactified Domain reaching Spatial Infinity:
        self.rs_I[0] = self.rs_I_tr[self.N_space]

        for k in range(1, self.N_space, 1):

            Omega_I = 1.0 - self.rho_I[k] / self.rho.infinity

            self.rs_I[k] = self.rho_I[k] / Omega_I

        self.rs_I[self.N_space] = np.inf

        # Physical domain (Schwarzschild coordinate). We are considering SIX domains.
        # SubDomains in the Schwarzschild coordinate:
        #  r_H:             r-Grid that goes from the end of the Transition Region to the Horizon using the
        #                   Hyperboloidal Compactification
        #  r_H_tr:          r-Grid that covers the transition region to the left of the Particle (Going to
        #                   the Horizon) from the non-compactified to the compactified regions
        #  r_H_RD:          r-Grids to the left of the Particle that coincide with the r_tortoise Grid, i.e.
        #                   the non-compactified Region
        #  r_H_pp:          r-Grid covering the Particle radial trajectory, also using the r_tortoise
        #                   coordinate [part of Horizon Domain]
        #  r_I_pp:          r-Grid covering the Particle radial trajectory, also using the r_tortoise
        #                   coordinate [part of Spatial Infinity Domain]
        #  r_I_RD:          r-Grids to the right of the Particle that coincide with the r_tortoise Grid,
        #                   i.e. the non-compactified Region
        #  r_I_tr:          r-Grid that covers the transition region to the right of the Particle (Going to
        #                   Spatial Infinity) from the non-compactified to the compactified regions
        #  r_I:             r-Grid that goes from the end of the Transition Region to Spatial Infinity using
        #                   the Hyperboloidal Compactification: rs_I(1:N+1)
        #
        # Physical Grid - Compactified Domain reaching the Horizon:
        self.r_H[0] = 2.0
        for k in range(1, self.N_space + 1, 1):
            self.r_H[k] = r_tortoise_to_r_schwarzschild(self.rs_H[k])

        # Physical Grid - Transition Region Domain going towards the Horizon:
        for k in range(0, self.N_space + 1, 1):
            self.r_H_tr[k] = r_tortoise_to_r_schwarzschild(self.rs_H_tr[k])

        # Physical Grids - Non-Compactified Domains to the Left of the Particle but in vacuum (Going towards the Horizon):
        for nd in range(0, self.N_Regular_Domains, 1):
            for k in range(0, self.N_space + 1, 1):
                self.r_H_RD[nd, k] = r_tortoise_to_r_schwarzschild(self.rs_H_RD[nd, k])

        # Physical Grid - Non-Compactified Domain covering the particle radial trajectory (part of Horizon domain):
        for k in range(0, self.N_space + 1, 1):
            self.r_H_pp[k] = r_tortoise_to_r_schwarzschild(self.rs_H_pp[k])

        # Physical Grid - Non-Compactified Domain covering the particle radial trajectory (part of Spatial Infinity domain):
        for k in range(0, self.N_space + 1, 1):
            self.r_I_pp[k] = r_tortoise_to_r_schwarzschild(self.rs_I_pp[k])

        # Physical Grids - Non-Compactified Domains to the Right of the Particle (Going towards Spatial Infinity):
        for nd in range(0, self.N_Regular_Domains, 1):
            for k in range(0, self.N_space + 1, 1):
                self.r_I_RD[nd, k] = r_tortoise_to_r_schwarzschild(self.rs_I_RD[nd, k])

        # Physical Grid - Transition Region Domain going towards Spatial Infinity:
        for k in range(0, self.N_space + 1, 1):
            self.r_I_tr[k] = r_tortoise_to_r_schwarzschild(self.rs_I_tr[k])

        # Physical Grid - Compactified Domain reaching Spatial Infinity:
        for k in range(0, self.N_space, 1):
            self.r_I[k] = r_tortoise_to_r_schwarzschild(self.rs_I[k])

        self.r_I[self.N_space] = np.inf

        #  Computation of the function H and DH ( = dH/drho ) as grid vectors in the six
        #  different Subdomains (from the Horizon Domain to the Spatial Infinity Domain)
        #
        # Compactified Domain reaching the Horizon:
        self.H_H[0] = 1.0
        self.DH_H[0] = 0.0

        for k in range(1, self.N_space, 1):

            Omega_H = 1.0 - self.rho_H[k] / self.rho.horizon

            self.H_H[k] = 1.0 - Omega_H ** 2
            self.DH_H[k] = (2.0 / self.rho.horizon) * Omega_H

        self.H_H[self.N_space] = (self.rho.H_C / self.rho.horizon) * (2.0 - self.rho.H_C / self.rho.horizon)
        self.DH_H[self.N_space] = (2.0 / self.rho.horizon) * (1.0 - self.rho.H_C / self.rho.horizon)

        # Transition Region Domain going towards the Horizon:
        self.H_H_tr[0] = self.H_H[self.N_space]
        self.DH_H_tr[0] = self.DH_H[self.N_space]

        for k in range(1, self.N_space, 1):

            width = self.rho.H_C - self.rho.H_S

            sigma = 0.5 * (np.pi) * (self.rho_H_tr[k] - self.rho.H_S) / width

            jacobian = 0.5 * np.pi / width

            f0 = f_transition(sigma, self.TF)

            Omega_H_tr = 1.0 - f0 * self.rho_H_tr[k] / self.rho.horizon

            f1 = jacobian * f_transition_1st(f0, sigma, self.TF)

            f2 = (jacobian ** 2) * f_transition_2nd(f0, sigma, self.TF)

            Omega_1st = -(f0 + f1 * self.rho_H_tr[k]) / self.rho.horizon

            Omega_2nd = -(2.0 * f1 + f2 * self.rho_H_tr[k]) / self.rho.horizon

            L_H_tr = Omega_H_tr - Omega_1st * self.rho_H_tr[k]

            self.H_H_tr[k] = 1.0 - (Omega_H_tr ** 2) / L_H_tr
            self.DH_H_tr[k] = -(Omega_H_tr / L_H_tr) * (2.0 * Omega_1st + (Omega_H_tr / L_H_tr) * Omega_2nd * self.rho_H_tr[k])

        self.H_H_tr[self.N_space] = 0.0
        self.DH_H_tr[self.N_space] = 0.0

        # Non-Compactified Domains to the Left of the Particle (Going towards the Horizon): Here H = DH = 0 => Nothing to do!

        # Non-Compactified Domain covering the particle radial trajectory (part of Horizon domain): Here H = DH = 0 => Nothing to do!

        # Non-Compactified Domains covering the particle radial trajectory (part of Spatial Infinity domain): Here H = DH = 0 => Nothing to do!

        # Non-Compactified Domain to the Right of the Particle (Going towards Spatial Infinity): Here H = DH = 0 => Nothing to do!

        # Transition Region Domain going towards Spatial Infinity:
        self.H_I_tr[0] = 0.0
        self.DH_I_tr[0] = 0.0

        for k in range(1, self.N_space, 1):

            width = self.rho.I_C - self.rho.I_S

            sigma = 0.5 * (np.pi) * (self.rho_I_tr[k] - self.rho.I_S) / width

            jacobian = 0.5 * np.pi / width

            f0 = f_transition(sigma, self.TF)

            Omega_I_tr = 1.0 - f0 * self.rho_I_tr[k] / self.rho.infinity

            f1 = jacobian * f_transition_1st(f0, sigma, self.TF)

            f2 = (jacobian ** 2) * f_transition_2nd(f0, sigma, self.TF)

            Omega_1st = -(f0 + f1 * self.rho_I_tr[k]) / self.rho.infinity

            Omega_2nd = -(2.0 * f1 + f2 * self.rho_I_tr[k]) / self.rho.infinity

            L_I_tr = Omega_I_tr - Omega_1st * self.rho_I_tr[k]

            self.H_I_tr[k] = 1.0 - (Omega_I_tr ** 2) / L_I_tr
            self.DH_I_tr[k] = -(Omega_I_tr / L_I_tr) * (2.0 * Omega_1st + (Omega_I_tr / L_I_tr) * Omega_2nd * self.rho_I_tr[k])

        self.H_I_tr[self.N_space] = (self.rho.I_C / self.rho.infinity) * (2.0 - self.rho.I_C / self.rho.infinity)
        self.DH_I_tr[self.N_space] = (2.0 / self.rho.infinity) * (1.0 - self.rho.I_C / self.rho.infinity)

        # Compactified Domain reaching Spatial Infinity:
        self.H_I[0] = self.H_I_tr[self.N_space]
        self.DH_I[0] = self.DH_I_tr[self.N_space]

        for k in range(1, self.N_space, 1):

            Omega = 1.0 - self.rho_I[k] / self.rho.infinity

            self.H_I[k] = 1.0 - Omega ** 2
            self.DH_I[k] = (2.0 / self.rho.infinity) * Omega

        self.H_I[self.N_space] = 1.0
        self.DH_I[self.N_space] = 0.0

        # Computing the Regge-Wheeler Potential as a Grid Function:
        for ll in range(0, self.ell_max + 1, 1):

            # Compactified Domain reaching the Horizon:
            self.V_RW_H[ll][0] = 0.0
            for k in range(1, self.N_space + 1, 1):
                xsch = r_tortoise_to_x_schwarzschild(self.rs_H[k])
                rsch = 2.0 * (1.0 + xsch)
                self.V_RW_H[ll][k] = xsch * (2.0 / (rsch ** 3)) * (ll * (ll + 1.0) + 2.0 * (self.sigma_spin) / rsch)

            # Transition Region Domain going towards the Horizon:
            for k in range(0, self.N_space + 1, 1):
                xsch = r_tortoise_to_x_schwarzschild(self.rs_H_tr[k])
                rsch = 2.0 * (1.0 + xsch)
                self.V_RW_H_tr[ll][k] = xsch * (2.0 / (rsch ** 3)) * (ll * (ll + 1.0) + 2.0 * (self.sigma_spin) / rsch)

            # Non-Compactified Domains to the Left of the Particle (Going towards the Horizon):
            for nd in range(0, self.N_Regular_Domains, 1):
                for k in range(0, self.N_space + 1, 1):
                    xsch = r_tortoise_to_x_schwarzschild(self.rs_H_RD[nd, k])
                    rsch = 2.0 * (1.0 + xsch)
                    self.V_RW_H_RD[nd, ll, k] = xsch * (2.0 / (rsch ** 3)) * (ll * (ll + 1.0) + 2.0 * (self.sigma_spin) / rsch)

            # Non-Compactified Domain to the Left of the Particle (Going towards the Horizon):
            for k in range(0, self.N_space + 1, 1):
                xsch = r_tortoise_to_x_schwarzschild(self.rs_H_pp[k])
                rsch = 2.0 * (1.0 + xsch)
                self.V_RW_H_pp[ll][k] = xsch * (2.0 / (rsch ** 3)) * (ll * (ll + 1.0) + 2.0 * (self.sigma_spin) / rsch)

            # Non-Compactified Domain to the Right of the Particle (Going towards Spatial Infinity):
            for k in range(0, self.N_space + 1, 1):
                xsch = r_tortoise_to_x_schwarzschild(self.rs_I_pp[k])
                rsch = 2.0 * (1.0 + xsch)
                self.V_RW_I_pp[ll][k] = xsch * (2.0 / (rsch ** 3)) * (ll * (ll + 1.0) + 2.0 * (self.sigma_spin) / rsch)

            # Non-Compactified Domains to the Right of the Particle (Going towards Spatial Infinity):
            for nd in range(0, self.N_Regular_Domains, 1):
                for k in range(0, self.N_space + 1, 1):
                    xsch = r_tortoise_to_x_schwarzschild(self.rs_I_RD[nd, k])
                    rsch = 2.0 * (1.0 + xsch)
                    self.V_RW_I_RD[nd, ll, k] = xsch * (2.0 / (rsch ** 3)) * (ll * (ll + 1.0) + 2.0 * (self.sigma_spin) / rsch)

            # Transition Region Domain going towards Spatial Infinity:
            for k in range(0, self.N_space + 1, 1):
                xsch = r_tortoise_to_x_schwarzschild(self.rs_I_tr[k])
                rsch = 2.0 * (1.0 + xsch)
                self.V_RW_I_tr[ll][k] = xsch * (2.0 / (rsch ** 3)) * (ll * (ll + 1.0) + 2.0 * (self.sigma_spin) / rsch)

            # Compactified Domain reaching Spatial Infinity:
            for k in range(0, self.N_space, 1):
                xsch = r_tortoise_to_x_schwarzschild(self.rs_I[k])
                rsch = 2.0 * (1.0 + xsch)
                self.V_RW_I[ll][k] = xsch * (2.0 / (rsch ** 3)) * (ll * (ll + 1.0) + 2.0 * (self.sigma_spin) / rsch)
            self.V_RW_I[ll][self.N_space] = 0.0

    # Instance Removal
    def __del__(self):

        print("Computational Grid deleted!")
        print("FRED Code (c) 2012-2020 C.F. Sopuerta")

    # Class that contains the boundary points (to be used for the different coordinates)
    class DomainBoundaries:
        def __init__(self, N_Regular_Domains):
            self.rho_IDS = 0.0  # self.rho_size_domains = 0.0

            self.H_RD_L = np.zeros(N_Regular_Domains)
            self.H_RD_R = np.zeros(N_Regular_Domains)
            self.I_RD_L = np.zeros(N_Regular_Domains)
            self.I_RD_R = np.zeros(N_Regular_Domains)

            self.horizon = 0.0
            self.infinity = 0.0

            self.H_S = 0.0
            self.H_C = 0.0
            self.I_S = 0.0
            self.I_C = 0.0

            self.H_L = 0.0
            self.H_R = 0.0
            self.H_tr_L = 0.0
            self.H_tr_R = 0.0
            self.H_pp_L = 0.0
            self.H_pp_R = 0.0

            self.I_pp_L = 0.0
            self.I_pp_R = 0.0
            self.I_tr_L = 0.0
            self.I_tr_R = 0.0
            self.H_L = 0.0
            self.H_R = 0.0

            self.peri = 0.0
            self.apo = 0.0

    # Class containing the data for the transition function used in the Hyperboloidal Compactification
    class TransitionFunction:
        def __init__(self):
            self.s = 0.0
            self.q = 0.0
