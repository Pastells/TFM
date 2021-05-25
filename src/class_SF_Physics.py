"""Class with physical quantities and arrays needed for the main program"""

import os
import time
import logging
import pickle
import numpy as np
from scipy.integrate import odeint
from scipy import special
from orbital_computations import (
    compute_radial_period,
    compute_azimuthal_frequency,
    sco_odes_rhs,
)
from Schwarzschild import r_schwarzschild_to_r_tortoise
from Some_functions import Jump_Value

# Disable some linter warnings and format
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name
# pylint: disable=line-too-long
# noqa: E221
# fmt: off


class Physical_Quantities:
    """Class with physical quantities and arrays needed for the main program"""
    def __init__(self, df, run=0, save=False):
        """Initializer / Instance Attributes"""

        self.run = run
        self.save = save

        self.read_df(df)

        # DOMAIN BOUNDARIES
        self.peri_apo()

        self.grid()

        # HYPERBOLOIDAL COMPACTIFICATION: Transition Function Parameters
        # Transition Function/Compactification Parameters
        self.TF = self.TransitionFunction()
        self.TF.q = df.q_transition[self.run]
        self.TF.s = df.s_transition[self.run]

        # ARRAYS FOR DIFFERENT VARIABLES
        self.SF_init()

        self.orbit()

        self.trajectory()

        self.chebyshev_coefs()

        self.spherical_harm()

        # Arrays for the Fourier Modes of the Jumps:
        self.J_lmn = np.zeros((self.ell_max + 1, self.ell_max + 1, 2 * self.N_Fourier + 1), dtype=np.complex128)

        # List of variables to be saved / retrieved with pickle
        self.var_list = ["R_H", "R_I", "Q_H", "Q_I", "rho_HOD", "J_lmn",
                         "SF_F_r_l_H", "SF_F_r_l_I", "SF_S_r_l_H", "SF_S_r_l_I"]
        if self.save:
            self.var_list += ["R_HD", "R_ID", "Q_HD", "Q_ID", "rho_HD", "rho_ID"]

    # ------------------------------------------------------------------------
    # Init functions
    # ------------------------------------------------------------------------

    def read_df(self, df):
        """Physical parameters of the run"""
        self.charge = df.particle_charge[self.run]
        self.mass_ratio = df.mass_ratio[self.run]
        self.field_spin = df.field_spin[self.run]
        self.sigma_spin = 1.0 - (self.field_spin) ** 2
        self.e_orbit = df.e_orbit[self.run]
        self.p_orbit = df.p_orbit[self.run]

        self.ell_max = df.Max_ell[self.run]
        self.N_HD = df.N_HD[self.run]
        self.N_OD = df.N_OD[self.run]
        self.N_ID = df.N_ID[self.run]
        self.N_time = df.N_time[self.run]
        self.N_Fourier = int(df.N_Fourier[self.run])

        self.Mode_accuracy = df.Mode_accuracy[self.run]
        self.BC_at_particle = df.BC_at_particle[self.run]

        self.rho_H  = df.rho_H[self.run]
        self.rho_HC = df.rho_HC[self.run]
        self.rho_HS = df.rho_HS[self.run]
        self.rho_IS = df.rho_IS[self.run]
        self.rho_IC = df.rho_IC[self.run]
        self.rho_I  = df.rho_I[self.run]

    # ------------------------------------------------------------------------

    def peri_apo(self):
        self.r_peri = self.p_orbit / (1.0 + self.e_orbit)
        self.r_apo = self.p_orbit / (1.0 - self.e_orbit)

        self.rho_peri = r_schwarzschild_to_r_tortoise(self.r_peri)
        self.rho_apo = r_schwarzschild_to_r_tortoise(self.r_apo)

    # ------------------------------------------------------------------------

    def grid(self):
        """
        GRIDS FOR ODE INTEGRATION:
        Remember we are integrate with respect to rho. Then, our grids use rho as a coordinate
        Given that the ODEs present singular behaviour both at the Horizon and at Infinity,
        we start the integration avoiding these points. To that end, we construct initial conditions
        based on approximate solutions of the ODEs both near the Horizon and near Infinity.
        """

        epsilon_H = 2.8e-15
        self.rho_H_plus = self.rho_H + epsilon_H

        self.rho_HOD = np.linspace(self.rho_peri, self.rho_apo, self.N_OD + 1)
        self.rho_IOD = np.linspace(self.rho_apo, self.rho_peri, self.N_OD + 1)

        if self.save:
            self.rho_HD = np.linspace(self.rho_H_plus, self.rho_peri, self.N_HD + 1)
            self.rho_ID = np.linspace(self.rho_I, self.rho_apo, self.N_ID + 1)

    # ------------------------------------------------------------------------

    def SF_init(self):
        _ell_max1 = self.ell_max + 1
        _N_OD1 = self.N_OD + 1
        _N_Fourier_1 = 2 * self.N_Fourier + 1

        # Arrays for the Regularization Parameters and Singular part of the Self-Force:
        self.SF_S_r_l_H = np.zeros((_ell_max1, _N_OD1))
        self.SF_S_r_l_I = np.zeros((_ell_max1, _N_OD1))

        # Arrays for the Computation of the Radial Component of the Bare (full) Self-Force:
        # We have the harmonic components (l,m), obtained in the Frequency Domain after adding up the Fourier Modes
        # And we have the l-Modes of the Radial Component of the Bare (full) Self-Force, obtained after sum over m-Modes
        self.SF_F_r_lm_H = np.zeros((_ell_max1, _ell_max1, _N_OD1), dtype=np.complex128)
        self.SF_F_r_lm_I = np.zeros((_ell_max1, _ell_max1, _N_OD1), dtype=np.complex128)

        self.SF_F_r_l_H = np.zeros((_ell_max1, _N_OD1), dtype=np.complex128)
        self.SF_F_r_l_I = np.zeros((_ell_max1, _N_OD1), dtype=np.complex128)

        # Arrays for Estimating the Error in the Fourier Series [to assess where to truncate the series]
        self.Accumulated_SF_F_r_lm_H = np.zeros((_ell_max1, _ell_max1, _N_OD1), dtype=np.complex128)
        self.Accumulated_SF_F_r_lm_I = np.zeros((_ell_max1, _ell_max1, _N_OD1), dtype=np.complex128)

        self.Estimated_Error = np.zeros(_N_OD1)

        # Arrays for the Radial Component of the Regular Self-Force:
        # We have the l-Modes of the Radial Component of the Regular Self-Force
        # And we have the Radial Component of the Regular Self-Force, obtained after sum over l-Modes
        self.SF_R_r_l_H = np.zeros((_ell_max1, _N_OD1), dtype=np.complex128)
        self.SF_R_r_l_I = np.zeros((_ell_max1, _N_OD1), dtype=np.complex128)

        self.SF_r_H = np.zeros(_N_OD1, dtype=np.complex128)
        self.SF_r_I = np.zeros(_N_OD1, dtype=np.complex128)

        # Arrays to store the values of R_lmn and Q_lmn [Bare (full) scalar field] at each domain
        # NOTE: In this version of the Code these Arrays are computed via ODE Integration:
        self.single_R_HOD = np.zeros(_N_OD1, dtype=np.complex128)
        self.single_Q_HOD = np.zeros(_N_OD1, dtype=np.complex128)
        self.single_R_IOD = np.zeros(_N_OD1, dtype=np.complex128)
        self.single_Q_IOD = np.zeros(_N_OD1, dtype=np.complex128)

        if self.save:
            self.single_R_HD = np.zeros(self.N_HD + 1, dtype=np.complex128)
            self.single_Q_HD = np.zeros(self.N_HD + 1, dtype=np.complex128)
            self.single_R_ID = np.zeros(self.N_ID + 1, dtype=np.complex128)
            self.single_Q_ID = np.zeros(self.N_ID + 1, dtype=np.complex128)

        # Arrays for the computation of the Bare (full) Self-Force at the Particle Location:
        # The values of R_lmn and Q_lmn at the Particle location, evaluated at the two domains that contain the Particle
        self.R_H = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, _N_OD1), dtype=np.complex128)
        self.R_I = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, _N_OD1), dtype=np.complex128)
        self.Q_H = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, _N_OD1), dtype=np.complex128)
        self.Q_I = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, _N_OD1), dtype=np.complex128)

        if self.save:
            self.R_HD = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, self.N_HD + 1), dtype=np.complex128)
            self.R_ID = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, self.N_HD + 1), dtype=np.complex128)
            self.Q_HD = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, self.N_ID + 1), dtype=np.complex128)
            self.Q_ID = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, self.N_ID + 1), dtype=np.complex128)

    # ------------------------------------------------------------------------

    def orbit(self):
        """Compute Fundamental Periods and Frequencies of the Particle Orbital motion"""
        self.T_r = compute_radial_period(self.p_orbit, self.e_orbit)
        self.omega_r = 2.0 * (np.pi) / (self.T_r)
        self.omega_phi = compute_azimuthal_frequency(
            self.p_orbit, self.e_orbit, self.T_r
        )
        self.T_phi = 2.0 * (np.pi) / (self.omega_phi)

        # Compute the Particle Orbital Energy and Angular Momentum
        e2 = (self.e_orbit) ** 2

        self.Ep = np.sqrt(
            ((self.p_orbit - 2 - 2 * self.e_orbit) * (self.p_orbit - 2 + 2 * self.e_orbit))
            / ((self.p_orbit) * (self.p_orbit - 3 - e2))
        )
        self.Lp = (self.p_orbit) / (np.sqrt(self.p_orbit - 3.0 - e2))

    # ------------------------------------------------------------------------

    def trajectory(self):
        """Compute the Orbital Trajectory solving the ODEs for the orbital motion of the SCO around the MBH
        Arrays for the (Spectral and non-Spectral) Time Grids:
          - Spectral Coordinates and Weights
          - Spectral Time: t_p_f -> [0,T_r] (full period!)
          - Spectral Schwarzschild and Tortoise Radial Coordinates, Angular Radial and Azimuthal (chi and phi) motion:
              r_p_f, rs_p_f, chi_p_f, phi_p_f
          - Spectral Coefficients associated with:
              t_p_f, r_p_f, rs_p_f, chi_p_f, phi_p_f
          - Non-Spectral Time: t_p -> [0,Tr/2] (half period!)
          - Uniform Radial grid Schwarzschild Coordinate: Uniformly distributed over [r_peri,r_apo]: r_p
          - Tortoise Radial coordinate, Angular coordinates for the radial (chi) and Azimuthal (phi) motion:
              rs_p, chi_p, phi_p
        """
        self.t_p = np.zeros(self.N_OD + 1)
        self.rs_p = np.zeros(self.N_OD + 1)
        self.chi_p = np.zeros(self.N_OD + 1)
        self.phi_p = np.zeros(self.N_OD + 1)

        # Computation of the Chebyshev-Lobatto Grid and Weights:
        self.Xt = np.zeros(self.N_time + 1)
        self.Xt[0] = -1.0
        self.Xt[self.N_time] = 1.0

        Wt = (np.pi / (self.N_time)) * np.ones(self.N_time + 1)
        Wt[0] = 0.5 * (Wt[0])
        Wt[self.N_time] = 0.5 * (Wt[self.N_time])

        N_time_half = self.N_time // 2  # code forces N_time to be even
        for k in range(1, N_time_half):
            self.Xt[k] = -np.cos(k * np.pi / (self.N_time))
            self.Xt[self.N_time - k] = -self.Xt[k]
        self.Xt[N_time_half] = 0.0

        print(self.Xt)

        # Integration Times at which we must solve the Orbit ODEs [From t = 0 to t = Tr]:
        # TODO: llavors el 0.5 no caldria
        self.t_p_f = 0.5 * (self.T_r) * (1.0 + self.Xt)

        # Solve the Orbit ODEs for the interval t in [0, Tr]:
        # chi_p_0 and phi_p_0 are hardwired to 0 because we want to start at r_p = r_peri

        y0 = [0, 0]
        self.ode_sol = odeint(
            sco_odes_rhs,
            y0,
            self.t_p_f,
            args=(self.p_orbit, self.e_orbit),
            rtol=1e-13,
            atol=1e-14,
        )

        self.chi_p_f = self.ode_sol[:, 0]
        self.phi_p_f = self.ode_sol[:, 1]

        # Computation of the Particle Radial Location [Schwarzschild Coordinate]:
        self.r_p_f = (self.p_orbit) / (1.0 + (self.e_orbit) * (np.cos(self.chi_p_f)))

        # Computation of the Particle Radial Location [Tortoise Coordinate]:
        self.rs_p_f = self.r_p_f - 2.0 * np.log(0.5 * (self.r_p_f) - 1.0)

        # Uniform Grid for the Schwarzschild and Tortoise Radial Coordinates: r_p, rs_p
        self.r_p = self.r_peri + ((self.r_apo - self.r_peri) / self.N_OD) * np.arange(self.N_OD + 1)
        self.rs_p = self.r_p - 2.0 * np.log(0.5 * (self.r_p) - 1.0)

        print(self.r_p_f)
        print(self.r_p)

    # ------------------------------------------------------------------------

    def chebyshev_coefs(self):
        """Coefficients for expanding in Chebyshev polynomials
        Obtained from values at known positions"""
        self.Ai_t_p_f   = np.zeros(self.N_time + 1)
        self.Ai_r_p_f   = np.zeros(self.N_time + 1)
        self.Ai_rs_p_f  = np.zeros(self.N_time + 1)
        self.Ai_chi_p_f = np.zeros(self.N_time + 1)
        self.Ai_phi_p_f = np.zeros(self.N_time + 1)

        n = self.N_time
        for k in range(n + 1):
            # Vector with the n-th Chebyshev Polynomials at the given spectral coordinate:
            # T_i_x = np.array([special.eval_chebyt(i, self.r_p_f[k]) for i in range(n + 1)])
            T_i_x = np.array([special.eval_chebyt(i, self.Xt[k]) for i in range(1, n + 2)])
            self.Ai_t_p_f   += self.t_p_f[k]   * T_i_x
            self.Ai_r_p_f   += self.r_p_f[k]   * T_i_x
            self.Ai_rs_p_f  += self.rs_p_f[k]  * T_i_x
            self.Ai_phi_p_f += self.phi_p_f[k] * T_i_x
            self.Ai_chi_p_f += self.chi_p_f[k] * T_i_x

        # Normalization
        self.Ai_t_p_f   = self.Ai_t_p_f   * 2 / (n + 1)
        self.Ai_r_p_f   = self.Ai_r_p_f   * 2 / (n + 1)
        self.Ai_rs_p_f  = self.Ai_rs_p_f  * 2 / (n + 1)
        self.Ai_phi_p_f = self.Ai_phi_p_f * 2 / (n + 1)
        self.Ai_chi_p_f = self.Ai_chi_p_f * 2 / (n + 1)

    # ------------------------------------------------------------------------

    def spherical_harm(self):
        """Create array for the d_lm coefficients associated with the SCO's energy-momentum tensor"""
        self.d_lm = np.zeros((self.ell_max + 1, self.ell_max + 1), dtype=np.complex128)

        for ll in range(self.ell_max + 1):
            for mm in range(ll + 1):

                # Check whether ell+m is even (for ell+m odd the contribution is zero)
                if (ll + mm) % 2 == 0:
                    self.d_lm[ll, mm] = special.sph_harm(mm, ll, 0.0, 0.5 * np.pi)
                else:
                    self.d_lm[ll, mm] = 0

    # ------------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------------

    def complete_m_mode(self, ll, mm):
        """Apply the missing factor to the Radial Component of the Bare (full)
        Self-Force at each particle location
        Add contribution of the m_Mode to the l_Mode of the Radial Component"""

        weight = (
            self.charge / self.r_p
            * self.d_lm[ll, mm]
            * np.exp(1j * mm * (self.phi_p - self.omega_phi * self.t_p))
        )

        self.SF_F_r_lm_H[ll, mm] *= weight
        self.SF_F_r_lm_I[ll, mm] *= weight

        # Add contribution to l-Mode (appendix A)
        if mm == 0:
            self.SF_F_r_l_H[ll] += self.SF_F_r_lm_H[ll, mm]
            self.SF_F_r_l_I[ll] += self.SF_F_r_lm_I[ll, mm]
        else:
            self.SF_F_r_l_H[ll] += 2 * np.real(self.SF_F_r_lm_H[ll, mm])
            self.SF_F_r_l_I[ll] += 2 * np.real(self.SF_F_r_lm_I[ll, mm])

    # ------------------------------------------------------------------------

    def complete_l_mode(self, ll):
        """substract l-Mode contribution from the Singular field"""
        # Computation of the l-Mode of the Radial Component of the Regular Self-Force:
        self.SF_R_r_l_H[ll] = self.SF_F_r_l_H[ll] - self.SF_S_r_l_H[ll]
        self.SF_R_r_l_I[ll] = self.SF_F_r_l_I[ll] - self.SF_S_r_l_I[ll]

        # Total Regular Self-Force
        self.SF_r_H = self.SF_r_H + self.SF_R_r_l_H[ll]
        self.SF_r_I = self.SF_r_I + self.SF_R_r_l_I[ll]

    # ------------------------------------------------------------------------

    def store(self, indices):
        """Save solution from compute_mode into permanent ndarray
        indices = (ll, mm, nf + PP.N_Fourier)"""

        self.R_H[indices] = self.single_R_HOD
        self.R_I[indices] = self.single_R_IOD
        self.Q_H[indices] = self.single_Q_HOD
        self.Q_I[indices] = self.single_Q_IOD

        if self.save:
            self.R_HD[indices] = self.single_R_HD
            self.R_ID[indices] = self.single_R_ID
            self.Q_HD[indices] = self.single_Q_HD
            self.Q_ID[indices] = self.single_Q_ID

    # ------------------------------------------------------------------------

    def rescale_mode(self, ll, mm, nf):
        """Compute the Values of the Field Modes (Phi,Psi)lmn [~ (R,Q)lmn in Frequency Domain] at the Particle Location
        at the different Time Collocation Points that satisfy the Boundary Conditions imposed by the Jump Conditions.
        [First with arbitrary boundary conditions, using the solution found in the previous point and afterwards,
        rescaling with the C_lmn coefficients to obtain the field modes (lmn) With the correct boundary conditions.
        this includes the computation of the C_lmn coefficients]:
        NOTE: We have projected the geodesics onto the Spatial Domain "containing" the Particle: [r_peri, r_apo]
        NOTE: This why the index goes form 0 to N_space instead of from 0 to N_time"""

        indices = (ll, mm, nf + self.N_Fourier)

        # Particle Location (rho/rstar and r Coordinates)
        rp = self.r_p

        # Schwarzschild metric function 'f = 1 - 2M/r'
        fp = 1.0 - 2.0 / rp

        # Value of the Jump
        # J_lmn = Jump_Value(ll, mm, nf, self)  # TODO provisional
        self.J_lmn[indices] = Jump_Value(ll, mm, nf, self)
        J_lmn = self.J_lmn[indices]

        # Compute the C_lmn Coefficients [for the Harmonic mode (ll,mm), Fourier mode 'nt', and location <=> time 'ns']
        wronskian_RQ = self.R_H[indices] * self.Q_I[indices] - self.R_I[indices] * self.Q_H[indices]
        Cm_lmn = self.R_I[indices] * J_lmn / wronskian_RQ
        Cp_lmn = self.R_H[indices] * J_lmn / wronskian_RQ

        # Compute the Values of the Bare Field Modes (R,Q)(ll,mm,nn) at the Particle Location 'ns'
        # using the Correct Boundary Conditions: RESCALING WITH THE C_lmn COEFFICIENTS
        self.R_H[indices] *= Cm_lmn
        self.Q_H[indices] *= Cm_lmn
        self.R_I[indices] *= Cp_lmn
        self.Q_I[indices] *= Cp_lmn

        # Contribution of the Fourier Mode 'nf' (l m nf) to the Radial Component of the Bare (full) Self-Force
        # [NOTE: This is the contribution up to a multiplicative factor that is applied below]
        exp_factor = np.exp(-1j * nf * self.omega_r * self.t_p)
        self.SF_F_r_lm_H[ll, mm] += (self.Q_H[indices] / fp - self.R_H[indices] / rp) * exp_factor
        self.SF_F_r_lm_I[ll, mm] += (self.Q_I[indices] / fp - self.R_I[indices] / rp) * exp_factor

        # print( f"l={ll} m={mm} n={nf}: c_H[{nf}]={self.SF_F_r_lm_H[ll, mm]}  c_I[{nf}]={self.SF_F_r_lm_I[ll, mm]}")

        # Estimate error and store contribution
        self.Estimated_Error = np.maximum(
            np.absolute(self.Accumulated_SF_F_r_lm_H[ll, mm] - self.SF_F_r_lm_H[ll, mm]),
            np.absolute(self.Accumulated_SF_F_r_lm_I[ll, mm] - self.SF_F_r_lm_I[ll, mm]),
        )

        self.Accumulated_SF_F_r_lm_H[ll, mm] = self.SF_F_r_lm_H[ll, mm]
        self.Accumulated_SF_F_r_lm_I[ll, mm] = self.SF_F_r_lm_I[ll, mm]

    # ------------------------------------------------------------------------

    def saving(self):
        """Save resulting modes"""
        logging.info("FRED RUN %f: Saving results", self.run)
        if os.path.isfile("results/R_H.pkl"):
            folder = time.strftime("results/%Y-%m-%d_%H:%M")
            os.mkdir(folder)
            logging.warning(
                "Pickle files already exist, moved into new folder %s",
                folder,
            )
        else:
            folder = "results"

        def _pickle_dump(var_str, directory=folder):
            object_to_save = getattr(self, var_str)
            object_name = directory + "/" + var_str + ".pkl"
            pickle.dump(object_to_save, open(object_name, "wb"))

        for var in self.var_list:
            _pickle_dump(var)

    # ------------------------------------------------------------------------

    def read(self, folder="results"):
        def _pickle_load(var_str, directory=folder):
            object_name = directory + "/" + var_str + ".pkl"
            object_to_load = pickle.load(open(object_name, "rb"))
            setattr(self, var_str, object_to_load)

        for var in self.var_list:
            _pickle_load(var)

    # ------------------------------------------------------------------------

    def __del__(self):
        """Instance Removal"""
        logging.info("Physical Parameter Class deleted!")
        logging.info("FRED Code (c) 2012-2021 C.F. Sopuerta")

    class TransitionFunction:
        """Class containing the data for the transition function used in the Hyperboloidal Compactification"""

        def __init__(self):
            self.s = 0.0
            self.q = 0.0
# fmt: on
