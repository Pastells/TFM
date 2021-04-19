import logging
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

# fmt: off
class Physical_Quantities:
    def __init__(self, DF, run):
        """Initializer / Instance Attributes"""

        self.run = run

        # PHYSICAL PARAMETERS OF THE RUN
        self.particle_charge = DF.particle_charge[self.run]
        self.mass_ratio = DF.mass_ratio[self.run]
        self.field_spin = DF.field_spin[self.run]
        self.sigma_spin = 1.0 - (self.field_spin) ** 2
        self.e_orbit = DF.e_orbit[self.run]
        self.p_orbit = DF.p_orbit[self.run]
        # RM self.chi_p_0 = DF.chi_p_ini[self.run]
        # RM self.phi_p_0 = DF.phi_p_ini[self.run]

        self.ell_max = DF.Max_ell[self.run]
        # RM self.N_Regular_Domains = DF.N_Regular_Domains[self.run]
        # RM self.N_space = DF.N_space[self.run]
        self.N_HD = DF.N_HD[self.run]
        self.N_OD = DF.N_OD[self.run]
        self.N_ID = DF.N_ID[self.run]
        self.N_time = DF.N_time[self.run]
        self.N_Fourier = DF.N_Fourier[self.run]

        self.Mode_accuracy = DF.Mode_accuracy[self.run]
        self.BC_at_particle = DF.BC_at_particle[self.run]

        # Internal variables
        _ell_max1 = self.ell_max + 1
        _N_OD1 = self.N_OD + 1
        _N_Fourier_1 = 2 * self.N_Fourier + 1

        # DOMAIN BOUNDARIES
        self.r_peri = self.p_orbit / (1.0 + self.e_orbit)
        self.r_apo = self.p_orbit / (1.0 - self.e_orbit)

        self.rho_peri = r_schwarzschild_to_r_tortoise(self.r_peri)
        self.rho_apo = r_schwarzschild_to_r_tortoise(self.r_apo)

        self.rho_H = DF.rho_H[self.run]
        self.rho_HC = DF.rho_HC[self.run]
        self.rho_HS = DF.rho_HS[self.run]

        self.rho_IS = DF.rho_IS[self.run]
        self.rho_IC = DF.rho_IC[self.run]
        self.rho_I = DF.rho_I[self.run]

        # GRIDS FOR ODE INTEGRATION:
        # NOTE: Remember we are integrating with respect to the rho Coordinate. Then, our grids use this Coordinate.
        # NOTE: Given that the ODEs present singular behaviour both at the Horizon and at Infinity we start the integration
        #       avoiding these points. To that end, we construct initial conditions based on approximate solutions of the ODEs
        #       both near the Horizon and near Infinity.
        epsilon_H = 2.8e-15
        self.rho_H_plus = self.rho_H + epsilon_H

        # self.rho_HD = np.linspace(self.rho_H_plus, self.rho_HC, self.N_HD + 1)
        # self.rho_HTD = np.linspace(self.rho_HC, self.rho_HS, self.N_HD + 1)
        # self.rho_HRD = np.linspace(self.rho_HS, self.rho_peri, self.N_HD + 1)
        self.rho_HD = np.linspace(self.rho_H_plus, self.rho_peri, self.N_HD + 1)
        self.rho_HOD = np.linspace(self.rho_peri, self.rho_apo, _N_OD1)

        self.rho_IOD = np.linspace(self.rho_apo, self.rho_peri, _N_OD1)
        self.rho_ID = np.linspace(self.rho_I, self.rho_apo, self.N_ID + 1)

        # HYPERBOLOIDAL COMPACTIFICATION: Transition Function Parameters
        # Transition Function/Compactification Parameters
        self.TF = self.TransitionFunction()
        self.TF.q = DF.q_transition[self.run]
        self.TF.s = DF.s_transition[self.run]

        # ARRAYS FOR DIFFERENT VARIABLES
        # Arrays for the Regularization Parameters and Singular part of the Self-Force:
        self.SF_S_r_l_H = np.zeros((_ell_max1, _N_OD1))
        self.SF_S_r_l_I = np.zeros((_ell_max1, _N_OD1))

        self.A_t_H = np.zeros((_ell_max1, _N_OD1))
        self.A_t_I = np.zeros((_ell_max1, _N_OD1))

        self.A_r_H = np.zeros((_ell_max1, _N_OD1))
        self.A_r_I = np.zeros((_ell_max1, _N_OD1))

        self.B_t = np.zeros((_ell_max1, _N_OD1))
        self.B_r = np.zeros((_ell_max1, _N_OD1))
        self.B_phi = np.zeros((_ell_max1, _N_OD1))

        self.D_r = np.zeros((_ell_max1, _N_OD1))

        # Arrays for the Computation of the Radial Component of the Bare (full) Self-Force:
        # We have the harmonic components (l,m), which in the Frequency Domain are obtained after adding up the Fourier Modes
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

        # Arrays to store the values of R_lmn and Q_lmn [Bare (full) scalar field] at each domain (2 variables x 6 domains = 12 arrays)
        # NOTE: In this version of the Code these Arrays are computed via ODE Integration:
        self.single_R_HD = np.zeros(self.N_HD + 1, dtype=np.complex128)
        self.single_Q_HD = np.zeros(self.N_HD + 1, dtype=np.complex128)
        self.single_R_HTD = np.zeros(self.N_HD + 1, dtype=np.complex128)
        self.single_Q_HTD = np.zeros(self.N_HD + 1, dtype=np.complex128)
        self.single_R_HRD = np.zeros(self.N_HD + 1, dtype=np.complex128)
        self.single_Q_HRD = np.zeros(self.N_HD + 1, dtype=np.complex128)
        self.single_R_HOD = np.zeros(_N_OD1, dtype=np.complex128)
        self.single_Q_HOD = np.zeros(_N_OD1, dtype=np.complex128)

        self.single_R_IOD = np.zeros(_N_OD1, dtype=np.complex128)
        self.single_Q_IOD = np.zeros(_N_OD1, dtype=np.complex128)
        self.single_R_ID = np.zeros(self.N_ID + 1, dtype=np.complex128)
        self.single_Q_ID = np.zeros(self.N_ID + 1, dtype=np.complex128)

        # Arrays for the computation of the Bare (full) Self-Force at the Particle Location:
        # The values of R_lmn and Q_lmn at the Particle location as evaluated at the two domains that contain the Particle
        self.R_H = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, _N_OD1), dtype=np.complex128)
        self.R_I = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, _N_OD1), dtype=np.complex128)
        self.Q_H = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, _N_OD1), dtype=np.complex128)
        self.Q_I = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, _N_OD1), dtype=np.complex128)

        # Array for the d_lm coefficients associated with the SCO's energy-momentum tensor
        self.d_lm = np.zeros((_ell_max1, _ell_max1), dtype=np.complex128)

        # Array for the C_lmn coefficients (for correcting the solutions of the master
        # equations according to the true boundary conditions at the particle location):
        self.Cp_lmn = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, _N_OD1), dtype=np.complex128)
        self.Cm_lmn = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1, _N_OD1), dtype=np.complex128)

        # Arrays for the Fourier Modes of the Jumps:
        self.J_lmn = np.zeros((_ell_max1, _ell_max1, _N_Fourier_1), dtype=np.complex128)

        # Computing Fundamental Periods and Frequencies of the Particle Orbital motion
        self.T_r = compute_radial_period(self.p_orbit, self.e_orbit)
        self.omega_r = 2.0 * (np.pi) / (self.T_r)
        self.omega_phi = compute_azimuthal_frequency(
            self.p_orbit, self.e_orbit, self.T_r
        )
        self.T_phi = 2.0 * (np.pi) / (self.omega_phi)

        # Computing the Particle Orbital Energy and Angular Momentum
        e2 = (self.e_orbit) ** 2

        self.Ep = np.sqrt(
            ( (self.p_orbit - 2.0 - 2.0 * self.e_orbit) * (self.p_orbit - 2.0 + 2.0 * self.e_orbit))
            / ((self.p_orbit) * (self.p_orbit - 3.0 - e2))
        )
        self.Lp = (self.p_orbit) / (np.sqrt(self.p_orbit - 3.0 - e2))

        # Computing the Orbital Trajectory solving the ODEs for the orbital motion of the SCO around the MBH
        # Computing N_time_half:
        # NOTE: The code forces N_time to be even
        self.N_time_half = self.N_time // 2

        # Arrays for the (Spectra and non-Spectrall) Time Grids:
        #   - Spectral Coordinates and Weights
        #   - Spectral Time: t_p_f -> [0,T_r] (full period!)
        #   - Spectral Schwarzschild and Tortoise Radial Coordinates, Angular Radial and Azimuthal (chi and phi) motion:
        #       r_p_f, rs_p_f, chi_p_f, phi_p_f
        #   - Spectral Coefficients associated with:
        #       t_p_f, r_p_f, rs_p_f, chi_p_f, phi_p_f
        #   - Non-Spectral Time: t_p -> [0,Tr/2] (half period!)
        #   - Uniform Radial grid Schwarzschild Coordinate: Uniformly distributed over [r_peri,r_apo]: r_p
        #   - Tortoise Radial coordinate, Angular coordinates for the radial (chi) and Azimuthal (phi) motion:
        #       rs_p, chi_p, phi_p
        self.Xt = np.zeros(self.N_time + 1)
        self.Wt = (np.pi / (self.N_time)) * np.ones(self.N_time + 1)

        self.t_p_f = np.zeros(self.N_time + 1)
        self.r_p_f = np.zeros(self.N_time + 1)
        self.rs_p_f = np.zeros(self.N_time + 1)
        self.chi_p_f = np.zeros(self.N_time + 1)
        self.phi_p_f = np.zeros(self.N_time + 1)

        self.An_t_p_f = np.zeros(self.N_time + 1)
        self.An_r_p_f = np.zeros(self.N_time + 1)
        self.An_rs_p_f = np.zeros(self.N_time + 1)
        self.An_chi_p_f = np.zeros(self.N_time + 1)
        self.An_phi_p_f = np.zeros(self.N_time + 1)

        self.t_p = np.zeros(_N_OD1)
        self.r_p = np.zeros(_N_OD1)
        self.rs_p = np.zeros(_N_OD1)
        self.chi_p = np.zeros(_N_OD1)
        self.phi_p = np.zeros(_N_OD1)

        # Computation of the Chebyshev-Lobatto Grid and Weights:
        self.Xt[0] = -1.0
        self.Xt[self.N_time] = 1.0

        self.Wt[0] = 0.5 * (self.Wt[0])
        self.Wt[self.N_time] = 0.5 * (self.Wt[self.N_time])

        for k in range(1, self.N_time_half):
            self.Xt[k] = -np.cos(k * np.pi / (self.N_time))
            self.Xt[self.N_time - k] = -self.Xt[k]
        self.Xt[self.N_time_half] = 0.0

        # Integration Times at which we must solve the Orbit ODEs [From t = 0 to t = Tr]:
        self.t_p_f = 0.5 * (self.T_r) * (1.0 + self.Xt)

        # Solving the Orbit ODEs for the interval t in [0, Tr]:
        # NOTE: The values of chi_p_0 and phi_p_0 are hardwired in this version of the code because
        #       we want to start at r_p = r_peri and phi
        self.chi_p_0 = 0.0
        self.phi_p_0 = 0.0

        y0 = [self.chi_p_0, self.phi_p_0]
        self.ode_sol = odeint(
            sco_odes_rhs,
            y0,
            self.t_p_f,
            args=(self.p_orbit, self.e_orbit),
            rtol=1.0e-13,
            atol=1.0e-14,
        )

        self.chi_p_f = self.ode_sol[:, 0]
        self.phi_p_f = self.ode_sol[:, 1]

        # Computation of the Particle Radial Location [Schwarzschild Coordinate]:
        self.r_p_f = (self.p_orbit) / (1.0 + (self.e_orbit) * (np.cos(self.chi_p_f)))

        # Computation of the Particle Radial Location [Tortoise Coordinate]:
        self.rs_p_f = self.r_p_f - 2.0 * np.log(0.5 * (self.r_p_f) - 1.0)

        # Uniform Grid for the Schwarzschild Radial Coordinate: r_p
        self.r_p = self.r_peri + ((self.r_apo - self.r_peri) / self.N_OD) * np.arange(_N_OD1)

        # Computing the d_lm coefficients:
        for ll in range(0, _ell_max1):
            for mm in range(0, ll + 1):

                # Checking whether ell+m is even (for ell+m odd the contribution is zero)
                if (ll + mm) % 2 == 0:
                    self.d_lm[ll, mm] = special.sph_harm(mm, ll, 0.0, 0.5 * np.pi)

                else:
                    self.d_lm[ll, mm] = 0.0


    # ------------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------------


    def complete_m_mode(self, ll, mm):
        """Apply the missing factor to the Radial Component of the Bare (full)
        Self-Force at each particle location
        Add contribution of the m_Mode to the l_Mode of the Radial Component"""

        self.SF_F_r_lm_H[ll, mm] *= (
            (self.particle_charge / self.r_p)
            * self.d_lm[ll, mm]
            * np.exp(1j * mm * (self.phi_p - self.omega_phi * self.t_p))
        )
        self.SF_F_r_lm_I[ll, mm] *= (
            (self.particle_charge / self.r_p)
            * self.d_lm[ll, mm]
            * np.exp(1j * mm * (self.phi_p - self.omega_phi * self.t_p))
        )

        # Add contribution to l-Mode
        if mm == 0:
            self.SF_F_r_l_H[ll] += self.SF_F_r_lm_H[ll, mm]
            self.SF_F_r_l_I[ll] += self.SF_F_r_lm_I[ll, mm]
        else:
            self.SF_F_r_l_H[ll] += 2 * np.real(self.SF_F_r_lm_H[ll, mm])
            self.SF_F_r_l_I[ll] += 2 * np.real(self.SF_F_r_lm_I[ll, mm])


    def complete_l_mode(self, ll):
        """substract l-Mode contribution from the Singular field"""
        # Computation of the l-Mode of the Radial Component of the Regular Self-Force:
        self.SF_R_r_l_H[ll] = self.SF_F_r_l_H[ll] - self.SF_S_r_l_H[ll]
        self.SF_R_r_l_I[ll] = self.SF_F_r_l_I[ll] - self.SF_S_r_l_I[ll]

        # Total Regular Self-Force
        self.SF_r_H = self.SF_r_H + self.SF_R_r_l_H[ll]
        self.SF_r_I = self.SF_r_I + self.SF_R_r_l_I[ll]


    def rescale_mode(self, ll, mm, nf):
        """Computing the Values of the Field Modes (Phi,Psi)lmn [~ (R,Q)lmn in Frequency Domain] at the Particle Location
        at the different Time Collocation Points that satisfy the Boundary Conditions imposed by the Jump Conditions.
        [First with arbitrary boundary conditions, using the solution found in the previous point and afterwards,
        rescaling with the C_lmn coefficients to obtain the field modes (lmn) With the correct boundary conditions.
        this includes the computation of the C_lmn coefficients]:
        NOTE: REMEMBER that we have projected the geodesics onto the Spatial Domain "containing" the Particle: [r_peri, r_apo]
        NOTE: This why the index goes form 0 to N_space instead of from 0 to N_time"""

        indices = (ll, mm, nf + self.N_Fourier)

        # Store computed modes from compute_mode
        self.R_H[indices] = self.single_R_HOD
        self.R_I[indices] = self.single_R_IOD
        self.Q_H[indices] = self.single_Q_HOD
        self.Q_I[indices] = self.single_Q_IOD


        # Particle Location (rho/rstar and r Coordinates)
        rp = self.r_p

        # Schwarzschild metric function 'f = 1 - 2M/r'
        fp = 1.0 - 2.0 / rp

        # Value of the Jump
        self.J_lmn[indices] = Jump_Value(ll, mm, nf, self)

        # Computing the C_lmn Coefficients [for the Harmonic mode (ll,mm), Fourier mode 'nt', and location <=> time 'ns']
        Wronskian_RQ = self.R_I[indices] * self.Q_H[indices] - self.R_H[indices] * self.Q_I[indices]

        self.Cm_lmn[indices] = -self.R_I[indices] * self.J_lmn[indices] / Wronskian_RQ
        self.Cp_lmn[indices] = -self.R_H[indices] * self.J_lmn[indices] / Wronskian_RQ

        # Computing the Values of the Bare Field Modes (R,Q)(ll,mm,nn) at the Particle Location 'ns'
        # using the Correct Boundary Conditions: RESCALING WITH THE C_lmn COEFFICIENTS
        self.R_H[indices] = self.Cm_lmn[indices] * self.R_H[indices]
        self.Q_H[indices] = self.Cm_lmn[indices] * self.Q_H[indices]

        self.R_I[indices] = self.Cp_lmn[indices] * self.R_I[indices]
        self.Q_I[indices] = self.Cp_lmn[indices] * self.Q_I[indices]

        # Computation of the contribution of the Fourier Mode 'nf' (l m nf) to the Radial Component of the Bare (full) Self-Force
        # [NOTE: This is the contribution up to a multiplicative factor that is applied below]
        self.SF_F_r_lm_H[ll, mm] = self.SF_F_r_lm_H[ll, mm] + (
            self.Q_H[indices] / fp - self.R_H[indices] / rp
        ) * np.exp(-1j * nf * self.omega_r * self.t_p)
        self.SF_F_r_lm_I[ll, mm] = self.SF_F_r_lm_I[ll, mm] + (
            self.Q_I[indices] / fp - self.R_I[indices] / rp
        ) * np.exp(-1j * nf * self.omega_r * self.t_p)

        # print( f"l={ll} m={mm} n={nf}: c_H[{nf}]={self.SF_F_r_lm_H[ll, mm]:.14f}  c_I[{nf}]={self.SF_F_r_lm_I[ll, mm]:.14f}")

        # Estimate error and store contribution
        self.Estimated_Error = np.maximum(
            np.absolute(self.Accumulated_SF_F_r_lm_H[ll, mm]-self.SF_F_r_lm_H[ll,mm]),
            np.absolute(self.Accumulated_SF_F_r_lm_I[ll, mm]-self.SF_F_r_lm_I[ll,mm]),
        )

        self.Accumulated_SF_F_r_lm_H[ll, mm] = self.SF_F_r_lm_H[ll, mm]
        self.Accumulated_SF_F_r_lm_I[ll, mm] = self.SF_F_r_lm_I[ll, mm]


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
