"""
Complex R and Q, Radau method may not work

NOTE: def compute_mode(ell,omega,CGC,PP):

This function computes a Single Scalar Field Mode determined by the numbers (l,m,n) at the two domains
(the Horizon one and the Spatial Infinity one).

The arguments are:

    * The Harmonic Number 'l'
    * The Mode Frequency 'w_mn' (which depends on the other two "quantum" numbers 'm' and 'n')
    * PP [an object that contains a number of physical quantities required for the computation].

NOTE: The dependence of the mode on 'm' and 'n' comes in the frequency 'omega'. The extra dependence on these
numbers as well as on other physical input (particle's trajectory) comes from the jumps. But the jumps do not
enter in this computation since here we are solving for generic boundary conditions at pericenter and apocenter.

NOTE: DOMAIN STRUCTURE: We have two Regions that intersect in one Domain, the Particle Domain [r_peri, r_apo]
The structure of each Region (Horizon and Infinity) is:

      HD             H_tr           HOD
 o-------------o=============o--------------o

                             o--------------o=============o-------------o
                                    IOD          I_tr            ID
where:

          HD: Horizon Domain touching the BH Horizon. It contains a Transition Region, here denoted as H_tr
          HOD: Horizon Orbital Domain, the Particle Domain
          IOD: Infinity Orbital Domain, the Particle Domain. It is diffeomorphic to HOD.
          ID: Infinity Domain touching Future Null Infinity

NOTATION: The original field equations at the two sides of the particle are:
NOTE: '-' corresponds to the Horizon Region and '+' to the Infinity Region

          ( -D^2_t + D^2_r* - V_RW_l(r) ) Psi^{+,-}_lm = 0

We introduce the following change of variables (VPhi = varphi):

          Phi^{+,-}_lm = (1+H) D_tau Psi^{+,-}_lm + H D_rho Psi^{+,-}_lm
          VPhi^{+,-}_lm = D_rho Psi^{+,-}_lm

In the Frequency Domain:

          Psi^{+,-}_lm = Sum_n(-oo,+oo) exp{-i omega_mn tau} R^{+,-}_lmn(rho)
          VPhi^{+,-}_lm = Sum_n(-oo,+oo) exp{-i omega_mn tau} Q^{+,-}_lmn(rho)

The function 'compute_mode' solves the ODEs for R^{+,-}_lmn and Q^{+,-}_lmn

The Boundary Condition at the horizon is:

          ( D_t - D_r* ) Psi^{-}_lm = 0

and the Boundary Condition at Spatial Infinity is:

          ( D_t + D_r* ) Psi^{+}_lm = 0

These conditions can be translated into conditions for the Fourier Modes:

          i omega_mn R^{-}_lmn + Q^{-}_lmn = 0

          i omega_mn R^{+}_lmn - Q^{+}_lmn = 0

Our strategy is to find first solutions for the Fourier Modes prescribing arbitrary Boundary Conditions
at the pericenter (r_peri) and at the apocenter (r_apo).  This Boundary Conditions are Dirichlet
Boundary Conditions. That is:

          \hat{R}^{-}_lmn(r_apo) = A_MINUS

          \hat{R}^{+}_lmn(r_peri) = A_PLUS
"""


import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from Schwarzschild import rstar_to_xsch
from hyperboloidal_compactification_tanh import (
    f_transition,
    f_transition_1st,
    f_transition_2nd,
)


def regge_wheeler_potential(rstar, ell):
    cb = ell * (ell + 1)
    xstar = 0.5 * rstar - 1.0
    xsch = rstar_to_xsch(rstar)
    rsch = 2.0 * (1.0 + xsch)
    # exp(xstar-x) = x
    return 2.0 * np.exp(xstar - xsch) * (cb + 2.0 / rsch) / (rsch ** 3)


def RHS_HD_zero_freq(rho, u, ell, PP):
    """This function defines the form of the ODE system that determine the solution to the Master Equation
    It returns the Right-Hand-Side (RHS) of the ODEs
    NOTE: This is ONLY for the Horizon Domain
    NOTE: This is ONLY for the Zero Frequency Modes"""

    re_R, im_R, re_Q, im_Q = u
    R = re_R + 1j * im_R
    Q = re_Q + 1j * im_Q
    dR, dQ = 0, 0

    # Integration through the Horizon Domain (HD)
    if PP.rho_H <= rho <= PP.rho_HC:

        Omega = 1.0 - rho / PP.rho_H
        H = 1.0 - Omega ** 2
        DH_over_1minusH = (2.0 / PP.rho_H) / Omega
        one_minus_Hsq = Omega ** 4

        rstar = rho / Omega
        Vl = regge_wheeler_potential(rstar, ell)

        dR = Q
        dQ = DH_over_1minusH * Q + Vl / one_minus_Hsq * R

    # Integration through the Horizon Transition Region
    elif PP.rho_HC < rho < PP.rho_HS:

        width = PP.rho_HC - PP.rho_HS
        sigma = 0.5 * (np.pi) * (rho - PP.rho_HS) / width
        jacobian = 0.5 * np.pi / width

        f0 = f_transition(sigma, PP.TF)
        f1 = jacobian * f_transition_1st(f0, sigma, PP.TF)
        f2 = (jacobian ** 2) * f_transition_2nd(f0, sigma, PP.TF)

        Omega = 1.0 - f0 * rho / PP.rho_H
        dOmega_drho = -(f0 + rho * f1) / PP.rho_H
        d2Omega_drho2 = -(2.0 * f1 + rho * f2) / PP.rho_H

        LH = Omega - rho * dOmega_drho
        H = 1.0 - (Omega ** 2) / LH
        one_minus_H = 1.0 - H
        DH = -(Omega / LH) * (2.0 * dOmega_drho + rho * (Omega / LH) * d2Omega_drho2)
        DH_over_1minusH = DH / one_minus_H
        one_minus_Hsq = one_minus_H ** 2

        rstar = rho / Omega
        Vl = regge_wheeler_potential(rstar, ell)

        dR = Q
        dQ = DH_over_1minusH * Q + Vl / one_minus_Hsq * R

    # Integration through the Regular Region  (i.e. rho = rstar)
    elif PP.rho_HS <= rho <= PP.rho_peri:
        Vl = regge_wheeler_potential(rho, ell)
        dR = Q
        dQ = Vl * R

    # Outside the Physical Integration Region
    else:
        print(
            "rho= ",
            rho,
            "  Out of Domain error during ODE Integration at the Horizon Domain",
        )

    du = [dR.real, dR.imag, dQ.real, dQ.imag]

    return du


def RHS_HD(Uo, rho, ell, w_mn, PP):
    """This function defines the form of the ODE system that determine the solution to the Master Equation
    NOTE: This is ONLY for the Horizon Domain"""

    R, Q = Uo
    dR, dQ = 0, 0

    # Integration through the Horizon Domain (HD)
    if PP.rho_H <= rho <= PP.rho_HC:

        epsilon_rho = rho - PP.rho_H

        # Integration near the Horizon
        # Using the initial condition (R = A, Q = -i w_mn R) and (dR = Q)
        # => R = A exp(-i w_mn epsilon), dR = -i w_mn A exp(-i w_mn epsilon)
        if epsilon_rho < 1.0e-6:

            dR = -1j * w_mn * R * np.exp(-1j * w_mn * epsilon_rho)
            dQ = -(w_mn ** 2) * R * np.exp(-1j * w_mn * epsilon_rho)

        # Integration not close to the Horizon
        else:
            Omega = 1.0 - rho / PP.rho_H
            rstar = rho / Omega
            xsch = rstar_to_xsch(rstar)
            rsch = 2.0 * (1.0 + xsch)
            rsch2 = rsch * rsch
            rsch3 = rsch2 * rsch

            exp_rstar_over_2M = np.exp(0.5 * rstar)
            regular_potential_factor = (
                np.exp(-(1.0 + xsch))
                * (2.0 / rsch3)
                * (ell * (ell + 1) + 2.0 * (PP.sigma_spin) / rsch)
            )

            H = 1.0 - Omega ** 2
            H_plus_one = 1.0 + H
            DH = (2.0 / PP.rho_H) * Omega
            DH_over_1minusH = (2.0 / PP.rho_H) / Omega
            Q_over_omega2 = Q / Omega ** 2
            R_over_omega2 = R / Omega ** 2

            dR = Q
            dQ = (
                DH_over_1minusH * (Q + 1j * w_mn * R)
                + 2j * w_mn * H * Q_over_omega2
                - (
                    H_plus_one * w_mn ** 2
                    - regular_potential_factor * (exp_rstar_over_2M / Omega ** 2)
                )
                * R_over_omega2
            )

    # Integration through the Horizon Transition Region
    elif PP.rho_HC < rho < PP.rho_HS:
        width = PP.rho_HC - PP.rho_HS
        sigma = 0.5 * (np.pi) * (rho - PP.rho_HS) / width
        jacobian = 0.5 * np.pi / width

        f0 = f_transition(sigma, PP.TF)
        f1 = jacobian * f_transition_1st(f0, sigma, PP.TF)
        f2 = (jacobian ** 2) * f_transition_2nd(f0, sigma, PP.TF)

        Omega = 1.0 - f0 * rho / PP.rho_H
        dOmega_drho = -(f0 + rho * f1) / PP.rho_H
        d2Omega_drho2 = -(2.0 * f1 + rho * f2) / PP.rho_H

        LH = Omega - rho * dOmega_drho
        H = 1.0 - (Omega ** 2) / LH
        one_plus_H = 1.0 + H
        one_minus_H = 1.0 - H
        DH = -(Omega / LH) * (2.0 * dOmega_drho + rho * (Omega / LH) * d2Omega_drho2)

        rstar = rho / Omega
        Vl = regge_wheeler_potential(rstar, ell)

        dR = Q
        dQ = (
            (DH / one_minus_H) * (re_QH + 1j * w_mn * R)
            + 2j * w_mn * (H / one_minus_H) * Q
            - (
                (one_plus_H / one_minus_H) * (w_mn ** 2)
                - (Vl / one_minus_H / one_minus_H)
            )
            * re_RH
        )

    # Integration through the Regular Region  (i.e. rho = rstar)
    elif rho >= PP.rho_HS:
        #    elif rho >= PP.rho_HS and rho <= PP.rho_peri:

        Vl = regge_wheeler_potential(rho, ell)

        dR = Q
        dQ = (Vl - w_mn ** 2) * R

    # Outside the Physical Integration Region
    else:

        print(
            "rho= ",
            rho,
            "  Out of Domain error during ODE Integration at the Horizon Domain",
        )

    dU = [dR, dQ]

    return dU


def RHS_OD(Uo, rho, ell, w_mn, PP):
    """This function defines the form of the ODE system that determine the solution to the Master Equation
    NOTE: This is ONLY for the Orbital Domains (Horizon or Infinity)"""

    R, Q = Uo
    Vl = regge_wheeler_potential(rho, ell)

    dR = Q
    dQ = (Vl - w_mn ** 2) * R
    dU = [dR, dQ]

    return dU


def RHS_ID(Uo, rho, ell, w_mn, PP):
    """This function defines the form of the ODE system that determine the solution to the Master Equation
    NOTE: This is ONLY for the Infinity Domain"""

    R, Q = Uo
    dR, dQ = 0, 0

    # Some common Definitions
    cb = ell * (ell + 1)

    # Integration through the Infinity Domain (ID)
    if PP.rho_IC <= rho <= PP.rho_I:

        epsilon_rho = PP.rho_I - rho

        # Integration near (null) Infinity
        if epsilon_rho < 1.0e-6:

            # Integration for the Particular Case of Zero-Frequency Modes
            if w_mn < 1.0e-8:

                sigma0 = 1.0

                sigma1 = (sigma0 / ((ell + 1) * PP.rho_I)) * (
                    cb - (1.0 / PP.rho_I) * (cb - 1)
                )

                sigma2 = (
                    sigma0 / (4.0 * (ell + 1) * (ell + 3 / 2) * (PP.rho_I ** 2))
                ) * (
                    2.0
                    * (ell ** 4 + 2.0 * ell ** 3 - ell ** 2 - 4.0 * ell - 1.0)
                    * (1.0 / (PP.rho_I ** 2))
                    - 4.0
                    * (ell + 1)
                    * (ell + 3 / 2)
                    * (ell * (ell + 1) - 1.0)
                    * (1.0 / PP.rho_I)
                    + 2.0 * ell * (ell + 1) ** 2 * (ell + 3 / 2)
                )

                sigma3 = (
                    sigma0
                    / (12.0 * (ell + 2) * (ell + 1) * (ell + 3 / 2) * (PP.rho_I ** 3))
                ) * (
                    -2.0
                    * (ell ** 2 + ell - 1)
                    * (ell ** 4 + 2.0 * ell ** 3 - ell ** 2 - 8.0 * ell - 7.0)
                    * (1.0 / (PP.rho_I ** 3))
                    + 6.0
                    * (ell + 2) ** 2
                    * (ell ** 4 + 2.0 * ell ** 3 - ell ** 2 - 4.0 * ell - 1.0)
                    * (1.0 / (PP.rho_I ** 2))
                    - 6.0
                    * (ell + 2) ** 2
                    * (ell + 1)
                    * (ell + 3 / 2)
                    * (ell ** 2 + ell - 1.0)
                    * (1.0 / PP.rho_I)
                    + 2 * (ell + 2) ** 2 * (ell + 1) ** 2 * ell * (ell + 3 / 2)
                )

                Sigma = (
                    sigma0
                    + sigma1 * epsilon_rho
                    + sigma2 * epsilon_rho ** 2
                    + sigma3 * epsilon_rho ** 3
                )
                dSigmadepsilon = (
                    sigma1
                    + 2.0 * sigma2 * epsilon_rho
                    + 3.0 * sigma3 * epsilon_rho ** 2
                )
                d2Sigmadepsilon2 = 2.0 * sigma2 + 6.0 * sigma3 * epsilon_rho

                # Particular Case: ell = 0
                if ell == 0:

                    dR = -dSigmadepsilon
                    dQ = -d2Sigmadepsilon2

                # Particular Case: ell = 1
                elif ell == 1:

                    dR = -Sigma - epsilon_rho * dSigmadepsilon
                    dQ = 2.0 * dSigmadepsilon + epsilon_rho * d2Sigmadepsilon2

                # All other ell different from 0 and 1
                else:

                    dR = -(epsilon_rho ** ell) * (
                        ell * Sigma + epsilon_rho * dSigmadepsilon
                    )
                    dQ = (epsilon_rho ** (ell - 2)) * (
                        ell * (ell - 1) * Sigma
                        + 2.0 * ell * epsilon_rho * dSigmadepsilon
                        + (epsilon_rho ** 2) * d2Sigmadepsilon2
                    )

            # Integration for non-zero Frequency Modes
            else:

                r1 = 1j * w_mn * R * (1.0 - cb / (2.0 * (PP.rho_I ** 2) * w_mn ** 2))

                q1real = (
                    -(w_mn ** 2)
                    * R.real
                    * (
                        1.0
                        - cb
                        * (
                            1.0
                            - (ell ** 2 + ell + 2.0) / (4 * (PP.rho_I ** 2) * w_mn ** 2)
                        )
                        / ((PP.rho_I ** 2) * w_mn ** 2)
                    )
                )
                q1imag = (
                    -(w_mn ** 2)
                    * R.imag
                    * (1.0 - cb * (1.0 - PP.rho_I))
                    / ((PP.rho_I ** 4) * w_mn ** 3)
                )

                r2 = q1real + 1j * q1imag

                q2real = (
                    (R.real / (4.0 * (PP.rho_I ** 4)))
                    * (cb * (PP.rho_I - 1.0) + 1.0)
                    * (2.0 - (cb + 6.0) / ((PP.rho_I ** 2) * w_mn ** 2))
                )
                q2imag = (w_mn ** 3 * R.real / 2.0) * (
                    1.0
                    - (3.0 * cb - 4.0) / (2.0 * (PP.rho_I ** 2) * w_mn ** 2)
                    + cb * (3.0 * cb - 22.0) / (4.0 * (PP.rho_I ** 4) * w_mn ** 4)
                    + 4.0
                    * (1.0 + 1.5 * (cb - 1.0) * (PP.rho_I))
                    / ((PP.rho_I ** 6) * w_mn ** 4)
                    - cb
                    * (ell + 3.0)
                    * (ell - 2.0)
                    * (cb + 2.0)
                    / (8.0 * (PP.rho_I ** 6) * w_mn ** 6)
                )

                r3 = q2real + 1j * q2imag

                q3real = (w_mn ** 4 * R.real / 6.0) * (
                    1.0
                    - 2.0 * (cb - 1.0) / ((PP.rho_I ** 2) * w_mn ** 2)
                    + (cb * (3.0 * cb + 2.0) + 24.0)
                    / (2.0 * (PP.rho_I ** 4) * w_mn ** 4)
                    + 12.0
                    * (cb - 1.0)
                    * (3.0 / ((PP.rho_I ** 2) * w_mn ** 2) - 1.0)
                    / ((PP.rho_I ** 5) * w_mn ** 4)
                    - 8.0 / ((PP.rho_I ** 6) * w_mn ** 4)
                    - cb
                    * (ell ** 4 + 2.0 * ell ** 3 - 7.0 * ell ** 2 - 8.0 * ell + 72.0)
                    / (2.0 * (PP.rho_I ** 6) * w_mn ** 6)
                    + (cb * (3.0 * cb - 2.0) + 27.0) / ((PP.rho_I ** 8) * w_mn ** 6)
                    + cb
                    * (ell + 3.0)
                    * (ell + 4.0)
                    * (ell - 2.0)
                    * (ell - 3.0)
                    * (cb + 2.0)
                    / (16.0 * (PP.rho_I ** 8) * w_mn ** 8)
                )
                q3imag = (R.real / 6.0) * (
                    -4.0 * cb * w_mn / (PP.rho_I ** 3)
                    + 4.0 * w_mn * (cb - 1.0) / (PP.rho_I ** 4)
                    + 4.0 * cb * (ell + 3.0) * (ell - 2.0) / ((PP.rho_I ** 5) * w_mn)
                    - 4.0
                    * (ell + 4.0)
                    * (ell - 3.0)
                    * (cb - 1.0)
                    / ((PP.rho_I ** 6) * w_mn)
                    + 48.0 / ((PP.rho_I ** 7) * w_mn)
                    - cb * (cb ** 2 - 18.0) / ((PP.rho_I ** 7) * w_mn ** 3)
                    + (cb - 1.0) * (cb ** 2 - 18.0) / ((PP.rho_I ** 8) * w_mn ** 3)
                )

                r4 = q3real + 1j * q3imag

                dR = (
                    -r1
                    + r2 * epsilon_rho
                    + r3 * epsilon_rho ** 2
                    + r4 * epsilon_rho ** 3
                )
                dre_QI_drho = (
                    -q1real
                    - 2.0 * q2real * epsilon_rho
                    - 3.0 * q3real * epsilon_rho ** 2
                )
                dim_QI_drho = (
                    -q1imag
                    - 2.0 * q2imag * epsilon_rho
                    - 3.0 * q3imag * epsilon_rho ** 2
                )

                dQ = dre_QI_drho + 1j * dim_QI_drho

        # Integration not "close" to (null) Infinity
        else:

            Omega = 1.0 - rho / PP.rho_I
            rstar = rho / Omega
            xsch = rstar_to_xsch(rstar)
            rsch = 2.0 * (1.0 + xsch)
            rsch2 = rsch * rsch
            rsch3 = rsch2 * rsch

            f = 1.0 - 2.0 / rsch
            regular_potential_factor = f * (
                ell * (ell + 1) + 2.0 * (PP.sigma_spin) / rsch
            )
            romega2 = (rho - 2.0 * Omega * np.log(xsch)) ** 2

            H = 1.0 - Omega ** 2
            H_plus_one = 1.0 + H
            DH = (2.0 / PP.rho_I) * Omega
            DH_over_1minusH = (2.0 / PP.rho_H) / Omega

            R_over_omega2 = R / Omega ** 2
            Q_over_omega2 = Q / Omega ** 2
            dR = Q

            dQ = (
                DH_over_1minusH * (Q - 1j * w_mn * R)
                - 2j * w_mn * H * Q_over_omega2
                - (H_plus_one * w_mn ** 2 - regular_potential_factor / romega2)
                * R_over_omega2
            )

    elif rho > PP.rho_IS and rho < PP.rho_IC:
        width = PP.rho_IC - PP.rho_IS
        sigma = 0.5 * (np.pi) * (rho - PP.rho_IS) / width
        jacobian = 0.5 * np.pi / width

        f0 = f_transition(sigma, PP.TF)
        f1 = jacobian * f_transition_1st(f0, sigma, PP.TF)
        f2 = (jacobian ** 2) * f_transition_2nd(f0, sigma, PP.TF)

        Omega = 1.0 - f0 * rho / PP.rho_I
        dOmega_drho = -(f0 + rho * f1) / PP.rho_I
        d2Omega_drho2 = -(2.0 * f1 + rho * f2) / PP.rho_I

        LI = Omega - rho * dOmega_drho
        H = 1.0 - (Omega ** 2) / LI
        one_plus_H = 1.0 + H
        one_minus_H = 1.0 - H
        DH = -(Omega / LI) * (2.0 * dOmega_drho + rho * (Omega / LI) * d2Omega_drho2)

        rstar = rho / Omega
        Vl = regge_wheeler_potential(rstar, ell)

        dR = Q

        dQ = (
            (DH / one_minus_H) * (Q - 1j * w_mn * R)
            - 2j * w_mn * H / one_minus_H * Q
            - ((one_plus_H / one_minus_H) * w_mn ** 2 - Vl / one_minus_H ** 2) * R
        )

    elif rho <= PP.rho_IS:
        #    elif rho <= PP.rho_IS and rho >= PP.rho_apo:
        Vl = regge_wheeler_potential(rho, ell)

        dR = Q
        dQ = (Vl - w_mn ** 2) * R

    else:
        print(
            "rho= ",
            rho,
            "  Out of Domain error during ODE Integration at the Horizon Domain",
        )

    dU = [dR, dQ]

    return dU


def compute_mode(ell, mm, nn, PP):
    """main routine to compute a given mode, solution of the master equation [note: see description at the beginning of the file]
    the mode is determined by the "quantum" numbers (l,m,n), as well as by the physical parameters contained in the object 'pp''
    of the class 'class_SF_Physics.py'."""

    # Printing some Information
    if PP.BC_at_particle == "R":
        print(
            "\n\nSolving the Linear Problem imposing Dirichlet Boundary Conditions on: R\n"
        )
    else:
        print(
            "\n\nSolving the Linear Problem imposing Dirichlet Boundary Conditions on: Q\n"
        )

    # Initializing Some Parameters:
    w_mn = mm * (PP.omega_phi) + nn * (PP.omega_r)  # Mode Frequency

    A_MINUS = 1.0 + 0j  # "Initial" Conditions at Horizon
    A_PLUS = 1.0 + 0j  # "Initial" Conditions at Infinity

    cb = ell * (ell + 1)  # Frequent combination of ell

    # Zero-Frequency Modes
    if mm == 0 and nn == 0:

        R_at_H = A_MINUS

        epsilon = PP.rho_H_plus - PP.rho_H
        Omega = 1.0 - PP.rho_H_plus / PP.rho_H
        rstar = PP.rho_H_plus / Omega

        Vl = regge_wheeler_potential(rstar, ell)

        R_HD_H = R_at_H * (1.0 + Vl)
        Q_HD_H = R_at_H * (PP.rho_H ** 2) * Vl / epsilon

        Uo = [R_HD_H, Q_HD_H]

        ODE_SOL_HD = solve_ivp(
            RHS_HD_zero_freq,
            [PP.rho_H_plus, PP.rho_peri],
            Uo,
            # method="Radau",
            t_eval=PP.rho_HD,
            args=(ell, PP),
            rtol=1.0e-12,
            atol=1.0e-14,
        )

        ODE_SOL_HTD = solve_ivp(
            RHS_HD_zero_freq,
            [PP.rho_H_plus, PP.rho_peri],
            Uo,
            # method="Radau",
            t_eval=PP.rho_HTD,
            args=(ell, PP),
            rtol=1.0e-12,
            atol=1.0e-14,
        )

        both = np.concatenate((PP.rho_HD, PP.rho_HTD[1:]))

        ODE_SOL_both = solve_ivp(
            RHS_HD_zero_freq,
            [PP.rho_H_plus, PP.rho_peri],
            Uo,
            # method="Radau",
            t_eval=both,
            args=(ell, PP),
            rtol=1.0e-12,
            atol=1.0e-14,
        )

        PP.single_R_HD = ODE_SOL_HD.y[0]
        PP.single_Q_HD = ODE_SOL_HD.y[1]

        print(ODE_SOL_HD.t)
        print(ODE_SOL_HD.t.shape)
        print(ODE_SOL_HD.message)
        print(ODE_SOL_HD.nfev)
        print(ODE_SOL_HD.njev)
        print()

        print(ODE_SOL_HTD.t)
        print(ODE_SOL_HTD.t.shape)
        print(ODE_SOL_HTD.message)
        print(ODE_SOL_HTD.nfev)
        print(ODE_SOL_HTD.njev)
        print()

        print(ODE_SOL_both.t)
        print(ODE_SOL_both.t.shape)
        print(ODE_SOL_both.message)
        print(ODE_SOL_both.nfev)
        print(ODE_SOL_both.njev)
        print()

        ODE_SOL_HD = solve_ivp(
            RHS_HD_zero_freq,
            [PP.rho_H_plus, PP.rho_HC],
            Uo,
            # method="Radau",
            t_eval=PP.rho_HD,
            args=(ell, PP),
            rtol=1.0e-12,
            atol=1.0e-14,
        )

        print(ODE_SOL_HD.t)
        print(ODE_SOL_HD.t.shape)
        print(ODE_SOL_HD.message)
        print(ODE_SOL_HD.nfev)
        print(ODE_SOL_HD.njev)
        print()

        ODE_SOL_HD = solve_ivp(
            RHS_HD_zero_freq,
            [PP.rho_HC, PP.rho_HS],
            Uo,
            # method="Radau",
            t_eval=PP.rho_HTD,
            args=(ell, PP),
            rtol=1.0e-12,
            atol=1.0e-14,
        )

        print(ODE_SOL_HD.t)
        print(ODE_SOL_HD.t.shape)
        print(ODE_SOL_HD.message)
        print(ODE_SOL_HD.nfev)
        print(ODE_SOL_HD.njev)
        print()

        limits = np.linspace(-2.5, 2.5, 100)
        ODE_SOL_HD = solve_ivp(
            RHS_HD_zero_freq,
            [PP.rho_HC, PP.rho_peri],
            Uo,
            # method="Radau",
            t_eval=limits,
            args=(ell, PP),
            rtol=1.0e-12,
            atol=1.0e-14,
        )

        print(ODE_SOL_HD.t)
        print(ODE_SOL_HD.t.shape)
        print(ODE_SOL_HD.message)
        print(ODE_SOL_HD.nfev)
        print(ODE_SOL_HD.njev)
        print()

        return ODE_SOL_HD

    # NON-Zero Frequency Modes:
    # else:

    # Solving the ODEs for (R,Q) at the HORIZON Domain [rho_H, rho_peri]:
    # NOTE: The initial values for (R,Q) are the boundary conditions at the Horizon:
    # NOTE: The initial value for R is arbitrary and is the equivalent of prescribing R or Q at rho_apo
    # NOTE: That is, we can rescale R so that we get the right value to satisfy the Jump Conditions.
    R_HD_H = A_MINUS
    Q_HD_H = -1j * w_mn * R_HD_H

    Uo = [R_HD_H, Q_HD_H]

    ODE_SOL_HD = odeint(
        RHS_HD,
        Uo,
        PP.rho_HD,
        args=(ell, w_mn, PP),
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    ODE_SOL_HD = solve_ivp(
        RHS_HD,
        [PP.rho_H, PP.rho_peri],
        Uo,
        method="Radau",
    )

    PP.single_R_HD = ODE_SOL_HD[:, 0]
    PP.single_Q_HD = ODE_SOL_HD[:, 1]

    # Solving the ODEs for (R,Q) at the Orbital HORIZON Domain [rho_H, rho_peri]:
    # NOTE: The initial values for (R,Q) are the values at 'pericenter' that come from the previous integration:
    R_HOD_peri = PP.single_R_HD[PP.N_HD]
    Q_HOD_peri = PP.single_Q_HD[PP.N_HD]

    Uo = [R_HOD_peri, Q_HOD_peri]

    ODE_SOL_HOD = odeint(
        RHS_OD,
        Uo,
        PP.rho_HOD,
        args=(
            ell,
            w_mn,
            PP,
        ),
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    PP.single_R_HOD = ODE_SOL_HOD[:, 0]
    PP.single_Q_HOD = ODE_SOL_HOD[:, 1]

    # Solving the ODEs for (R,Q) at the INFINITY Domain [rho_apo, rho_I]:
    # NOTE: The initial values for (R,Q) are the boundary conditions at Infinity:
    # NOTE: The initial value for R is arbitrary and is the equivalent of prescribing R or Q at rho_apo
    # NOTE: That is, we can rescale R so that we get the right value to satisfy the Jump Conditions.
    # NOTE: CONTRARY TO WHAT HAPPENS WITH THE HORIZON REGION, AT INFINITY THE POTENTIAL HAS AN INFLUENCE
    #       AND MODIFIES THE EXPECTED SIMPLE SOMMERFELD OUTGOING BOUNDARY CONDITIONS [SEE NOTES FOR DETAILS]
    # NOTE: Since we integrate from the (null) Infinity to the Apocenter Location we have to reverse the Arrays.

    if w_mn < 1.0e-8:

        if ell == 0:
            R_ID_I = A_PLUS
            Q_ID_I = -R_ID_I / (PP.rho_I ** 2)

        else:
            R_ID_I = 0.0

            if ell == 1:
                Q_ID_I = A_PLUS

            else:
                Q_ID_I = 0.0

    else:
        R_ID_I = A_PLUS
        Q_ID_I = 1j * w_mn * R_ID_I * (1.0 - cb / (2.0 * (PP.rho_I ** 2) * w_mn ** 2))

    Uo = [R_ID_I, Q_ID_I]

    ODE_SOL_ID = odeint(
        RHS_ID,
        Uo,
        PP.rho_ID,
        args=(
            ell,
            w_mn,
            PP,
        ),
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    PP.single_R_ID = ODE_SOL_ID[:, 0]
    PP.single_Q_ID = ODE_SOL_ID[:, 1]

    # Solving the ODEs for (R,Q) at the Orbital INFINITY Domain [rho_apo, rho_I]:
    # NOTE: The Integration started from rho_I in the previous step and stopped at rho_apo.
    # NOTE: The Initial Values for (R,Q) are precisely the values at 'Apocenter' that come
    #       from the previous Integration:
    R_IOD_apo = PP.single_R_ID[PP.N_ID]
    Q_IOD_apo = PP.single_Q_ID[PP.N_ID]

    Uo = [R_IOD_apo, Q_IOD_apo]

    ODE_SOL_IOD = odeint(
        RHS_OD,
        Uo,
        PP.rho_IOD,
        args=(
            ell,
            w_mn,
            PP,
        ),
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    PP.single_R_IOD = ODE_SOL_IOD[:, 0]
    PP.single_Q_IOD = ODE_SOL_IOD[:, 1]

    return
