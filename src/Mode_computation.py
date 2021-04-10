import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from Schwarzschild import rstar_to_xsch, r_tortoise_to_x_schwarzschild
from hyperboloidal_compactification_tanh import (
    f_transition,
    f_transition_1st,
    f_transition_2nd,
)

####################################################################################################################
#
# NOTE: def compute_mode(ell,omega,CGC,PP):
#
# This function computes a Single Scalar Field Mode determined by the numbers (l,m,n) at the two domains
# (the Horizon one and the Spatial Infinity one).
#
# The arguments are:
#
#     * The Harmonic Number 'l'
#     * The Mode Frequency 'w_mn' (which depends on the other two "quantum" numbers 'm' and 'n')
#     * PP [an object that contains a number of physical quantities required for the computation].
#
# NOTE: The dependence of the mode on 'm' and 'n' comes in the frequency 'omega'. The extra dependence on these
# numbers as well as on other physical input (particle's trajectory) comes from the jumps. But the jumps do not
# enter in this computation since here we are solving for generic boundary conditions at pericenter and apocenter.
#
# NOTE: DOMAIN STRUCTURE: We have two Regions that intersect in one Domain, the Particle Domain [r_peri, r_apo]
# The structure of each Region (Horizon and Infinity) is:
#
#       HD             H_tr           HOD
#  o-------------o=============o--------------o
#
#                              o--------------o=============o-------------o
#                                     IOD          I_tr            ID
# where:
#
#           HD: Horizon Domain touching the BH Horizon. It contains a Transition Region, here denoted as H_tr
#           HOD: Horizon Orbital Domain, the Particle Domain
#           IOD: Infinity Orbital Domain, the Particle Domain. It is diffeomorphic to HOD.
#           ID: Infinity Domain touching Future Null Infinity
#
# NOTATION: The original field equations at the two sides of the particle are:
# NOTE: '-' corresponds to the Horizon Region and '+' to the Infinity Region
#
#           ( -D^2_t + D^2_r* - V_RW_l(r) ) Psi^{+,-}_lm = 0
#
# We introduce the following change of variables (VPhi = varphi):
#
#           Phi^{+,-}_lm = (1+H) D_tau Psi^{+,-}_lm + H D_rho Psi^{+,-}_lm
#           VPhi^{+,-}_lm = D_rho Psi^{+,-}_lm
#
# In the Frequency Domain:
#
#           Psi^{+,-}_lm = Sum_n(-oo,+oo) exp{-i omega_mn tau} R^{+,-}_lmn(rho)
#           VPhi^{+,-}_lm = Sum_n(-oo,+oo) exp{-i omega_mn tau} Q^{+,-}_lmn(rho)
#
# The function 'compute_mode' solves the ODEs for R^{+,-}_lmn and Q^{+,-}_lmn
#
# The Boundary Condition at the horizon is:
#
#           ( D_t - D_r* ) Psi^{-}_lm = 0
#
# and the Boundary Condition at Spatial Infinity is:
#
#           ( D_t + D_r* ) Psi^{+}_lm = 0
#
# These conditions can be translated into conditions for the Fourier Modes:
#
#           i omega_mn R^{-}_lmn + Q^{-}_lmn = 0
#
#           i omega_mn R^{+}_lmn - Q^{+}_lmn = 0
#
# Our strategy is to find first solutions for the Fourier Modes prescribing arbitrary Boundary Conditions
# at the pericenter (r_peri) and at the apocenter (r_apo).  This Boundary Conditions are Dirichlet
# Boundary Conditions. That is:
#
#           \hat{R}^{-}_lmn(r_apo) = A_MINUS
#
#           \hat{R}^{+}_lmn(r_peri) = A_PLUS
#
####################################################################################################################

# This function defines the form of the ODE system that determine the solution to the Master Equation
# It returns the Right-Hand-Side (RHS) of the ODEs
# NOTE: This is ONLY for the Horizon Domain
# NOTE: This is ONLY for the Zero Frequency Modes


def RHS_HD_zero_freq(rho, Uo, ell, PP):

    re_RH, im_RH, re_QH, im_QH = Uo

    dre_RH_drho = 0.0
    dim_RH_drho = 0.0
    dre_QH_drho = 0.0
    dim_QH_drho = 0.0

    # Useful Quantities
    cb = ell * (ell + 1)

    # Integration through the Horizon Domain (HD)
    if PP.rho_H <= rho <= PP.rho_HC:

        Omega = 1.0 - rho / PP.rho_H
        rstar = rho / Omega
        xstar = 0.5 * rstar - 1.0
        xsch = rstar_to_xsch(rstar)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch

        H = 1.0 - Omega ** 2
        # DH = (2.0/PP.rho_H)*Omega
        DH_over_1minusH = (2.0 / PP.rho_H) / Omega
        one_minus_Hsq = Omega ** 4

        Vl = 2.0 * np.exp(xstar - xsch) * (cb + 2.0 / rsch) / (rsch ** 3)

        rhs_reQ_term1 = DH_over_1minusH * re_QH
        rhs_reQ_term2 = (Vl / one_minus_Hsq) * re_RH

        rhs_imQ_term1 = DH_over_1minusH * im_QH
        rhs_imQ_term2 = (Vl / one_minus_Hsq) * im_RH

        dre_RH_drho = re_QH
        dim_RH_drho = im_QH
        dre_QH_drho = rhs_reQ_term1 + rhs_reQ_term2
        dim_QH_drho = rhs_imQ_term1 + rhs_imQ_term2

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
        # one_plus_H = 1.0 + H
        one_minus_H = 1.0 - H
        DH = -(Omega / LH) * (2.0 * dOmega_drho + rho * (Omega / LH) * d2Omega_drho2)
        DH_over_1minusH = DH / one_minus_H
        one_minus_Hsq = one_minus_H ** 2

        rstar = rho / Omega
        xstar = 0.5 * rstar - 1.0
        xsch = rstar_to_xsch(rstar)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch

        Vl = 2.0 * np.exp(xstar - xsch) * (cb + 2.0 / rsch) / (rsch ** 3)

        rhs_reQ_term1 = DH_over_1minusH * re_QH
        rhs_reQ_term2 = (Vl / one_minus_Hsq) * re_RH

        rhs_imQ_term1 = DH_over_1minusH * im_QH
        rhs_imQ_term2 = (Vl / one_minus_Hsq) * im_RH

        dre_RH_drho = re_QH
        dim_RH_drho = im_QH
        dre_QH_drho = rhs_reQ_term1 + rhs_reQ_term2
        dim_QH_drho = rhs_imQ_term1 + rhs_imQ_term2

    # Integration through the Regular Region  (i.e. rho = rstar)
    elif rho >= PP.rho_HS:
        #    elif rho >= PP.rho_HS and rho <= PP.rho_peri:

        xstar = 0.5 * rho - 1.0
        xsch = rstar_to_xsch(rho)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch

        Vl = xsch * (2.0 / rsch3) * (cb + 2.0 / rsch)

        dre_RH_drho = re_QH
        dim_RH_drho = im_QH
        dre_QH_drho = Vl * re_RH
        dim_QH_drho = Vl * im_RH

    # Outside the Physical Integration Region
    else:

        print(
            "rho= ",
            rho,
            "  Out of Domain error during ODE Integration at the Horizon Domain",
        )

    dU_drho = [dre_RH_drho, dim_RH_drho, dre_QH_drho, dim_QH_drho]

    return dU_drho


# This function computes the Jacobian of the Right-Hand-Side (RHS) of the ODEs associated with the Master Equation
# NOTE: This is ONLY for the Horizon Domain
# NOTE: This is ONLY for the Zero Frequency Modes


def Jac_RHS_HD_zero_freq(rho, Uo, ell, PP):

    # Useful Quantities
    cb = ell * (ell + 1)

    # Integration through the Horizon Domain (HD)
    if PP.rho_H <= rho <= PP.rho_HC:

        Omega = 1.0 - rho / PP.rho_H
        rstar = rho / Omega
        xstar = 0.5 * rstar - 1.0
        xsch = rstar_to_xsch(rstar)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch

        Vl = 2.0 * np.exp(xstar - xsch) * (cb + 2.0 / rsch) / (rsch ** 3)

        H = 1.0 - Omega ** 2
        # DH = (2.0/PP.rho_H)*Omega
        DH_over_1minusH = (2.0 / PP.rho_H) / Omega
        one_minus_Hsq = Omega ** 4

        J_rRrR = 0.0
        J_rRiR = 0.0
        J_rRrQ = 1.0
        J_rRiQ = 0.0

        J_iRrR = 0.0
        J_iRiR = 0.0
        J_iRrQ = 0.0
        J_iRiQ = 1.0

        J_rQrR = Vl / one_minus_Hsq
        J_rQiR = 0.0
        J_rQrQ = DH_over_1minusH
        J_rQiQ = 0.0

        J_iQrR = 0.0
        J_iQiR = Vl / one_minus_Hsq
        J_iQrQ = 0.0
        J_iQiQ = DH_over_1minusH

    # Integration through the Horizon Transition Region
    elif rho > PP.rho_HC and rho < PP.rho_HS:

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
        # one_plus_H = 1.0 + H
        one_minus_H = 1.0 - H
        DH = -(Omega / LH) * (2.0 * dOmega_drho + rho * (Omega / LH) * d2Omega_drho2)
        DH_over_1minusH = DH / one_minus_H
        one_minus_Hsq = one_minus_H ** 2

        rstar = rho / Omega
        xstar = 0.5 * rstar - 1.0
        xsch = rstar_to_xsch(rstar)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch

        Vl = 2.0 * np.exp(xstar - xsch) * (cb + 2.0 / rsch) / (rsch ** 3)

        J_rRrR = 0.0
        J_rRiR = 0.0
        J_rRrQ = 1.0
        J_rRiQ = 0.0

        J_iRrR = 0.0
        J_iRiR = 0.0
        J_iRrQ = 0.0
        J_iRiQ = 1.0

        J_rQrR = Vl / one_minus_Hsq
        J_rQiR = 0.0
        J_rQrQ = DH_over_1minusH
        J_rQiQ = 0.0

        J_iQrR = 0.0
        J_iQiR = Vl / one_minus_Hsq
        J_iQrQ = 0.0
        J_iQiQ = DH_over_1minusH

    # Integration through the Regular Region  (i.e. rho = rstar)
    elif rho >= PP.rho_HS:
        #    elif rho >= PP.rho_HS and rho <= PP.rho_peri:

        xstar = 0.5 * rho - 1.0
        xsch = rstar_to_xsch(rho)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch

        Vl = xsch * (2.0 / rsch3) * (cb + 2.0 / rsch)

        J_rRrR = 0.0
        J_rRiR = 0.0
        J_rRrQ = 1.0
        J_rRiQ = 0.0

        J_iRrR = 0.0
        J_iRiR = 0.0
        J_iRrQ = 0.0
        J_iRiQ = 1.0

        J_rQrR = Vl
        J_rQiR = 0.0
        J_rQrQ = 0.0
        J_rQiQ = 0.0

        J_iQrR = 0.0
        J_iQiR = Vl
        J_iQrQ = 0.0
        J_iQiQ = 0.0

    # Outside the Physical Integration Region
    else:

        print(
            "rho= ",
            rho,
            "  Out of Domain error during ODE Integration at the Horizon Domain",
        )

    Jacobian_RHS = np.array(
        [
            [J_rRrR, J_rRiR, J_rRrQ, J_rRiQ],
            [J_iRrR, J_iRiR, J_iRrQ, J_iRiQ],
            [J_rQrR, J_rQiR, J_rQrQ, J_rQiQ],
            [J_iQrR, J_iQiR, J_iQrQ, J_iQiQ],
        ]
    )

    return Jacobian_RHS


# This function defines the form of the ODE system that determine the solution to the Master Equation
# NOTE: This is ONLY for the Horizon Domain


def RHS_HD(Uo, rho, ell, w_mn, PP):

    re_RH, im_RH, re_QH, im_QH = Uo

    dre_RH_drho = 0.0
    dim_RH_drho = 0.0
    dre_QH_drho = 0.0
    dim_QH_drho = 0.0

    # Integration through the Horizon Domain (HD)
    if PP.rho_H <= rho <= PP.rho_HC:

        epsilon_rho = rho - PP.rho_H

        # Integration near the Horizon
        if epsilon_rho < 1.0e-6:

            dre_RH_drho = -w_mn * re_RH * np.sin(w_mn * epsilon_rho)
            dim_RH_drho = -w_mn * re_RH * np.cos(w_mn * epsilon_rho)
            dre_QH_drho = -(w_mn ** 2) * re_RH * np.cos(w_mn * epsilon_rho)
            dim_QH_drho = (w_mn ** 2) * re_RH * np.sin(w_mn * epsilon_rho)

        # Integration not close to the Horizon
        else:
            Omega = 1.0 - rho / PP.rho_H
            rstar = rho / Omega
            xsch = r_tortoise_to_x_schwarzschild(rstar)
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
            reQ_over_omega2 = re_QH / Omega / Omega
            imQ_over_omega2 = im_QH / Omega / Omega
            reR_over_omega2 = re_RH / Omega / Omega
            imR_over_omega2 = im_RH / Omega / Omega

            rhs_reQ_term1 = DH_over_1minusH * (re_QH - w_mn * im_RH)
            rhs_reQ_term2 = -2.0 * w_mn * H * imQ_over_omega2
            rhs_reQ_term3 = -H_plus_one * (w_mn ** 2) * reR_over_omega2
            rhs_reQ_term4 = (
                regular_potential_factor
                * (exp_rstar_over_2M / Omega / Omega)
                * reR_over_omega2
            )

            rhs_imQ_term1 = DH_over_1minusH * (im_QH + w_mn * re_RH)
            rhs_imQ_term2 = 2.0 * w_mn * H * reQ_over_omega2
            rhs_imQ_term3 = -H_plus_one * (w_mn ** 2) * imR_over_omega2
            rhs_imQ_term4 = (
                regular_potential_factor
                * (exp_rstar_over_2M / Omega / Omega)
                * imR_over_omega2
            )

            dre_RH_drho = re_QH
            dim_RH_drho = im_QH
            dre_QH_drho = rhs_reQ_term1 + rhs_reQ_term2 + rhs_reQ_term3 + rhs_reQ_term4
            dim_QH_drho = rhs_imQ_term1 + rhs_imQ_term2 + rhs_imQ_term3 + rhs_imQ_term4

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
        xsch = r_tortoise_to_x_schwarzschild(rstar)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch
        V_l = xsch * (2.0 / rsch3) * (ell * (ell + 1.0) + 2.0 * (PP.sigma_spin) / rsch)

        rhs_reQ_term1 = (DH / one_minus_H) * (re_QH - w_mn * im_RH)
        rhs_reQ_term2 = -2.0 * w_mn * (H / one_minus_H) * im_QH
        rhs_reQ_term3 = -(one_plus_H / one_minus_H) * (w_mn ** 2) * re_RH
        rhs_reQ_term4 = (V_l / one_minus_H / one_minus_H) * re_RH

        rhs_imQ_term1 = (DH / one_minus_H) * (im_QH + w_mn * re_RH)
        rhs_imQ_term2 = 2.0 * w_mn * (H / one_minus_H) * re_QH
        rhs_imQ_term3 = -(one_plus_H / one_minus_H) * (w_mn ** 2) * im_RH
        rhs_imQ_term4 = (V_l / one_minus_H / one_minus_H) * im_RH

        dre_RH_drho = re_QH
        dim_RH_drho = im_QH
        dre_QH_drho = rhs_reQ_term1 + rhs_reQ_term2 + rhs_reQ_term3 + rhs_reQ_term4
        dim_QH_drho = rhs_imQ_term1 + rhs_imQ_term2 + rhs_imQ_term3 + rhs_imQ_term4

    # Integration through the Regular Region  (i.e. rho = rstar)
    elif rho >= PP.rho_HS:
        #    elif rho >= PP.rho_HS and rho <= PP.rho_peri:

        xsch = r_tortoise_to_x_schwarzschild(rho)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch

        V_l = xsch * (2.0 / rsch3) * (ell * (ell + 1.0) + 2.0 * (PP.sigma_spin) / rsch)

        dre_RH_drho = re_QH
        dim_RH_drho = im_QH
        dre_QH_drho = (V_l - w_mn ** 2) * re_RH
        dim_QH_drho = (V_l - w_mn ** 2) * im_RH

    # Outside the Physical Integration Region
    else:

        print(
            "rho= ",
            rho,
            "  Out of Domain error during ODE Integration at the Horizon Domain",
        )

    dU_drho = [dre_RH_drho, dim_RH_drho, dre_QH_drho, dim_QH_drho]

    return dU_drho


# This function defines the form of the ODE system that determine the solution to the Master Equation
# NOTE: This is ONLY for the Horizon Orbital Domain


def RHS_HOD(Uo, rho, ell, w_mn, PP):

    re_RH, im_RH, re_QH, im_QH = Uo

    xsch = r_tortoise_to_x_schwarzschild(rho)
    rsch = 2.0 * (1.0 + xsch)
    rsch2 = rsch * rsch
    rsch3 = rsch2 * rsch

    V_l = xsch * (2.0 / rsch3) * (ell * (ell + 1.0) + 2.0 * (PP.sigma_spin) / rsch)

    dre_RH_drho = re_QH
    dim_RH_drho = im_QH
    dre_QH_drho = (V_l - w_mn ** 2) * re_RH
    dim_QH_drho = (V_l - w_mn ** 2) * im_RH

    dU_drho = [dre_RH_drho, dim_RH_drho, dre_QH_drho, dim_QH_drho]

    return dU_drho


# This function defines the form of the ODE system that determine the solution to the Master Equation
# NOTE: This is ONLY for the Infinity Domain


def RHS_ID(Uo, rho, ell, w_mn, PP):

    re_RI, im_RI, re_QI, im_QI = Uo

    dre_RI_drho = 0.0
    dim_RI_drho = 0.0
    dre_QI_drho = 0.0
    dim_QI_drho = 0.0

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

                    dre_RI_drho = -dSigmadepsilon
                    dim_RI_drho = 0.0
                    dre_QI_drho = -d2Sigmadepsilon2
                    dim_QI_drho = 0.0

                # Particular Case: ell = 1
                elif ell == 1:

                    dre_RI_drho = -Sigma - epsilon_rho * dSigmadepsilon
                    dim_RI_drho = 0.0
                    dre_QI_drho = 2.0 * dSigmadepsilon + epsilon_rho * d2Sigmadepsilon2
                    dim_QI_drho = 0.0

                # All other ell different from 0 and 1
                else:

                    dre_RI_drho = -(epsilon_rho ** ell) * (
                        ell * Sigma + epsilon_rho * dSigmadepsilon
                    )
                    dim_RI_drho = 0.0
                    dre_QI_drho = (epsilon_rho ** (ell - 2)) * (
                        ell * (ell - 1) * Sigma
                        + 2.0 * ell * epsilon_rho * dSigmadepsilon
                        + (epsilon_rho ** 2) * d2Sigmadepsilon2
                    )
                    dim_QI_drho = 0.0

            # Integration for non-zero Frequency Modes
            else:

                qireal = 0.0
                qiimag = w_mn * re_RI * (1.0 - cb / (2.0 * (PP.rho_I ** 2) * w_mn ** 2))

                r1real = qireal
                r1imag = qiimag

                q1real = (
                    -(w_mn ** 2)
                    * re_RI
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
                    * re_RI
                    * (1.0 - cb * (1.0 - PP.rho_I))
                    / ((PP.rho_I ** 4) * w_mn ** 3)
                )

                r2real = -0.5 * q1real
                r2imag = -0.5 * q1imag

                q2real = (
                    (re_RI / (4.0 * (PP.rho_I ** 4)))
                    * (cb * (PP.rho_I - 1.0) + 1.0)
                    * (2.0 - (cb + 6.0) / ((PP.rho_I ** 2) * w_mn ** 2))
                )
                q2imag = (w_mn ** 3 * re_RI / 2.0) * (
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

                r3real = -q2real / 3.0
                r3imag = -q2imag / 3.0

                q3real = (w_mn ** 4 * re_RI / 6.0) * (
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
                q3imag = (re_RI / 6.0) * (
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

                r4real = -0.25 * q3real
                r4imag = -0.25 * q3imag

                dre_RI_drho = (
                    -r1real
                    - 2.0 * r2real * epsilon_rho
                    - 3.0 * r3real * epsilon_rho ** 2
                    - 4.0 * r4real * epsilon_rho ** 3
                )
                dim_RI_drho = (
                    -r1imag
                    - 2.0 * r2imag * epsilon_rho
                    - 3.0 * r3imag * epsilon_rho ** 2
                    - 4.0 * r4imag * epsilon_rho ** 3
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

        # Integration not "close" to (null) Infinity
        else:

            Omega = 1.0 - rho / PP.rho_I
            rstar = rho / Omega
            xsch = r_tortoise_to_x_schwarzschild(rstar)
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

            reQ_over_omega2 = re_QI / Omega / Omega
            imQ_over_omega2 = im_QI / Omega / Omega
            reR_over_omega2 = re_RI / Omega / Omega
            imR_over_omega2 = im_RI / Omega / Omega

            rhs_reQ_term1 = DH_over_1minusH * (re_QI + w_mn * im_RI)
            rhs_reQ_term2 = 2.0 * w_mn * H * imQ_over_omega2
            rhs_reQ_term3 = -H_plus_one * (w_mn ** 2) * reR_over_omega2
            rhs_reQ_term4 = regular_potential_factor * reR_over_omega2 / romega2

            rhs_imQ_term1 = DH_over_1minusH * (im_QI - w_mn * re_RI)
            rhs_imQ_term2 = -2.0 * w_mn * H * reQ_over_omega2
            rhs_imQ_term3 = -H_plus_one * (w_mn ** 2) * imR_over_omega2
            rhs_imQ_term4 = regular_potential_factor * imR_over_omega2 / romega2

            dre_RI_drho = re_QI
            dim_RI_drho = im_QI
            dre_QI_drho = rhs_reQ_term1 + rhs_reQ_term2 + rhs_reQ_term3 + rhs_reQ_term4
            dim_QI_drho = rhs_imQ_term1 + rhs_imQ_term2 + rhs_imQ_term3 + rhs_imQ_term4

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
        xsch = r_tortoise_to_x_schwarzschild(rstar)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch
        V_l = xsch * (2.0 / rsch3) * (ell * (ell + 1.0) + 2.0 * (PP.sigma_spin) / rsch)

        rhs_reQ_term1 = (DH / one_minus_H) * (re_QI + w_mn * im_RI)
        rhs_reQ_term2 = 2.0 * w_mn * (H / one_minus_H) * im_QI
        rhs_reQ_term3 = -(one_plus_H / one_minus_H) * (w_mn ** 2) * re_RI
        rhs_reQ_term4 = (V_l / one_minus_H / one_minus_H) * re_RI

        rhs_imQ_term1 = (DH / one_minus_H) * (im_QI - w_mn * re_RI)
        rhs_imQ_term2 = -2.0 * w_mn * (H / one_minus_H) * re_QI
        rhs_imQ_term3 = -(one_plus_H / one_minus_H) * (w_mn ** 2) * im_RI
        rhs_imQ_term4 = (V_l / one_minus_H / one_minus_H) * im_RI

        dre_RI_drho = re_QI
        dim_RI_drho = im_QI
        dre_QI_drho = rhs_reQ_term1 + rhs_reQ_term2 + rhs_reQ_term3 + rhs_reQ_term4
        dim_QI_drho = rhs_imQ_term1 + rhs_imQ_term2 + rhs_imQ_term3 + rhs_imQ_term4

    elif rho <= PP.rho_IS:
        #    elif rho <= PP.rho_IS and rho >= PP.rho_apo:
        xsch = r_tortoise_to_x_schwarzschild(rho)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch

        V_l = xsch * (2.0 / rsch3) * (ell * (ell + 1.0) + 2.0 * (PP.sigma_spin) / rsch)

        dre_RI_drho = re_QI
        dim_RI_drho = im_QI
        dre_QI_drho = (V_l - w_mn ** 2) * re_RI
        dim_QI_drho = (V_l - w_mn ** 2) * im_RI

    else:
        print(
            "rho= ",
            rho,
            "  Out of Domain error during ODE Integration at the Horizon Domain",
        )

    dU_drho = [dre_RI_drho, dim_RI_drho, dre_QI_drho, dim_QI_drho]

    return dU_drho


# This function defines the form of the ODE system that determine the solution to the Master Equation
# NOTE: This is ONLY for the Infinity Orbital Domain


def RHS_IOD(Uo, rho, ell, w_mn, PP):
    re_RI, im_RI, re_QI, im_QI = Uo

    xsch = r_tortoise_to_x_schwarzschild(rho)
    rsch = 2.0 * (1.0 + xsch)
    rsch2 = rsch * rsch
    rsch3 = rsch2 * rsch

    V_l = xsch * (2.0 / rsch3) * (ell * (ell + 1.0) + 2.0 * (PP.sigma_spin) / rsch)

    dre_RI_drho = re_QI
    dim_RI_drho = im_QI
    dre_QI_drho = (V_l - w_mn ** 2) * re_RI
    dim_QI_drho = (V_l - w_mn ** 2) * im_RI

    dU_drho = [dre_RI_drho, dim_RI_drho, dre_QI_drho, dim_QI_drho]

    return dU_drho


# MAIN ROUTINE TO COMPUTE A GIVEN MODE, SOLUTION OF THE MASTER EQUATION [NOTE: SEE DESCRIPTION AT THE BEGINNING OF THE FILE]
# THE MODE IS DETERMINED BY THE "QUANTUM" NUMBERS (l,m,n), AS WELL AS BY THE PHYSICAL PARAMETERS CONTAINED IN THE OBJECT 'PP''
# OF THE CLASS 'class_SF_Physics.py'.


def compute_mode(ell, mm, nn, PP):

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

    A_MINUS = 1.0  # "Initial" Conditions at Horizon
    A_PLUS = 1.0  # "Initial" Conditions at Infinity:

    cb = ell * (ell + 1)  # Frequent combination of ell

    # Zero-Frequency Modes
    if mm == 0 and nn == 0:

        R_at_H = A_MINUS

        epsilon = PP.rho_H_plus - PP.rho_H
        Omega = 1.0 - PP.rho_H_plus / PP.rho_H
        rstar = PP.rho_H_plus / Omega
        xstar = 0.5 * rstar - 1.0
        xsch = rstar_to_xsch(rstar)
        rsch = 2.0 * (1 + xsch)

        # exp(xstar-x) = x 
        Vl = 2.0 * np.exp(xstar - xsch) * (cb + 2.0 / rsch) / (rsch ** 3)

        real_R_HD_H = R_at_H * (1.0 + Vl)
        imag_R_HD_H = 0.0
        real_Q_HD_H = R_at_H * (PP.rho_H ** 2) * Vl / epsilon
        imag_Q_HD_H = 0.0

        Uo = [real_R_HD_H, imag_R_HD_H, real_Q_HD_H, imag_Q_HD_H]

        ODE_SOL_HD = solve_ivp(
            RHS_HD_zero_freq,
            [PP.rho_H_plus, PP.rho_peri],
            Uo,
            method="Radau",
            t_eval=PP.rho_HD,
            dense_output=False,
            events=None,
            vectorized=False,
            args=(
                ell,
                PP,
            ),
            jac=Jac_RHS_HD_zero_freq,
            rtol=1.0e-12,
            atol=1.0e-14,
        )

        PP.single_R_HD = ODE_SOL_HD.y[0] + 1j * ODE_SOL_HD.y[1]
        PP.single_Q_HD = ODE_SOL_HD.y[2] + 1j * ODE_SOL_HD.y[3]

        print(ODE_SOL_HD.t)
        print(ODE_SOL_HD.t.shape)
        print(ODE_SOL_HD.message)
        print(ODE_SOL_HD.nfev)
        print(ODE_SOL_HD.njev)

        return

    # NON-Zero Frequency Modes:
    else:

        # Solving the ODEs for (R,Q) at the HORIZON Domain [rho_H, rho_peri]:
        # NOTE: The initial values for (R,Q) are the boundary conditions at the Horizon:
        # NOTE: The initial value for R is arbitrary and is the equivalent of prescribing R or Q at rho_apo
        # NOTE: That is, we can rescale R so that we get the right value to satisfy the Jump Conditions.
        real_R_HD_H = A_MINUS
        imag_R_HD_H = 0.0
        real_Q_HD_H = w_mn * imag_R_HD_H
        imag_Q_HD_H = -w_mn * real_R_HD_H

        Uo = [real_R_HD_H, imag_R_HD_H, real_Q_HD_H, imag_Q_HD_H]

        ODE_SOL_HD = odeint(
            Master_ODE_rhs_HD,
            Uo,
            PP.rho_HD,
            args=(
                ell,
                w_mn,
                PP,
            ),
            rtol=1.0e-13,
            atol=1.0e-14,
        )

        ODE_SOL_HD = solve_ivp(
            Master_ODE_rhs_HD,
            [PP.rho_H, PP.rho_peri],
            Uo,
            method="Radau",
            t_eval=None,
            dense_output=False,
            events=None,
            vectorized=False,
            args=None,
            **options
        )

        PP.single_R_HD = ODE_SOL_HD[:, 0] + 1j * ODE_SOL_HD[:, 1]
        PP.single_Q_HD = ODE_SOL_HD[:, 2] + 1j * ODE_SOL_HD[:, 3]

        # Solving the ODEs for (R,Q) at the Orbital HORIZON Domain [rho_H, rho_peri]:
        # NOTE: The initial values for (R,Q) are the values at 'pericenter' that come from the previous integration:
        real_R_HOD_peri = PP.single_R_HD[PP.N_HD].real
        imag_R_HOD_peri = PP.single_R_HD[PP.N_HD].imag
        real_Q_HOD_peri = PP.single_Q_HD[PP.N_HD].real
        imag_Q_HOD_peri = PP.single_Q_HD[PP.N_HD].imag

        Uo = [real_R_HOD_peri, imag_R_HOD_peri, real_Q_HOD_peri, imag_Q_HOD_peri]

        ODE_SOL_HOD = odeint(
            Master_ODE_rhs_HOD,
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

        PP.single_R_HOD = ODE_SOL_HOD[:, 0] + 1j * ODE_SOL_HOD[:, 1]
        PP.single_Q_HOD = ODE_SOL_HOD[:, 2] + 1j * ODE_SOL_HOD[:, 3]

    # Solving the ODEs for (R,Q) at the INFINITY Domain [rho_apo, rho_I]:
    # NOTE: The initial values for (R,Q) are the boundary conditions at Infinity:
    # NOTE: The initial value for R is arbitrary and is the equivalent of prescribing R or Q at rho_apo
    # NOTE: That is, we can rescale R so that we get the right value to satisfy the Jump Conditions.
    # NOTE: CONTRARY TO WHAT HAPPENS WITH THE HORIZON REGION, AT INFINITY THE POTENTIAL HAS AN INFLUENCE
    #       AND MODIFIES THE EXPECTED SIMPLE SOMMERFELD OUTGOING BOUNDARY CONDITIONS [SEE NOTES FOR DETAILS]
    # NOTE: Since we integrate from the (null) Infinity to the Apocenter Location we have to reverse the Arrays.

    if w_mn < 1.0e-8:

        if ell == 0:

            real_R_ID_I = A_PLUS
            imag_R_ID_I = 0.0
            real_Q_ID_I = -real_R_ID_I / (PP.rho_I ** 2)
            imag_Q_ID_I = 0.0

        else:

            real_R_ID_I = 0.0
            imag_R_ID_I = 0.0

            if ell == 1:

                real_Q_ID_I = A_PLUS
                imag_Q_ID_I = 0.0

            else:

                real_Q_ID_I = 0.0
                imag_Q_ID_I = 0.0

    else:

        real_R_ID_I = A_PLUS
        imag_R_ID_I = 0.0
        real_Q_ID_I = (
            -w_mn * imag_R_ID_I * (1.0 - cb / (2.0 * (PP.rho_I ** 2) * w_mn ** 2))
        )
        imag_Q_ID_I = (
            w_mn * real_R_ID_I * (1.0 - cb / (2.0 * (PP.rho_I ** 2) * w_mn ** 2))
        )

    Uo = [real_R_ID_I, imag_R_ID_I, real_Q_ID_I, imag_Q_ID_I]

    ODE_SOL_ID = odeint(
        Master_ODE_rhs_ID,
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

    PP.single_R_ID = ODE_SOL_ID[:, 0] + 1j * ODE_SOL_ID[:, 1]
    PP.single_Q_ID = ODE_SOL_ID[:, 2] + 1j * ODE_SOL_ID[:, 3]

    # Solving the ODEs for (R,Q) at the Orbital INFINITY Domain [rho_apo, rho_I]:
    # NOTE: The Integration started from rho_I in the previous step and stopped at rho_apo.
    # NOTE: The Initial Values for (R,Q) are precisely the values at 'Apocenter' that come
    #       from the previous Integration:
    real_R_IOD_apo = PP.single_R_ID[PP.N_ID].real
    imag_R_IOD_apo = PP.single_R_ID[PP.N_ID].imag
    real_Q_IOD_apo = PP.single_Q_ID[PP.N_ID].real
    imag_Q_IOD_apo = PP.single_Q_ID[PP.N_ID].imag

    Uo = [real_R_IOD_apo, imag_R_IOD_apo, real_Q_IOD_apo, imag_Q_IOD_apo]

    ODE_SOL_IOD = odeint(
        Master_ODE_rhs_IOD,
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

    PP.single_R_IOD = ODE_SOL_IOD[:, 0] + 1j * ODE_SOL_IOD[:, 1]
    PP.single_Q_IOD = ODE_SOL_IOD[:, 2] + 1j * ODE_SOL_IOD[:, 3]

    return
