import logging
import numpy as np
from scipy import special
import matplotlib.pyplot as plt


# This function computes the spectral coefficients associated with the collocation values
# (associated with a Chebyshev-Lobatto Grid) of a given function
def compute_spectral_coefficients_real(Ui):

    # Estimating the Number of Collocation Points:
    N = Ui.size - 1

    # Allocating Array for Spectral Coefficients:
    An = np.zeros(N + 1)

    # Computing Spectral Coefficients:
    for nn in range(0, N + 1, 1):
        An[nn] = Ui[0] + ((-1) ** nn) * Ui[N]

        for kk in range(1, N, 1):
            An[nn] = An[nn] + 2.0 * (np.cos((np.pi) * kk * nn / N)) * Ui[kk]
        if nn == 0:
            An[nn] = 0.5 * An[nn] / N
        elif nn == N:
            An[nn] = 0.5 * ((-1) ** N) * An[nn] / N
        else:
            An[nn] = ((-1) ** nn) * An[nn] / N

    return An


# This function computes the spectral coefficients associated with the collocation values
# (associated with a Chebyshev-Lobatto Grid) of a given function
def compute_spectral_coefficients_complex(Ui):

    # Estimating the Number of Collocation Points:
    N = Ui.size - 1

    # Allocating Array for Spectral Coefficients:
    An = np.zeros(N + 1, dtype=np.complex128)

    # Computing Spectral Coefficients:
    for nn in range(0, N + 1, 1):
        An[nn] = Ui[0] + ((-1) ** nn) * Ui[N]

        for kk in range(1, N, 1):
            An[nn] = An[nn] + 2.0 * (np.cos((np.pi) * kk * nn / N)) * Ui[kk]
        if nn == 0:
            An[nn] = 0.5 * An[nn] / N
        elif nn == N:
            An[nn] = 0.5 * ((-1) ** N) * An[nn] / N
        else:
            An[nn] = ((-1) ** nn) * An[nn] / N

    return An


# This function evaluates a spectral function at a given spectral point given
# the value of the Spectral Coordinate and the set of Spectral Coefficients of the Function.
def value_at_a_point(X, An):

    # Estimating the Number of Spectral Coefficients/Collocation Points:
    N = An.size - 1

    # Initializing the value
    Value_at_Point = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, N + 1, 1):

        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)

        Value_at_Point = Value_at_Point + An[ns] * T_nn_X

    return Value_at_Point


# This function returns the value of the integration of a given integrand over half period of time
# (from 0 to T_r/2). The integration technique uses a spectral method based on a Chebyshev-Lobatto Grid
def spectral_time_integration(integrand, T_r):

    #  Array Size:
    N = integrand.size - 1

    if np.remainder(N, 2) == 0:
        Nmax = N // 2
    else:
        Nmax = (N - 1) // 2

    # Allocating Array for Spectral Coefficients:
    An = np.zeros(N + 1, dtype=np.complex128)

    #  Computing Spectral Coefficients of the integrand
    An = compute_spectral_coefficients_complex(integrand)

    #  Computing Integral
    integral_value = 0.0

    for k in range(0, Nmax + 1, 1):
        integral_value = integral_value - An[2 * k] / (4 * k ** 2 - 1)
    integral_value = 0.5 * T_r * integral_value

    return integral_value


# Testing consistency of the Run Parameters:
def run_basic_tests(DF, run):

    if DF.e_orbit[run] < 1.0e-14:
        logging.error("This version of the FRED Code computes the Self-Force for\n")
        logging.error("Eccentric Orbits. For Circular Orbits use an adapted version\n")
        logging.error("of the FRED Code (version 1.2).  Thanks!\n")
        fred_goodbye()

    if DF.N_time[run] % 2 != 0:
        # print("The number of Points in the Orbital Domain, 'N_OD', must be even.\n")
        logging.error(
            "The number of Collocation Points in Time, 'N_time', must be even."
        )
        fred_goodbye()

    if DF.BC_at_particle[run] != "R" and DF.BC_at_particle[run] != "Q":
        logging.error("The Parameter 'BC_at_particle' has an illegal value.\n")
        logging.error("It must be either 'R' or 'Q'. Thanks!\n")
        fred_goodbye()

    return


# Computing the Fourier Modes of the Jumps [induced by the presence of the Particle]:
def Jump_Value(ll, mm, nf, PP):

    # Computing the Jump Global Coefficient [ONLY lm dependence]
    # [NOTE: The factor 8.0 instead of 4.0 is because the integration routine 'spectral_time_integration' uses the interval (0,T_r/2), not (0,T_r)]
    gen_coeff = -8.0 * np.pi * PP.particle_charge * PP.Ep * PP.d_lm[ll, mm] / PP.T_r

    # Computing the Fourier mode of the Jump for (ll,mm,nf)
    basic_integrand = (
        PP.r_p_f
        / (PP.r_p_f ** 2 + PP.Lp ** 2)
        * np.exp(-1j * mm * (PP.phi_p_f - PP.omega_phi * PP.t_p_f))
        * np.exp(1j * nf * PP.omega_r * PP.t_p_f)
    )

    Jump_lmn = gen_coeff * spectral_time_integration(basic_integrand, PP.T_r)

    return Jump_lmn


# Printing the Different Run Parameters:
def show_parameters(PP, run):
    # if __name__ == "__main__":
    logging.info("------------------------------------------------------------")
    logging.info(f"-----------  PARAMETERS FOR RUN # {run} -----------------------")
    logging.info(f"Max_ell                = {PP.ell_max}")
    logging.info(f"N_HD                   = {PP.N_HD}")
    logging.info(f"N_OD                   = {PP.N_OD}")
    logging.info(f"N_ID                   = {PP.N_ID}")
    logging.info(f"N_Fourier              = {PP.N_Fourier}")
    logging.info(f"field_spin             = {PP.field_spin}")
    logging.info(f"mass_ratio             = {PP.mass_ratio}")
    logging.info(f"particle_charge        = {PP.particle_charge}")
    logging.info(f"e_orbit                = {PP.e_orbit}")
    logging.info(f"p_orbit                = {PP.p_orbit}")
    logging.info(f"rho_horizon            = {PP.rho_H}")
    logging.info(f"rho_H_C                = {PP.rho_HC}")
    logging.info(f"rho_H_S                = {PP.rho_HS}")
    logging.info(f"rho_peri               = {PP.rho_peri}")
    logging.info(f"rho_apo                = {PP.rho_apo}")
    logging.info(f"rho_I_S                = {PP.rho_IS}")
    logging.info(f"rho_I_C                = {PP.rho_IC}")
    logging.info(f"rho_infinity           = {PP.rho_I}")
    logging.info(f"q_transition           = {PP.TF.q}")
    logging.info(f"s_transition           = {PP.TF.s}")
    logging.info("------------------------------------------------------------")
    logging.info("-----------  EXTRA INFORMATION  ----------------------------")
    logging.info("Radial Period:  T_r    = {PP.T_r}")
    logging.info("Azimutal Period: T_phi = {PP.T_phi}")
    return


#  This funtion just says Goodbye :-)
def fred_goodbye():
    # if __name__ == "__main__":
    logging.info("Thanks for using the FRED Code (version py-v3.0)")
    logging.info("FRED Code (c) 2012-2021 C.F. Sopuerta")
    logging.info("Goodbye!")
    quit()


#  This functions plots the whole Minus and Plus Sectors separately
def plotall(CGC, PP):

    # PLOT of R_H versus r_H
    plt.plot(CGC.r_H, PP.single_R_H, label="H Interval")
    plt.plot(CGC.r_H_tr, PP.single_R_H_tr, label="H_tr Interval")
    plt.plot(CGC.r_H_pv, PP.single_R_H_pv, label="H_pv Internal")
    plt.plot(CGC.r_H_pp, PP.single_R_H_pp, label="H_pp Interval")

    plt.xlabel("r_H")
    plt.ylabel("R_H")

    plt.title("FRED Plot of R_H(r_H)")
    plt.legend()
    plt.show()

    # PLOT of R_H versus rs_H
    plt.plot(CGC.rs_H, PP.single_R_H, label="H Interval")
    plt.plot(CGC.rs_H_tr, PP.single_R_H_tr, label="H_tr Interval")
    plt.plot(CGC.rs_H_pv, PP.single_R_H_pv, label="H_pv Internal")
    plt.plot(CGC.rs_H_pp, PP.single_R_H_pp, label="H_pp Interval")

    plt.xlabel("rs_H")
    plt.ylabel("R_H")

    plt.title("FRED Plot of R_H(rs_H)")
    plt.legend()
    plt.show()

    # PLOT of R_I versus r_I
    plt.plot(CGC.r_I, PP.single_R_I, label="I Interval")
    plt.plot(CGC.r_I_tr, PP.single_R_I_tr, label="I_tr Interval")
    plt.plot(CGC.r_I_pv, PP.single_R_I_pv, label="I_pv Internal")
    plt.plot(CGC.r_I_pp, PP.single_R_I_pp, label="I_pp Interval")

    plt.xlabel("r_I")
    plt.ylabel("R_I")

    plt.title("FRED Plot of R_I(r_I)")
    plt.legend()
    plt.show()

    # PLOT of Q_H versus r_H
    plt.plot(CGC.r_H, PP.single_Q_H, label="H Interval")
    plt.plot(CGC.r_H_tr, PP.single_Q_H_tr, label="H_tr Interval")
    plt.plot(CGC.r_H_pv, PP.single_Q_H_pv, label="H_pv Internal")
    plt.plot(CGC.r_H_pp, PP.single_Q_H_pp, label="H_pp Interval")

    plt.xlabel("r_H")
    plt.ylabel("Q_H")

    plt.title("FRED Plot of Q_H(r_H)")
    plt.legend()
    plt.show()

    # PLOT of Q_H versus rs_H
    plt.plot(CGC.rs_H, PP.single_Q_H, label="H Interval")
    plt.plot(CGC.rs_H_tr, PP.single_Q_H_tr, label="H_tr Interval")
    plt.plot(CGC.rs_H_pv, PP.single_Q_H_pv, label="H_pv Internal")
    plt.plot(CGC.rs_H_pp, PP.single_Q_H_pp, label="H_pp Interval")

    plt.xlabel("rs_H")
    plt.ylabel("Q_H")

    plt.title("FRED Plot of Q_H(rs_H)")
    plt.legend()
    plt.show()

    # PLOT of Q_I versus r_I
    plt.plot(CGC.r_I, PP.single_Q_I, label="I Interval")
    plt.plot(CGC.r_I_tr, PP.single_Q_I_tr, label="I_tr Interval")
    plt.plot(CGC.r_I_pv, PP.single_Q_I_pv, label="I_pv Internal")
    plt.plot(CGC.r_I_pp, PP.single_Q_I_pp, label="I_pp Interval")

    plt.xlabel("r_I")
    plt.ylabel("Q_I")

    plt.title("FRED Plot of Q_I(r_I)")
    plt.legend()
    plt.show()

    # PLOT of R versus r
    plt.plot(CGC.r_H, PP.single_R_H, label="H Interval")
    plt.plot(CGC.r_H_tr, PP.single_R_H_tr, label="H_tr Interval")
    plt.plot(CGC.r_H_pv, PP.single_R_H_pv, label="H_pv Internal")
    plt.plot(CGC.r_H_pp, PP.single_R_H_pp, label="H_pp Interval")
    plt.plot(CGC.r_I, PP.single_R_I, label="I Interval")
    plt.plot(CGC.r_I_tr, PP.single_R_I_tr, label="I_tr Interval")
    plt.plot(CGC.r_I_pv, PP.single_R_I_pv, label="I_pv Internal")
    plt.plot(CGC.r_I_pp, PP.single_R_I_pp, label="I_pp Interval")

    plt.xlabel("r")
    plt.ylabel("R_000(r)")

    plt.title("FRED Plot of R(r)")
    plt.legend()
    plt.show()

    # PLOT of Q versus r
    plt.plot(CGC.r_H, PP.single_Q_H, label="H Interval")
    plt.plot(CGC.r_H_tr, PP.single_Q_H_tr, label="H_tr Interval")
    plt.plot(CGC.r_H_pv, PP.single_Q_H_pv, label="H_pv Internal")
    plt.plot(CGC.r_H_pp, PP.single_Q_H_pp, label="H_pp Interval")
    plt.plot(CGC.r_I, PP.single_Q_I, label="I Interval")
    plt.plot(CGC.r_I_tr, PP.single_Q_I_tr, label="I_tr Interval")
    plt.plot(CGC.r_I_pv, PP.single_Q_I_pv, label="I_pv Internal")
    plt.plot(CGC.r_I_pp, PP.single_Q_I_pp, label="I_pp Interval")

    plt.xlabel("r")
    plt.ylabel("Q_000(r)")

    plt.title("FRED Plot of Q(r)")
    plt.legend()
    plt.show()

    return


#  This functions makes a single plot from the Data given
def plotsomething(X, Y):

    plt.plot(X, Y, label="Y=Y(X)")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title("FRED Plot")

    plt.legend()

    plt.show()

    return


#  This functions makes a multiple plot from the Data given
def plotseveral(arg1, *Pargs):

    n = 0
    for argf in Pargs:
        plt.plot(arg1, argf, label="Plot " + str(n))
        n += 1

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title("FRED Plot")

    plt.legend()

    plt.show()

    # for np in range(1,nplots+1,1):

    return
