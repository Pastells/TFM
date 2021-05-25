"""Functions related to the Schwarzschild metric"""

import numpy as np
from scipy import optimize
from scipy import special


def r_schwarzschild_to_r_tortoise(r_sch):
    """Gives the value of the Tortoise Radial Coordinate given a value of the
    Schwarzachild radial coordinate.   This Routine assumes units in which the Black Hole Mass is
    the unity."""

    r_tortoise = r_sch + 2.0 * np.log(0.5 * r_sch - 1.0)

    return r_tortoise


def x_schwarzschild_to_r_tortoise(x_sch, r_tortoise):
    """Given a value of the following function of the Schwarzachild radial coordinate
    returns the value of the Tortoise Radial Coordinate:
    [assuming units in which the Black Hole Mass is the unity]:
         Xsch = Rsch/2 - 1"""

    dr_tortoise = 2.0 * (1.0 + x_sch + np.log(x_sch)) - r_tortoise

    return dr_tortoise


def r_tortoise_to_x_schwarzschild(rstar):
    """This function gives the value of the Schwarzachild radial coordinate given a value of the
    Tortoise Radial coordinate.   This Routine assumes units in which the Black Hole Mass is the unity."""

    if rstar < 0.1:
        xsch_guess = np.exp(0.5 * rstar - 1.0)

        if xsch_guess < 1.0e-200:
            xsch = 0.0
        else:
            xsch = optimize.newton(
                x_schwarzschild_to_r_tortoise,
                xsch_guess,
                fprime=lambda x, rstar: 2.0 + 2.0 / x,
                args=(rstar,),
                fprime2=lambda x, rstar: -2.0 / (x ** 2),
            )

    elif rstar >= 0.1 and rstar <= 7.6:
        xsch_guess = 3.0 - np.sqrt(8.0 - rstar)
        xsch = optimize.newton(
            x_schwarzschild_to_r_tortoise,
            xsch_guess,
            fprime=lambda x, rstar: 2.0 + 2.0 / x,
            args=(rstar,),
            fprime2=lambda x, rstar: -2.0 / (x ** 2),
        )

    else:
        xsch_guess = 0.5 * rstar - 1.0 - np.log(0.5 * rstar - 1.0)
        xsch = optimize.newton(
            x_schwarzschild_to_r_tortoise,
            xsch_guess,
            fprime=lambda x, rstar: 2.0 + 2.0 / x,
            args=(rstar,),
            fprime2=lambda x, rstar: -2.0 / (x ** 2),
        )

    return xsch


def r_tortoise_to_r_schwarzschild(rstar):
    """Gives the value of the Schwarzschild radial coordinate given a value of the
    Tortoise Radial coordinate.   This Routine assumes units in which the Black Hole Mass is the unity."""

    xsch = r_tortoise_to_x_schwarzschild(rstar)
    r_sch = 2.0 * (xsch + 1.0)

    return r_sch


def rstar_to_xsch(rstar):
    """This function gives the value of the Schwarzachild radial coordinate given a value of the
    Tortoise Radial coordinate.   This Routine assumes units in which the Black Hole Mass is the unity.
    The Routine uses the Lambert function
    NOTE: Not working properly, some x_sch are zero"""

    xstar = 0.5 * rstar - 1.0
    x_sch = special.lambertw(np.exp(xstar), k=0, tol=1e-16)
    if np.isnan(x_sch):
        print("Nan in rstar_to_xsch")
        x_sch = r_tortoise_to_x_schwarzschild(rstar)

    return np.real(x_sch)


def rstar_to_rsch(rstar):
    """This function gives the value of the Schwarzschild radial coordinate given a value of the
    Tortoise Radial coordinate.   This Routine assumes units in which the Black Hole Mass is the unity.
    The Routine uses the Lambert function"""

    x_sch = rstar_to_xsch(rstar)
    r_sch = 2.0 * (x_sch + 1.0)

    return np.real(r_sch)


# TODO check all _p_at_x functions


# TODO: move to class
def obtain_coefs(PP):
    # Coefficients from values at known positions
    n = PP.N_time
    for k in range(n + 1):
        # Vector with the n-th Chebyshev Polynomials at the given spectral coordinate:
        # T_i_x = np.array([special.eval_chebyt(i, PP.r_p_f[k]) for i in range(n + 1)])
        T_i_x = np.array([special.eval_chebyt(i, PP.Xt[k]) for i in range(1, n + 2)])
        PP.Ai_t_p_f += PP.t_p_f[k] * T_i_x
        PP.Ai_r_p_f += PP.r_p_f[k] * T_i_x
        PP.Ai_rs_p_f += PP.rs_p_f[k] * T_i_x
        PP.Ai_phi_p_f += PP.phi_p_f[k] * T_i_x
        PP.Ai_chi_p_f += PP.chi_p_f[k] * T_i_x

    # Normalization
    PP.Ai_t_p_f = PP.Ai_t_p_f * 2 / (n + 1)
    PP.Ai_r_p_f = PP.Ai_r_p_f * 2 / (n + 1)
    PP.Ai_rs_p_f = PP.Ai_rs_p_f * 2 / (n + 1)
    PP.Ai_phi_p_f = PP.Ai_phi_p_f * 2 / (n + 1)
    PP.Ai_chi_p_f = PP.Ai_chi_p_f * 2 / (n + 1)


def non_spectral_t_phi(PP):
    # Values at extremes
    obtain_coefs(PP)
    PP.t_p[0] = PP.t_p_f[0]
    PP.phi_p[0] = PP.phi_p_f[0]
    PP.t_p[PP.N_OD] = PP.t_p_f[PP.N_time]
    PP.phi_p[PP.N_OD] = PP.phi_p_f[PP.N_time]

    # Values at spectral grid
    for nn in range(PP.N_OD + 1):
        PP.t_p[nn] = 0
        PP.phi_p[nn] = 0
        for jj in range(PP.N_time + 1):
            T_jj_x = special.eval_chebyt(jj, PP.r_p[nn])
            PP.t_p[nn] += PP.Ai_t_p_f[jj] * T_jj_x
            PP.phi_p[nn] += PP.Ai_phi_p_f[jj] * T_jj_x


def eval_at_x(PP, var_str, x):
    """Evaluate variable at an arbitrary value of the Time Spectral Coordinate
    var_str: string with name of variable"""
    var_x = 0.0

    # Coefficients for vairable
    Ai_var = getattr(PP, "Ai_" + var_str + "_f")

    # Adding the contribution of each Spectral Mode:
    for ii in range(PP.N_time + 1):
        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_ii_x = special.eval_chebyt(ii, x)
        var_x += Ai_var[ii] * T_ii_x

    return var_x


def zero_of_r_p_at_x(x, r_goal, PP):
    """Used by the scipy scalar minimization routine to find the x that correspond to a given
    value of the Schwarzschild radial coordinate r, called 'r_goal'"""
    r_p = eval_at_x(PP, "r_p", x)
    # abs_error = np.abs(a - r_goal)
    # print(x, r_p, r_goal, abs_error)
    error = r_p - r_goal
    return error
