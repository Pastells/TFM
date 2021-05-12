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


# TODO check all _p_at_X functions


def t_p_at_X(X, PP):
    """This function evaluates the Time Coordinate at an arbitrary value of the Time Spectral Coordinate"""
    # Initializing the value
    t_p_X = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, PP.N_time + 1, 1):
        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)
        t_p_X = t_p_X + PP.An_t_p_f[ns] * T_nn_X

    return t_p_X


def r_p_at_X(X, PP):
    """This function evaluates the Schwarzschild Radial Coordinate at an arbitrary value of the Time Spectral Coordinate"""
    # Initializing the value
    r_p_X = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, PP.N_time + 1, 1):
        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)
        r_p_X = r_p_X + PP.An_r_p_f[ns] * T_nn_X

    return r_p_X


def zero_of_r_p_at_X(X, r_goal, PP):
    """This function is defined to be used by the scipy scalar minimization routine to find the X that correspond to a given
    value of the Schwarzschild radial coordinate r, called 'r_goal'"""

    return np.abs(r_p_at_X(X, PP) - r_goal)


def rs_p_at_X(X, PP):
    """This function evaluates the Tortoise Radial Coordinate at an arbitrary value of the Time Spectral Coordinate"""
    # Initializing the value
    rs_p_X = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, PP.N_time + 1, 1):
        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)
        rs_p_X = rs_p_X + PP.An_rs_p_f[ns] * T_nn_X

    return rs_p_X


def chi_p_at_X(X, PP):
    """This function evaluates the Angle associated with the PERIODIC radial motion of bounded geodesics at an arbitrary value of the Time Spectral Coordinate"""
    # Initializing the value
    chi_p_X = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, PP.N_time + 1, 1):
        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)
        chi_p_X = chi_p_X + PP.An_chi_p_f[ns] * T_nn_X

    return chi_p_X


def phi_p_at_X(X, PP):
    """This function evaluates the Azimuthal Coordinate at an arbitrary value of the Time Spectral Coordinate"""
    # Initializing the value
    phi_p_X = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, PP.N_time + 1, 1):
        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)
        phi_p_X = phi_p_X + PP.An_phi_p_f[ns] * T_nn_X

    return phi_p_X
