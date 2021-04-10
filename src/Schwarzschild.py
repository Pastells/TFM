import numpy as np
from scipy import optimize
from scipy import special

# This function gives the value of the Tortoise Radial Coordinate given a value of the
# Schwarzachild radial coordinate.   This Routine assumes units in which the Black Hole Mass is
# the unity.
def r_schwarzschild_to_r_tortoise(r_sch):

    r_tortoise = r_sch + 2.0 * np.log(0.5 * r_sch - 1.0)

    return r_tortoise


# This function, given a value of the following function of the Schwarzachild radial coordinate
# [assuming units in which the Black Hole Mass is the unity]:
#
#      Xsch = Rsch/2 - 1
#
# returns the value of the Tortoise Radial Coordinate:
def x_schwarzschild_to_r_tortoise(x_sch, r_tortoise):

    dr_tortoise = 2.0 * (1.0 + x_sch + np.log(x_sch)) - r_tortoise

    return dr_tortoise


# This function gives the value of the Schwarzschild radial coordinate given a value of the
# Tortoise Radial coordinate.   This Routine assumes units in which the Black Hole Mass is the unity.
def r_tortoise_to_r_schwarzschild(rstar):

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

    r_sch = 2.0 * (xsch + 1.0)

    return r_sch


# This function gives the value of the Schwarzachild radial coordinate given a value of the
# Tortoise Radial coordinate.   This Routine assumes units in which the Black Hole Mass is the unity.
def r_tortoise_to_x_schwarzschild(rstar):

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


# This function gives the value of the Schwarzschild radial coordinate given a value of the
# Tortoise Radial coordinate.   This Routine assumes units in which the Black Hole Mass is the unity.
# The Routine uses the Lambert function
def rstar_to_rsch(rstar):

    xstar = 0.5 * rstar - 1.0
    x_sch = special.lambertw(np.exp(xstar), k=0, tol=1e-16)

    r_sch = 2.0 * (x_sch + 1.0)

    return np.real(r_sch)


# This function gives the value of the Schwarzachild radial coordinate given a value of the
# Tortoise Radial coordinate.   This Routine assumes units in which the Black Hole Mass is the unity.
# The Routine uses the Lambert function
def rstar_to_xsch(rstar):

    xstar = 0.5 * rstar - 1.0
    x_sch = special.lambertw(np.exp(xstar), k=0, tol=1e-16)

    return np.real(x_sch)


# This function evaluates the Time Coordinate at an arbitrary value of the Time Spectral Coordinate
def t_p_at_X(X, PP):

    # Initializing the value
    t_p_X = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, PP.N_time + 1, 1):

        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)

        t_p_X = t_p_X + PP.An_t_p_f[ns] * T_nn_X

    return t_p_X


# This function evaluates the Schwarzschild Radial Coordinate at an arbitrary value of the Time Spectral Coordinate
def r_p_at_X(X, PP):

    # Initializing the value
    r_p_X = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, PP.N_time + 1, 1):

        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)

        r_p_X = r_p_X + PP.An_r_p_f[ns] * T_nn_X

    return r_p_X


# This function is defined to be used by the scipy scalar minimization routine to find the X that correspond to a given
# value of the Schwarzschild radial coordinate r, called 'r_goal'
def zero_of_r_p_at_X(X, r_goal, PP):

    return np.abs(r_p_at_X(X, PP) - r_goal)


# This function evaluates the Tortoise Radial Coordinate at an arbitrary value of the Time Spectral Coordinate
def rs_p_at_X(X, PP):

    # Initializing the value
    rs_p_X = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, PP.N_time + 1, 1):

        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)

        rs_p_X = rs_p_X + PP.An_rs_p_f[ns] * T_nn_X

    return rs_p_X


# This function evaluates the Angle associated with the PERIODIC radial motion of bounded geodesics at an arbitrary value of the Time Spectral Coordinate
def chi_p_at_X(X, PP):

    # Initializing the value
    chi_p_X = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, PP.N_time + 1, 1):

        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)

        chi_p_X = chi_p_X + PP.An_chi_p_f[ns] * T_nn_X

    return chi_p_X


# This function evaluates the Azimuthal Coordinate at an arbitrary value of the Time Spectral Coordinate
def phi_p_at_X(X, PP):

    # Initializing the value
    phi_p_X = 0.0

    # Adding the contribution of each Spectral Mode:
    for ns in range(0, PP.N_time + 1, 1):

        # Value of the n-th Chebyshev Polynomial at the given spectral coordinate:
        T_nn_X = special.eval_chebyt(ns, X)

        phi_p_X = phi_p_X + PP.An_phi_p_f[ns] * T_nn_X

    return phi_p_X
