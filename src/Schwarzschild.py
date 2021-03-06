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
