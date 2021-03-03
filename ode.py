import sys
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad
from scipy.optimize import fsolve
from scipy.integrate import odeint, complex_ode, ode


def regge_wheeler_pot(r, ll):
    """Regge-Wheeler potential for spin-0 fields"""
    M = 1
    return (1 - 2 * M / r) * (ll * (ll + 1) / r ** 2 + 2 * M / r ** 3)


def obtain_frequencies(M, E, L, e):
    """Angular frequencies for radial and azimuthal coordinates"""
    omega_r = 1
    omega_phi = 2
    return omega_r, omega_phi


# --------------------------------------------------


def radius_from_rho(rho, sign):
    """Get r coordinate from r*(rho)"""
    M = 1

    r_tort = rho / Omega_pm(rho, sign)

    def tort(r, r_tort):
        """We want to find the zero of this function"""
        if r / (2 * M) <= 1:
            func = 1e5
        else:
            func = (r_tort - (r + 2 * M * np.log(r / (2 * M) - 1))) ** 2
        return func

    r = fsolve(tort, 3 * M, r_tort)[0]
    return r


# --------------------------------------------------


def Omega_pm(rho, sign):
    """Compactification function"""
    S_min = -5
    rho_T_min = -3
    R_min = 0
    R_plus = 3
    rho_T_plus = 5
    S_plus = 10
    s = 1
    q = 1

    def sigma(rho, rho_i, rho_f):
        return np.pi / 2 * (rho - rho_i) / (rho_f - rho_i)

    def dsigma(rho, rho_i, rho_f):
        return np.pi / 2 / (rho_f - rho_i)

    def F_T(sigma):
        # return 0.5 * (1 + np.tanh(s / np.pi * (np.tan(sigma) - q ** 2 / np.tan(sigma))))
        return 0.5 * (1 + jnp.tanh(sigma))

    def dF_T(sigma):
        """From wolfram alpha"""
        return (
            0.159155
            * s
            * (q ** 2 / np.sin(sigma) ** 2 + 1 / np.cos(sigma) ** 2)
            * 1
            / np.cosh(s * (np.tan(sigma) - q ** 2 / np.sin(sigma)) / np.pi)
        )

    if S_min <= rho < rho_T_min:
        F = 1
    elif rho_T_min <= rho < R_min:
        F = F_T(sigma(rho, rho_T_min, R_min))
    elif R_min <= rho < R_plus:
        F = 0
    elif R_plus <= rho < rho_T_plus:
        F = F_T(sigma(rho, R_plus, rho_T_plus))
    elif rho_T_plus <= rho < S_plus:
        F = 1
    else:
        raise (ValueError)

    if sign == +1:
        return 1 - F * rho / S_plus
    elif sign == -1:
        return 1 - F * rho / S_min
    else:
        raise (ValueError)


# --------------------------------------------------


def H_func(rho, sign):
    """1-drho/dr*"""
    Omega, dOmega = value_and_grad(Omega_pm, 0)(rho, sign)
    return 1 - Omega ** 2 / (Omega - rho * dOmega)


# --------------------------------------------------


def derivatives(rho, y, sign, ll, omega):
    """Return derivatives of R_lmn and Q_lmn = dR_lmn"""
    R, Q = y[0], y[1]
    r = radius_from_rho(rho, sign)
    V = regge_wheeler_pot(r, ll)
    H, dH = value_and_grad(H_func, 0)(rho, sign)
    dR = Q
    dQ = (
        dH / (1 - H) * (Q + sign * 1j * omega * R)
        + 2j * sign * omega * H / (1 - H) * Q
        - ((1 + H) / (1 - H) * omega ** 2 - V / (1 - H) ** 2) * R
    )
    return [dR, dQ]


# --------------------------------------------------


def main():
    sign = -1
    ll, m, n = 0, 0, 0
    omega_r, omega_phi = obtain_frequencies(0, 0, 0, 0)
    omega = n * omega_r + m * omega_phi

    # Initial conditions
    R = 1
    y0 = (R, sign * 1j * omega * R)
    rho0 = -4.0
    rho1 = 9.5
    drho = 0.1

    result = ode(derivatives).set_integrator("zvode", method="bdf")
    result.set_initial_value(y0, rho0).set_f_params(sign, ll, omega)
    while result.successful() and result.t < rho1:
        print(result.t + drho, result.integrate(result.t + drho))


# --------------------------------------------------

if __name__ == "__main__":
    main()
