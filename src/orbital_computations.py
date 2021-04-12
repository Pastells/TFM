import numpy as np
from scipy import integrate


def integrand_radial_period(chi, p_orbit, e_orbit):
    """This function returns the value of the integrand that defines the Orbital Radial Period."""
    p2 = p_orbit ** 2
    e2 = e_orbit ** 2

    oneplusecoschi = 1.0 + e_orbit * (np.cos(chi))
    oneplusecoschi2 = oneplusecoschi ** 2

    dt_dchi = p2 * (
        np.sqrt(
            ((p_orbit - 2.0) ** 2 - 4.0 * e2) / (p_orbit - 4.0 - 2.0 * oneplusecoschi)
        )
        / (oneplusecoschi2 * (p_orbit - 2.0 * oneplusecoschi))
    )

    return dt_dchi


def integrand_azimuthal_frequency(chi, p_orbit, e_orbit):
    """This function returns the value of the integrand that defines the Azimuthal Frequency."""
    psqrt = np.sqrt(p_orbit)
    argsqrt = p_orbit - 6.0 - 2.0 * e_orbit * (np.cos(chi))

    dphi_dchi = psqrt / (np.sqrt(argsqrt))

    return dphi_dchi


def compute_radial_period(p_orbit, e_orbit):
    """This function provides the values of the radial periods of the orbital motion."""
    T_r_int, error = integrate.quad(
        integrand_radial_period,
        0.0,
        np.pi,
        args=(p_orbit, e_orbit),
        epsabs=1.0e-14,
        epsrel=1.0e-13,
    )
    T_r = 2.0 * T_r_int

    if __name__ == "__main__":
        if error > 1.0e-12:
            print(
                "WARNING: There may be a significant error in the Computation of the Radial Period"
            )

    return T_r


def compute_azimuthal_frequency(p_orbit, e_orbit, T_r):
    """This function provides the values of the azimuthal frequency of the orbital motion."""
    omega_phi_int, error = integrate.quad(
        integrand_azimuthal_frequency,
        0.0,
        np.pi,
        args=(p_orbit, e_orbit),
        epsabs=1.0e-14,
        epsrel=1.0e-13,
    )

    omega_phi = 2.0 * omega_phi_int / T_r

    return omega_phi


def sco_odes_rhs(y, t, p_orbit, e_orbit):
    """This function defines the form of the ODE system that determine the SCO orbits around the MBH."""
    chi_p, phi_p = y

    dchi_pdt = (
        (p_orbit - 2.0 * (1.0 + e_orbit * (np.cos(chi_p))))
        * ((1.0 + e_orbit * (np.cos(chi_p))) ** 2)
        * (
            np.sqrt(
                (p_orbit - 6.0 - 2.0 * (e_orbit) * (np.cos(chi_p)))
                / ((p_orbit - 2.0) ** 2 - 4.0 * (e_orbit ** 2))
            )
            / (p_orbit ** 2)
        )
    )

    dphi_pdt = (
        (p_orbit - 2.0 * (1.0 + e_orbit * (np.cos(chi_p))))
        * ((1.0 + e_orbit * (np.cos(chi_p))) ** 2)
        / (np.sqrt((p_orbit - 2.0) ** 2 - 4.0 * (e_orbit) ** 2) * ((p_orbit) ** (1.5)))
    )

    dy_dt = [dchi_pdt, dphi_pdt]

    return dy_dt
