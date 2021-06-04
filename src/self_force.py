"""FRED Code (c) 2012-2021 C.F. Sopuerta"""

import os
import sys
import time
import datetime
import logging
import traceback
import pickle
import argparse
import pandas as pd
import numpy as np
from scipy import special
from class_SF_Physics import Physical_Quantities
from utils import (
    run_basic_tests,
    show_parameters,
    fred_goodbye,
    logging_func,
    value_at_a_point,
)


# ---------------------------------------------------------------------


def parsing():
    """Parse imput arguments and return them in args object"""
    parser = argparse.ArgumentParser(description="FRED program")
    parser.add_argument("data", type=str, help="Input csv data file")
    parser.add_argument("-log_print", action="store_true", help="Print all log output")
    parser.add_argument(
        "-save",
        action="store_true",
        help="Save modes in whole horizon and infinite regions in results/ folder",
    )
    parser.add_argument(
        "-ell_init",
        type=int,
        default=0,
        help="Start computation from some ell mode. Useful to continue already started problem.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------


def init():
    """Read data, import julia, initialize parser and logger
    Returns df dataframe and resfilename"""

    args = parsing()

    # --- Read Data from Parameter File into a Pandas DataFrame [df] ---
    filename = args.data
    df = pd.read_csv(filename)

    # logging
    logging_func(filename, args.log_print)

    # Preparing Files for saving Self-Force Results
    resfilename = filename[:-4] + "_results.csv"
    if os.path.isfile(resfilename):
        logging.error(f"{resfilename} already exists, save with other name or remove")
        fred_goodbye()

    with open(resfilename, "w") as resultsfile:
        resultsfile.write(
            f'# FRED RESULTS FILE [File opened on {time.strftime("%Y-%m-%d %H:%M")}]\n'
        )

    return df, resfilename, args


# ---------------------------------------------------------------------


def main(df, resfilename, args):
    """Execute main program"""

    start_time = time.time()  # Initialize clock

    N_runs = df.shape[0]

    # Starting the different Runs in the Parameter File 'data'
    for run in range(N_runs):
        PP = main_run(df, run, resfilename, args)
        PP.saving()

    # Final Time after Computations
    end_time = time.time()
    logging.info("Execution Time: %f seconds", end_time - start_time)

    fred_goodbye()


# ---------------------------------------------------------------------


def main_run(df, run, resfilename, args):
    """Perform computation for given run number"""
    # Set Clock to measure the Computational Time
    run_start_time = time.time()

    # Checking the Run Parameters
    run_basic_tests(df, run)

    # Setting up Physical Quantities
    PP = Physical_Quantities(df, run, args.save)
    logging.info("FRED RUN %d: Class Physical Quantities Created", run)
    show_parameters(PP, run)  # Show parameters of the Run

    # Projecting the geodesic into the Particle Domains (from Horizon and to Infinity)
    project_geodesic(PP, run)

    # Computation of the Singular Part of the Self-Force:
    singular_part(PP, run)

    # NOTE: Big Loop starts Here
    # Computing ell-Modes
    for ll in range(args.ell_init, PP.ell_max + 1):  # Harmonic Number l
        for mm in range(ll + 1):  # Harmonic Number m

            # Check whether ell+m is odd (contribution is zero, so it can be skipped)
            if (ll + mm) % 2 == 1:
                continue

            # Computing Fourier Modes (n-Modes):
            n_estimated_error = 10.0 * PP.Mode_accuracy
            nf = 0  # Fourier Mode Number
            while nf < PP.N_Fourier and n_estimated_error > PP.Mode_accuracy:
                do_mode(ll, mm, nf, PP, run, args.save)

                n_estimated_error = np.amax(PP.Estimated_Error)
                logging.info(
                    "FRED RUN %d: Mode (l,m,n) = (%d, %d, %02d) Computed (error=%.2e)",
                    run,
                    ll,
                    mm,
                    nf,
                    n_estimated_error,
                )
                nf += 1  # Increase the Fourier Mode Counter

            # Complete m-Mode Computation and add contribution to l-Mode
            PP.complete_m_mode(ll, mm)

        # After having finished the Computation of an l-Mode we need to substract the Contribution from the Singular field:
        PP.complete_l_mode(ll)

    # Once the Computation of the Self-Force has ended we save/print the results:
    run_prints(PP, run, resfilename, run_start_time)
    return PP


# ---------------------------------------------------------------------


def do_mode(ll, mm, nf, PP, run, save):
    """Computing the Bare Field Mode with Frequency omega_mn
    [REMEMBER: Psi is the scalar field and Phi its radial (tortoise) derivative]"""
    # TODO change name of function

    compute_mode(ll, mm, nf, PP, save=save)

    indices = (ll, mm, nf + PP.N_Fourier)

    # Store computed modes from compute_mode
    PP.store(indices)

    PP.rescale_mode(ll, mm, nf)

    # For nf != 0 there are both positive and negative frequencies
    if nf > 0:
        do_mode(ll, mm, -nf, PP, run, save)


# ---------------------------------------------------------------------


def project_geodesic(PP, run):
    """Project the geodesic into the Particle Domains (from Horizon to Infinity)"""
    PP.t_p[0] = PP.t_p_f[0]
    PP.phi_p[0] = PP.phi_p_f[0]
    PP.t_p[PP.N_OD] = PP.t_p_f[PP.N_time]
    PP.phi_p[PP.N_OD] = PP.phi_p_f[PP.N_time]

    rs_apo = PP.r_apo - 2 * np.log(0.5 * PP.r_apo - 1)
    rs_peri = PP.r_peri - 2 * np.log(0.5 * PP.r_peri - 1)

    for nn in range(1, PP.N_OD):
        x_v = (2 * PP.rs_p[nn] - rs_apo - rs_peri) / (rs_apo - rs_peri)
        PP.t_p[nn] = value_at_a_point(x_v, PP.An_t_p_f)
        PP.phi_p[nn] = value_at_a_point(x_v, PP.An_phi_p_f)

    logging.info(
        "FRED RUN %d: Projection of the geodesic onto the Radial Spatial Domain Done",
        run,
    )


# ---------------------------------------------------------------------


def singular_part(PP, run):
    """Computation of the Singular Part of the Self-Force"""

    # Schwarzschild Metric Function f at tbe Particle Location:
    f_p = 1 - 2 / PP.r_p

    # Computing rdot:
    q1 = PP.p_orbit / PP.r_p
    q3 = PP.p_orbit - 2 * q1
    q4 = np.sqrt((PP.p_orbit - 2) ** 2 - 4 * PP.e_orbit ** 2)
    q5 = q3 * (q1 ** 2) / (q4 * PP.p_orbit)

    chi_dot = q5 * (np.sqrt(PP.p_orbit - 4 - 2 * q1)) / (PP.p_orbit)

    dr_p_dtau = (
        PP.Ep
        * PP.r_p ** 2
        * PP.e_orbit
        * np.sin(PP.chi_p)
        * chi_dot
        / (f_p * PP.p_orbit)
    )

    # Computing other relevant quantities:
    qaux1 = 1 + PP.Lp ** 2 / PP.r_p ** 2
    qaux2 = qaux1 ** (1.5)
    alpha = PP.Lp ** 2 / (PP.r_p ** 2 + PP.Lp ** 2)

    # Computing the Value of some Hypergeometric Functions:
    ff_1d2 = special.hyp2f1(0.5, 0.5, 1, alpha)
    ff_m1d2 = special.hyp2f1(-0.5, 0.5, 1, alpha)

    # Computing Regularization Parameters, not dependent on ell:
    A_r_H = PP.Ep / (f_p * qaux1)
    A_r_I = -PP.Ep / (f_p * qaux1)

    B_r = (
        (dr_p_dtau ** 2 - 2 * PP.Ep ** 2) * ff_1d2
        + (dr_p_dtau ** 2 + PP.Ep ** 2) * ff_m1d2
    ) / (2 * f_p * qaux2)

    # Computing Singular Part of the Self-Force (only the radial component at the moment):
    for ll in range(PP.ell_max + 1):
        PP.SF_S_r_l_H[ll] = PP.charge * ((ll + 0.5) * A_r_H + B_r) / (PP.r_p ** 2)
        PP.SF_S_r_l_I[ll] = PP.charge * ((ll + 0.5) * A_r_I + B_r) / (PP.r_p ** 2)

    logging.info("FRED RUN %d: Singular Part of the Self-Force Computed", run)


# ---------------------------------------------------------------------


def run_prints(PP, run, resfilename, run_start_time):
    """Printing results"""
    # fmt: off
    # Taking end time of the Run and computing total time for the Run:
    run_end_time = time.time()
    run_total_time = run_end_time - run_start_time
    str_time = str(datetime.timedelta(seconds=run_total_time))

    # Printing the l-components of the Bare Self-Force
    ns = 3
    print("\n\nResults for the l-Components of the Bare Self-Force at time t = ", PP.t_p[ns])
    for ll in range(PP.ell_max + 1):
        print("l=", ll, " FF- =", np.real(PP.SF_F_r_l_H[ll, ns]), " FF+ =", np.real(PP.SF_F_r_l_I[ll, ns]),
              " FS- =", np.real(PP.SF_S_r_l_H[ll, ns]), " FS+ =", np.real(PP.SF_S_r_l_I[ll, ns]),
              " FR- =", np.real(PP.SF_R_r_l_H[ll, ns]), " FR+ =", np.real(PP.SF_R_r_l_I[ll, ns]),
             )

    # Total number of (l m)-Modes
    number_modes = int((PP.ell_max + 1) * (PP.ell_max + 2) / 2)

    # Saving the results of the Computation into a File:
    with open(resfilename, "a") as resultsfile:
        resultsfile.write(f"# RUN NUMBER:  {run}\n")
        resultsfile.write("# RUN INFO:\n")
        resultsfile.write(f"#   Maximum Harmonic Number:                       ell = {PP.ell_max}\n")
        resultsfile.write(f"#   Number of Modes to be computed:            N_modes = {number_modes}\n")
        resultsfile.write(f"#   Number of Collocation Points per Domain:         N = {2*PP.N_OD+1}\n")
        resultsfile.write(f"#   Orbital Eccentricity:                            e = {PP.e_orbit}\n")
        resultsfile.write(f"#   Orbital Semilatus Rectum:                        p = {PP.p_orbit}\n")
        resultsfile.write(f"#   Total Run Time:                                 Tc = {str_time}\n#\n")
        resultsfile.write("#   r_p            F_r-            F_r+\n")
        for ns in range(PP.N_OD):
            resultsfile.write(f"{PP.r_p[ns]:.6f},{np.real(PP.SF_r_H[ns]):.14f},{np.real(PP.SF_r_I[ns]):.14f}\n")

    # fmt: on


# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Init is executed before importing julia for fast argparse help

    start_time = time.time()  # Initialize clock

    df, resfilename, args = init()

    # --- julia imports ---
    try:
        import julia

        # jl = julia.Julia(compiled_modules=False, depwarn=True)
        jl = julia.Julia(compiled_modules=False, depwarn=True, sysimage="sysimage.so")
        from julia import Main

        Main.include("src/mode_comp.jl")
        compute_mode = Main.eval("compute_mode")  # global function
    except Exception as ex:
        logging.error("Error importing julia")
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)

    logging.info("julia imports done")
    # ------------------------------------------------------
    setup_time = time.time()
    logging.info("Setup Time: %d seconds", setup_time - start_time)

    try:
        main(df, resfilename, args)
    except Exception as ex:
        logging.error("Error in main program")
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
