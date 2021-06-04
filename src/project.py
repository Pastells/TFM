"""FRED Code (c) 2012-2021 C.F. Sopuerta"""

import sys
import time
import logging
import traceback
import argparse
import pandas as pd
import numpy as np
from class_SF_Physics import Physical_Quantities
from utils import run_basic_tests, show_parameters, value_at_a_point


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

    # Preparing Files for saving Self-Force Results
    resfilename = filename[:-4] + "_results.csv"

    return df, resfilename, args


# ---------------------------------------------------------------------


def main(df, resfilename, args):
    """Execute main program"""
    start_time = time.time()  # Initialize clock
    N_runs = df.shape[0]
    for run in range(N_runs):
        PP = main_run(df, run, resfilename, args)


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
    sys.exit()


# ---------------------------------------------------------------------


def r_p(X, PP):
    return X * (PP.r_apo - PP.r_peri) / 2 + (PP.r_apo + PP.r_peri) / 2


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

if __name__ == "__main__":
    # Init is executed before importing julia for fast argparse help

    start_time = time.time()  # Initialize clock

    df, resfilename, args = init()

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
