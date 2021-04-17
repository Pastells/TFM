"""FRED Code (c) 2012-2021 C.F. Sopuerta"""

import time
import logging
import argparse
import pandas as pd
import numpy as np
from scipy import special
from scipy.optimize import minimize_scalar
from class_SF_Physics import Physical_Quantities
from Some_functions import (
    run_basic_tests,
    Jump_Value,
    show_parameters,
    fred_goodbye,
    logging_func,
)
from Schwarzschild import zero_of_r_p_at_X, t_p_at_X, phi_p_at_X


# ---------------------------------------------------------------------


def init():
    """Read data, import julia, initialize parser and logger
    Returns SFdf dataframe and resfilename"""

    # --- Parser ---
    parser = argparse.ArgumentParser(description="FRED program")
    parser.add_argument("data", type=str, help="Input csv data file")
    parser.add_argument("-log_print", action="store_true", help="Print all log output")
    args = parser.parse_args()

    # --- Read Data from Parameter File into a Pandas DataFrame [SFdf] ---
    filename = args.data
    SFdf = pd.read_csv(filename)

    # Preparing Files for saving Self-Force Results
    resfilename = filename[:-4] + "_results.csv"
    with open(resfilename, "w") as resultsfile:
        resultsfile.write(
            f'# FRED RESULTS FILE [File opened on {time.strftime("%Y-%m-%d %H:%M")}]\n'
        )

    # logging
    logging_func(filename, args.log_print)

    return SFdf, resfilename


# ---------------------------------------------------------------------


def main(SFdf, resfilename):
    """Execute main program"""

    start_time = time.time()  # Initialize clock

    N_runs = SFdf.shape[0]

    # Starting the different Runs in the Parameter File 'data'
    for run in range(0, N_runs):
        main_run(SFdf, run, resfilename)

    # Final Time after Computations
    end_time = time.time()
    logging.info(f"Execution Time: {end_time - start_time} seconds")

    fred_goodbye()


# ---------------------------------------------------------------------


def main_run(SFdf, run, resfilename):
    # Set Clock to measure the Computational Time
    run_start_time = time.time()

    # Checking the Run Parameters
    run_basic_tests(SFdf, run)

    # Setting up Physical Quantities
    PP = Physical_Quantities(SFdf, run)
    logging.info(f"FRED RUN {run}: Class Physical Quantities Created")
    show_parameters(PP, run)  # Show parameters of the Run

    # Setting up the Grids and Grid-related Quantities
    # RM CG = Computational_Grid(SFdf, run)
    # RM print(f'FRED RUN {run}: Class Computational Grid Created')

    # Projecting the geodesic into the Particle Domains (from Horizon and to Infinity)
    project_geodesic(PP, run)

    # Computation of the Singular Part of the Self-Force:
    singular_part(PP, run)

    # NOTE: Big Loop starts Here!
    # Computing ell-Modes
    for ll in range(0, PP.ell_max + 1):  # Harmonic Number l
        for mm in range(0, ll + 1):  # Harmonic Number m

            # Check whether ell+m is odd (contribution is zero, so it can be skipped)
            if (ll + mm) % 2 == 1:
                continue

            # Computing Fourier Modes (n-Modes):
            n_estimated_error = 10.0 * PP.Mode_accuracy
            nf = 0  # Fourier Mode Number
            while nf <= PP.N_Fourier and n_estimated_error > PP.Mode_accuracy:
                do_mode(ll, mm, nf, PP, run)
                # Increasing the Fourier Mode Counter
                nf += 1
                n_estimated_error = np.amax(PP.Estimated_Error)

            # Aplying the missing factor to the Radial Component of the Bare (full) Self-Force at each
            # Particle Location (as determined by the SPACE index 'ns'):
            # This completes what we could call the m-Mode Computation
            PP.SF_F_r_lm_H[ll, mm] = (
                PP.SF_F_r_lm_H[ll, mm]
                * (PP.particle_charge / PP.r_p)
                * PP.d_lm[ll, mm]
                * np.exp(1j * mm * (PP.phi_p - PP.omega_phi * PP.t_p))
            )
            PP.SF_F_r_lm_I[ll, mm] = (
                PP.SF_F_r_lm_I[ll, mm]
                * (PP.particle_charge / PP.r_p)
                * PP.d_lm[ll, mm]
                * np.exp(1j * mm * (PP.phi_p - PP.omega_phi * PP.t_p))
            )

            # Computation of the Contribution of the m-Mode to the l-Mode of the Radial Component of the Bare (full) Self-Force
            # NOTE: We have to distinguish the m=0 more from the rest:
            if mm == 0:
                PP.SF_F_r_l_H[ll] = PP.SF_F_r_l_H[ll] + PP.SF_F_r_lm_H[ll, mm]
                PP.SF_F_r_l_I[ll] = PP.SF_F_r_l_I[ll] + PP.SF_F_r_lm_I[ll, mm]
            else:
                PP.SF_F_r_l_H[ll] = PP.SF_F_r_l_H[ll] + 2.0 * np.real(
                    PP.SF_F_r_lm_H[ll, mm]
                )
                PP.SF_F_r_l_I[ll] = PP.SF_F_r_l_I[ll] + 2.0 * np.real(
                    PP.SF_F_r_lm_I[ll, mm]
                )

        # After having finished the Computation of an l-Mode we need to substract the Contribution from the Singular field:
        # Computation of the l-Mode of the Radial Component of the Regular Self-Force:
        PP.SF_R_r_l_H[ll] = PP.SF_F_r_l_H[ll] - PP.SF_S_r_l_H[ll]
        PP.SF_R_r_l_I[ll] = PP.SF_F_r_l_I[ll] - PP.SF_S_r_l_I[ll]

        # Computation of the Contribution of the l-Mode of the Radial Component of the Regular Self-Force to the Total Regular Self-Force
        PP.SF_r_H = PP.SF_r_H + PP.SF_R_r_l_H[ll]
        PP.SF_r_I = PP.SF_r_I + PP.SF_R_r_l_I[ll]

    # Once the Computation of the Self-Force has ended we save/print the results:
    run_prints(PP, run, resfilename, run_start_time)


# ---------------------------------------------------------------------


def do_mode(ll, mm, nf, PP, run):
    """Computing the Bare Field Mode with Frequency omega_mn
    [REMEMBER: Psi is the scalar field and Phi its radial (tortoise) derivative]"""
    # TODO change name of function

    omega_mn = nf * (PP.omega_r) + mm * (PP.omega_phi)

    compute_mode(ll, omega_mn, PP)
    logging.info(f"FRED RUN {run}: Mode (l,m,n) = ({ll},{mm},{nf}) Computed")

    PP.R_H[ll, mm, nf + PP.N_Fourier, :] = PP.single_R_HOD
    PP.R_I[ll, mm, nf + PP.N_Fourier, :] = PP.single_R_IOD
    PP.Q_H[ll, mm, nf + PP.N_Fourier, :] = PP.single_Q_HOD
    PP.Q_I[ll, mm, nf + PP.N_Fourier, :] = PP.single_Q_IOD

    # Computing the Values of the Field Modes (Phi,Psi)lmn [~ (R,Q)lmn in Frequency Domain] at the Particle Location
    # at the different Time Collocation Points that satisfy the Boundary Conditions imposed by the Jump Conditions.
    # [FIRST WITH ARBITRARY BOUNDARY CONDITIONS, USING THE SOLUTION FOUND IN THE PREVIOUS POINT AND AFTERWARDS,
    # RESCALING WITH THE C_lmn COEFFICIENTS TO OBTAIN THE FIELD MODES (lmn) WITH THE CORRECT BOUNDARY CONDITIONS.
    # THIS INCLUDES THE COMPUTATION OF THE C_lmn COEFFICIENTS]:
    # NOTE: REMEMBER that we have projected the geodesics onto the Spatial Domain "containing" the Particle: [r_peri, r_apo]
    # NOTE: This why the index goes form 0 to N_space instead of from 0 to N_time

    # Computing the Value of the Jump:
    PP.J_lmn[ll, mm, nf + PP.N_Fourier] = Jump_Value(ll, mm, nf, PP)

    # Particle Location (rho/rstar and r Coordinates):
    rp = PP.r_p

    # Schwarzschild metric function 'f = 1 - 2M/r':
    fp = 1.0 - 2.0 / rp

    # Computing the Value of the Jump:
    PP.J_lmn[ll, mm, nf + PP.N_Fourier] = Jump_Value(ll, mm, nf, PP)

    indices = (ll, mm, nf + PP.N_Fourier)

    # Computing the C_lmn Coefficients [for the Harmonic mode (ll,mm), Fourier mode 'nt', and location <=> time 'ns']:
    Wronskian_RQ = PP.R_I[indices] * PP.Q_H[indices] - PP.R_H[indices] * PP.Q_I[indices]

    PP.Cm_lmn[indices] = -PP.R_I[indices] * PP.J_lmn[indices] / Wronskian_RQ
    PP.Cp_lmn[indices] = -PP.R_H[indices] * PP.J_lmn[indices] / Wronskian_RQ

    # Computing the Values of the Bare Field Modes (R,Q)(ll,mm,nn) at the Particle Location 'ns'
    # using the Correct Boundary Conditions: RESCALING WITH THE C_lmn COEFFICIENTS:
    PP.R_H[indices] = PP.Cm_lmn[indices] * PP.R_H[indices]
    PP.Q_H[indices] = PP.Cm_lmn[indices] * PP.Q_H[indices]

    PP.R_I[indices] = PP.Cp_lmn[indices] * PP.R_I[indices]
    PP.Q_I[indices] = PP.Cp_lmn[indices] * PP.Q_I[indices]

    # Computation of the contribution of the Fourier Mode 'nf' (l m nf) to the Radial Component of the Bare (full) Self-Force:
    # [NOTE: This is the contribution up to a multiplicative factor that is applied below]
    PP.SF_F_r_lm_H[ll, mm] = PP.SF_F_r_lm_H[ll, mm] + (
        PP.Q_H[indices] / fp - PP.R_H[indices] / rp
    ) * np.exp(-1j * nf * PP.omega_r * PP.t_p)
    PP.SF_F_r_lm_I[ll, mm] = PP.SF_F_r_lm_I[ll, mm] + (
        PP.Q_I[indices] / fp - PP.R_I[indices] / rp
    ) * np.exp(-1j * nf * PP.omega_r * PP.t_p)

    # print( f"l={ll} m={mm} n={nf}: c_H[{nf}]={PP.SF_F_r_lm_H[ll, mm]:.14f}  c_I[{nf}]={PP.SF_F_r_lm_I[ll, mm]:.14f}")

    # Store Contribution and Estimate Error
    # TODO accumulated might not be necessary
    PP.Accumulated_SF_F_r_lm_H[ll, mm] = PP.SF_F_r_lm_H[ll, mm]
    PP.Accumulated_SF_F_r_lm_I[ll, mm] = PP.SF_F_r_lm_I[ll, mm]

    PP.Estimated_Error = np.maximum(
        np.absolute(PP.Accumulated_SF_F_r_lm_H[ll, mm]),
        np.absolute(PP.Accumulated_SF_F_r_lm_I[ll, mm]),
    )

    # For nf != 0 there are both positive and negative frequencies
    if nf > 0:
        do_mode(ll, mm, -nf, PP, run)
    return


# ---------------------------------------------------------------------


def project_geodesic(PP, run):
    """Projecting the geodesic into the Particle Domains (from Horizon and to Infinity)"""
    PP.t_p[0] = PP.t_p_f[0]
    PP.phi_p[0] = PP.phi_p_f[0]
    # PP.r_p[0] = PP.r_p_f[0]
    # PP.rs_p[0] = PP.rs_p_f[0]
    # PP.chi_p[0] = PP.chi_p_f[0]

    for ii in range(1, PP.N_OD):
        ntime = 0
        r_goal = PP.r_p[ii]

        error0 = PP.r_apo - PP.r_peri
        for nt in range(0, PP.N_time + 1):
            error = np.abs(r_goal - PP.r_p_f[nt])

            if error < error0:
                error0 = error
                ntime = nt

        X_guess = PP.Xt[ntime]  # TODO check if needed

        res = minimize_scalar(
            zero_of_r_p_at_X,
            bounds=(-1, 1),
            args=(r_goal, PP),
            method="bounded",
            options={"xatol": 1e-15, "maxiter": 50000000000},
        )
        Xv = res.x

        PP.t_p[ii] = t_p_at_X(Xv, PP)
        PP.phi_p[ii] = phi_p_at_X(Xv, PP)

    PP.t_p[PP.N_OD] = PP.t_p_f[PP.N_time]
    PP.phi_p[PP.N_OD] = PP.phi_p_f[PP.N_time]

    PP.rs_p = PP.r_p - 2.0 * np.log(0.5 * (PP.r_p) - 1.0)

    logging.info(
        f"FRED RUN {run}: Projection of the geodesic onto the Radial Spatial Domain Done"
    )


# ---------------------------------------------------------------------


def singular_part(PP, run):
    """Computation of the Singular Part of the Self-Force"""
    for ns in range(0, PP.N_OD + 1):

        # Radial Location of the Particle:
        chi_p_now = PP.chi_p[ns]
        r_p_now = PP.r_p[ns]
        phi_p_now = PP.phi_p[ns]  # TODO check if needed

        # Schwarzschild Metric Function f at tbe Particle Location:
        f_p = 1.0 - 2.0 / r_p_now

        # Computing rdot:
        q1 = PP.p_orbit / r_p_now
        q3 = PP.p_orbit - 2.0 * q1
        q4 = np.sqrt((PP.p_orbit - 2.0) ** 2 - 4.0 * (PP.e_orbit ** 2))
        q5 = q3 * (q1 ** 2) / (q4 * PP.p_orbit)

        chi_dot = q5 * (np.sqrt(PP.p_orbit - 4.0 - 2.0 * q1)) / (PP.p_orbit)
        dr_p_now_dtau = (
            (PP.Ep)
            * (r_p_now ** 2)
            * (PP.e_orbit)
            * (np.sin(chi_p_now))
            * chi_dot
            / (f_p * (PP.p_orbit))
        )

        # Computing other relevant quantities:
        qaux1 = 1.0 + ((PP.Lp) ** 2) / (r_p_now ** 2)
        qaux2 = (qaux1) ** (1.5)
        alpha = ((PP.Lp) ** 2) / (r_p_now ** 2 + (PP.Lp) ** 2)

        # Computing the Value of some Hypergeometric Functions:
        ff_1d2 = special.hyp2f1(0.5, 0.5, 1.0, alpha)
        ff_m1d2 = special.hyp2f1(-0.5, 0.5, 1.0, alpha)
        ff_3d2 = special.hyp2f1(1.5, 0.5, 1.0, alpha)
        ff_5d2 = special.hyp2f1(2.5, 0.5, 1.0, alpha)

        # Computing Regularization Parameters:
        for ll in range(0, PP.ell_max + 1):

            delta_r = -1.0
            PP.A_t_H[ll][ns] = delta_r * dr_p_now_dtau / qaux1
            PP.A_r_H[ll][ns] = -delta_r * (PP.Ep) / (f_p * qaux1)

            delta_r = 1.0
            PP.A_t_I[ll][ns] = delta_r * dr_p_now_dtau / qaux1
            PP.A_r_I[ll][ns] = -delta_r * (PP.Ep) / (f_p * qaux1)
            PP.B_t[ll][ns] = (PP.Ep) * dr_p_now_dtau * (0.5 * ff_1d2 - ff_m1d2) / qaux2

            PP.B_r[ll][ns] = (
                (dr_p_now_dtau ** 2 - 2.0 * (PP.Ep) ** 2) * ff_1d2
                + (dr_p_now_dtau ** 2 + (PP.Ep) ** 2) * ff_m1d2
            ) / (2.0 * f_p * qaux2)

            PP.D_r[ll][ns] = 0.0

            # Computing Singular Part of the Self-Force (only the radial component at the moment):
            PP.SF_S_r_l_H[ll][ns] = (
                (PP.particle_charge)
                * ((ll + 0.5) * (PP.A_r_H[ll][ns]) + PP.B_r[ll][ns])
                / (r_p_now ** 2)
            )
            PP.SF_S_r_l_I[ll][ns] = (
                (PP.particle_charge)
                * ((ll + 0.5) * (PP.A_r_I[ll][ns]) + PP.B_r[ll][ns])
                / (r_p_now ** 2)
            )

    logging.info(f"FRED RUN {run}: Singular Part of the Self-Force Computed")


# ---------------------------------------------------------------------


def run_prints(PP, run, resfilename, run_start_time):
    """Printing the different field components with zero Fourier Mode (nf = 0)
    P for ll in range(0, PP.ell_max+1):                         # Harmonic Number l
    P for mm in range(0, ll+1):                             # Harmonic Number m
    P ns = 3
    P print('l=',ll,' m=',mm,' R- =', PP.R_H[ll, mm, PP.N_Fourier, ns],
    ' Q- =', PP.Q_H[ll, mm, PP.N_Fourier, ns],' R+ =', PP.R_I[ll, mm, PP.N_Fourier, ns],' Q+ =', PP.Q_I[ll, mm, PP.N_Fourier, ns])"""
    # fmt: off
    # Taking end time of the Run and computing total time for the Run:
    run_end_time = time.time()
    run_total_time = run_end_time - run_start_time

    # Printing the l-components of the Bare Self-Force
    ns = 3
    print("\n\nResults for the l-Components of the Bare Self-Force at time t = ", PP.t_p[ns])
    for ll in range(0, PP.ell_max + 1):
        print("l=", ll, " FF- =", np.real(PP.SF_F_r_l_H[ll, ns]), " FF+ =", np.real(PP.SF_F_r_l_I[ll, ns]),
                " FS- =", np.real(PP.SF_S_r_l_H[ll, ns]), " FS+ =", np.real(PP.SF_S_r_l_I[ll, ns]),
                " FR- =", np.real(PP.SF_R_r_l_H[ll, ns]), " FR+ =", np.real(PP.SF_R_r_l_I[ll, ns]),
        )

    # # Printing in the Computer Screen the Radial Component of the Regular Self-Force at each Time:
    # for ns in range(0, PP.N_space+1):
    #     print(PP.t_p_s[ns], PP.SF_r_H[ns], PP.SF_r_I[ns])

    # Total number of (l m)-Modes
    Number_modes = (PP.ell_max + 1) * (PP.ell_max + 2) / 2

    # Saving the results of the Computation into a File:
    with open(resfilename, "a") as resultsfile:
        resultsfile.write(f"# RUN NUMBER:  {run}\n")
        resultsfile.write("# RUN INFO:\n")
        resultsfile.write(f"#   Maximum Harmonic Number:                       ell = {PP.ell_max}\n")
        resultsfile.write(f"#   Number of Modes to be computed:            N_modes = {Number_modes}\n")
        resultsfile.write(f"#   Number of Collocation Points per Domain:         N = {2*PP.N_OD+1}\n")
        resultsfile.write(f"#   Orbital Eccentricity:                            e = {PP.e_orbit}\n")
        resultsfile.write(f"#   Orbital Semilatus Rectum:                        p = {PP.p_orbit}\n")
        resultsfile.write(f"#   Total Run Time:                                 Tc = {run_total_time}\n#\n")
        resultsfile.write("#   r_p            F_r-            F_r+\n")
        for ns in range(0, PP.N_OD):
            resultsfile.write(f"{PP.r_p[ns]:.6f},{np.real(PP.SF_r_H[ns]):.14f},{np.real(PP.SF_r_I[ns]):.14f}\n")

    # fmt: on


# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Init is executed before importing julia for fast argparse help

    start_time = time.time()  # Initialize clock
    SFdf, resfilename = init()

    # --- julia imports ---
    """
    from julia.api import LibJulia
    api = LibJulia.load()
    api.sysimage = "sysimage2/ODEs.so"
    api.init_julia()
    """
    import julia

    jl = julia.Julia(compiled_modules=False, depwarn=True, sysimage="sysimage/ODEs.so")
    # from julia.api import Julia
    # Julia(debug=True)
    from julia import Main

    Main.include("src/mode_comp.jl")
    compute_mode = Main.eval("compute_mode")  # global function
    logging.info("julia imports done")
    # ------------------------------------------------------
    setup_time = time.time()
    logging.info(f"Setup Time: {setup_time - start_time} seconds")

    main(SFdf, resfilename)
