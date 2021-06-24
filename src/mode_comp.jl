using StaticArrays
using OrdinaryDiffEq
using PyCall
using Plots.PlotMeasures
using Plots; pyplot(size=(1500, 1000), bottom_margin=5mm, linewidth=1.5, legend = :outertopleft)
Plots.scalefontsizes(2)
using LaTeXStrings

# Get absolute path
FOLDER = normpath(joinpath(@__FILE__,"..",".."))

machinery = pyimport("importlib.machinery")
loader = machinery.SourceFileLoader("Schwarzschild", FOLDER * "/src/Schwarzschild.py")
schw = loader.load_module("Schwarzschild")
include("./compactification.jl")

# Global variables
const ABSTOL = 1e-12
const RTOL = 1e-8
const METHOD = KenCarp5()
# -----------------------------------------------

function regge_wheeler_potential(rstar, ll)
    cb = ll * (ll + 1)
    xstar = 0.5 * rstar - 1
    xsch = schw.r_tortoise_to_x_schwarzschild(rstar)
    rsch = 2 * (1 + xsch)
    # exp(xstar-x) = x

    Vl =  2 * exp(xstar - xsch) * (cb + 2 / rsch) / (rsch^3)

    return Vl
end

# -----------------------------------------------

"""ODE system that determines the solution to the Master Equation in the Horizon Domain"""
function RHS_HD(u, p, rho)
    ll, w_mn, PP, backward = p
    if backward
        rho = -rho
    end
    re_R, im_R, re_Q, im_Q = u
    R = re_R + 1im * im_R
    Q = re_Q + 1im * im_Q
    dR, dQ = 0, 0

    # Integration through the Horizon Domain (HD)
    if PP.rho_H <= rho <= PP.rho_HC
        epsilon_rho = rho + 20

        # Integration near the Horizon
        # Using the initial condition (R = A, Q = -i w_mn R) and (dR = Q)
        # => R = A exp(-i w_mn epsilon), dR = -i w_mn A exp(-i w_mn epsilon)
        if epsilon_rho < 1e-6

            # dR = -1im * w_mn * R * exp(-1im * w_mn * epsilon_rho)
            # dQ = -(w_mn^2) * R * exp(-1im * w_mn * epsilon_rho)
            dR = -1im * w_mn * R
            dQ = -1im * w_mn * Q

            # Integration not close to the Horizon
        else
            Omega = 1 - rho / PP.rho_H
            rstar = rho / Omega
            xsch = schw.r_tortoise_to_x_schwarzschild(rstar)
            rsch = 2 * (1 + xsch)
            rsch2 = rsch * rsch
            rsch3 = rsch2 * rsch

            exp_rstar_over_2M = exp(0.5 * rstar)
            regular_potential_factor = ( exp(-(1 + xsch)) * (2 / rsch3)
                                        * (ll * (ll + 1) + 2 * (PP.sigma_spin) / rsch)
                                       )

            H = 1 - Omega^2
            H_plus_one = 1 + H
            DH = (2 / PP.rho_H) * Omega
            DH_over_1minusH = (2 / PP.rho_H) / Omega
            Q_over_omega2 = Q / Omega^2
            R_over_omega2 = R / Omega^2

            dR = Q
            dQ = DH_over_1minusH * (Q + 1im * w_mn * R) +
                  + 2im * w_mn * H * Q_over_omega2 +
                  - ( H_plus_one * w_mn^2 - regular_potential_factor * (exp_rstar_over_2M / Omega^2)) * R_over_omega2
        end

    # Integration through the Horizon Transition Region
    elseif PP.rho_HC < rho < PP.rho_HS
        width = PP.rho_HC - PP.rho_HS
        sigma = 0.5 * (pi) * (rho - PP.rho_HS) / width
        jacobian = 0.5 * pi / width

        f0 = f_transition(sigma, PP.TF)
        f1 = jacobian * f_transition_1st(f0, sigma, PP.TF)
        f2 = (jacobian^2) * f_transition_2nd(f0, sigma, PP.TF)

        Omega = 1 - f0 * rho / PP.rho_H
        dOmega_drho = -(f0 + rho * f1) / PP.rho_H
        d2Omega_drho2 = -(2 * f1 + rho * f2) / PP.rho_H

        LH = Omega - rho * dOmega_drho
        H = 1 - (Omega^2) / LH
        one_plus_H = 1 + H
        one_minus_H = 1 - H
        DH = -(Omega / LH) * (2 * dOmega_drho + rho * (Omega / LH) * d2Omega_drho2)

        rstar = rho / Omega
        Vl = regge_wheeler_potential(rstar, ll)

        dR = Q
        dQ = (
              (DH / one_minus_H) * (Q + 1im * w_mn * R)
              + 2im * w_mn * (H / one_minus_H) * Q +
              - ( (one_plus_H / one_minus_H) * (w_mn^2)
                 - (Vl / one_minus_H / one_minus_H))
              * R
             )

    # Integration through the Regular Region  (i.e. rho = rstar)
    elseif rho >= PP.rho_HS
        Vl = regge_wheeler_potential(rho, ll)

        dR = Q
        dQ = (Vl - w_mn^2) * R

    # Outside the Physical Integration Region
    else
        print("rho= ",rho,"  Out of Domain error during ODE Integration at the Horizon Domain")
    end

    if backward
        dR, dQ = -dR, -dQ
    end

    @SVector [real(dR), imag(dR), real(dQ), imag(dQ)]
end

# -----------------------------------------------

"""ODE system that determines the solution to the Master Equation in the Horizon Orbital Domain"""
function RHS_HOD(u, p, rho)
    ll, w_mn, _, backward = p
    if backward
        rho = -rho
    end
    re_R, im_R, re_Q, im_Q = u
    R = re_R + 1im * im_R
    Q = re_Q + 1im * im_Q

    Vl = regge_wheeler_potential(rho, ll)

    dR = Q
    dQ = (Vl - w_mn^2) * R

    if backward
        dR, dQ = -dR, -dQ
    end

    @SVector [real(dR), imag(dR), real(dQ), imag(dQ)]
end

# -----------------------------------------------

"""ODE system that determines the solution to the Master Equation in the Infinity Orbital Domain
Defined backward, e.i. rho -> -rho, so can be integrated from right to left"""
function RHS_IOD(u, p, rho)
    ll, w_mn, _, backward = p
    if backward
        rho = -rho
    end
    re_R, im_R, re_Q, im_Q = u
    R = re_R + 1im * im_R
    Q = re_Q + 1im * im_Q

    Vl = regge_wheeler_potential(rho, ll)

    dR = Q
    dQ = (Vl - w_mn^2) * R

    if backward
        dR, dQ = -dR, -dQ
    end

    @SVector [real(dR), imag(dR), real(dQ), imag(dQ)]
end

# -----------------------------------------------

"""ODE system that determines the solution to the Master Equation in the Infinity Domain
Defined backward, e.i. rho -> -rho, so can be integrated from right to left"""
function RHS_ID(u, p, rho)
    ll, w_mn, PP, backward = p
    if backward
        rho = -rho
    end
    re_R, im_R, re_Q, im_Q = u
    re_RI, im_RI, re_QI, im_QI = u
    R = re_R + 1im * im_R
    Q = re_Q + 1im * im_Q
    dR, dQ = 0, 0

    cb = ll * (ll + 1)

    # Integration through the Infinity Domain (ID)
    if PP.rho_IC <= rho <= PP.rho_I
        epsilon_rho = PP.rho_I - rho

        # Integration near (null) Infinity
        if epsilon_rho < 1e-6

            # Integration for the Particular Case of Zero-Frequency Modes
            if abs(w_mn) < 1e-8

                rho_I_1 = 1/PP.rho_I
                ell_1 = ll + 1
                ell_2 = ll + 2
                ell_32 = ll + 3/2

                sigma0 = 1

                sigma1 = sigma0 * (cb - rho_I_1*(cb-1)) * rho_I_1 / ell_1

                sigma2 = sigma0 / (4 * ell_1 * ell_32 * PP.rho_I^2) * (
                           2 * (ll^4 + 2*ll^3 - ll^2 - 4*ll - 1) * rho_I_1^2 +
                         - 4 * ell_1*ell_32 * (ll*ell_1-1) * rho_I_1 +
                         + 2*ll*ell_1^2*ell_32
                        )

                sigma3 = sigma0 / (12 * ell_2 * ell_1 * ell_32 * PP.rho_I^3) * (
                            - 2 * (ll^2+ll-1)*(ll^4+2*ll^3-ll^2-8*ll-7) * rho_I_1^3 +
                            + 6 * ell_2^2 * (ll^4+2*ll^3-ll^2-4*ll-1) * rho_I_1^2 +
                            - 6 * ell_2^2 * ell_1 * ell_32 * (ll^2+ll-1) * rho_I_1 +
                            + 2 * ell_2^2 * ell_1^2 * ll * ell_32
                        )

                Sigma = sigma0 + sigma1 * epsilon_rho + sigma2 * epsilon_rho^2 + sigma3 * epsilon_rho^3
                dSigmadepsilon = sigma1 + 2 * sigma2 * epsilon_rho + 3 * sigma3 * epsilon_rho^2
                d2Sigmadepsilon2 = 2 * sigma2 + 6 * sigma3 * epsilon_rho

                # Particular Case: ll = 0
                if ll == 0
                    dR = -dSigmadepsilon
                    dQ = d2Sigmadepsilon2

                # Particular Case: ll = 1
                elseif ll == 1

                    dR = -Sigma - epsilon_rho * dSigmadepsilon
                    dQ = 2 * dSigmadepsilon + epsilon_rho * d2Sigmadepsilon2

                # All other ll different from 0 and 1
                else
                    dR = -epsilon_rho^(ll-1) * ( ll*Sigma + epsilon_rho*dSigmadepsilon )
                    dQ = epsilon_rho^(ll-2) * (
                            ll*(ll-1)*Sigma +
                            + 2*ll*epsilon_rho * dSigmadepsilon +
                            + epsilon_rho^2 * d2Sigmadepsilon2
                            )
                end

            # Integration for non-zero Frequency Modes
            else
                qireal = 0
                qiimag = w_mn*re_RI*(1 - cb / (2*PP.rho_I^2 * w_mn^2))

                r1real = qireal
                r1imag = qiimag

                q1real = -w_mn^2*re_RI*( 1 - cb*( 1 - (ll^2+ll+2) / (4*PP.rho_I^2 * w_mn^2) ) / (PP.rho_I^2 * w_mn^2) )
                q1imag = -w_mn^2*re_RI*( 1 - cb*(1-PP.rho_I) ) / (PP.rho_I^4 * w_mn^3)

                r2real = -0.5*q1real
                r2imag = -0.5*q1imag

                q2real = (re_RI / (4*PP.rho_I^4))*( cb*(PP.rho_I - 1) + 1 )*( 2 - (cb + 6) / (PP.rho_I^2 * w_mn^2) )
                q2imag = (w_mn^3*re_RI/2) * (
                          + 1 - (3*cb-4) / (2*PP.rho_I^2 * w_mn^2) +
                          + cb*(3*cb-22) / (4*PP.rho_I^4 * w_mn^4) +
                          + 4*(1 + 1.5*(cb-1)*(PP.rho_I)) / (PP.rho_I^6 * w_mn^4) +
                          - cb*(ll+3)*(ll-2)*(cb+2) / (8*PP.rho_I^6 * w_mn^6)
                         )

                r3real = -q2real/3
                r3imag = -q2imag/3

                q3real = (w_mn^4*re_RI/6) * (
                          + 1 - 2*(cb-1) / (PP.rho_I^2 * w_mn^2) +
                          + (cb*(3*cb+2)+24) / (2*PP.rho_I^4 * w_mn^4) +
                          + 12*(cb-1)*(3 / (PP.rho_I^2 * w_mn^2) - 1) / (PP.rho_I^5 * w_mn^4) +
                          - 8 / (PP.rho_I^6 * w_mn^4) +
                          - cb*(ll^4 + 2*ll^3 - 7*ll^2 - 8*ll + 72) / (2*PP.rho_I^6 * w_mn^6) +
                          + ( cb*(3*cb-2) + 27 ) / (PP.rho_I^8 * w_mn^6) +
                          + cb*(ll+3)*(ll+4)*(ll-2)*(ll-3)*(cb+2) / (16*PP.rho_I^8 * w_mn^8)
                         )

                q3imag = (re_RI/6) * (
                          - 4*cb*w_mn / PP.rho_I^3 +
                          + 4*w_mn*(cb-1) / (PP.rho_I^4) +
                          + 4*cb*(ll+3)*(ll-2) / ((PP.rho_I^5)*w_mn) +
                          - 4*(ll+4)*(ll-3)*(cb-1) / ((PP.rho_I^6)*w_mn) +
                          + 48 / (PP.rho_I^7*w_mn) +
                          - cb*(cb^2-18) / (PP.rho_I^7*w_mn^3) +
                          + (cb-1)*(cb^2-18) / (PP.rho_I^8*w_mn^3)
                         )

                r4real = -0.25*q3real
                r4imag = -0.25*q3imag

                dR_real = -r1real - 2*r2real*epsilon_rho - 3*r3real*epsilon_rho^2 - 4*r4real*epsilon_rho^3
                dR_imag = -r1imag - 2*r2imag*epsilon_rho - 3*r3imag*epsilon_rho^2 - 4*r4imag*epsilon_rho^3
                dQ_real = -q1real - 2*q2real*epsilon_rho - 3*q3real*epsilon_rho^2
                dQ_imag = -q1imag - 2*q2imag*epsilon_rho - 3*q3imag*epsilon_rho^2
                dR = dR_real + 1im * dR_imag
                dQ = dQ_real + 1im * dQ_imag
            end

        # Integration not "close" to (null) Infinity
        else
            Omega = 1 - rho / PP.rho_I
            rstar = rho / Omega
            xsch = schw.r_tortoise_to_x_schwarzschild(rstar)
            rsch = 2 * (1 + xsch)
            rsch2 = rsch * rsch
            rsch3 = rsch2 * rsch

            f = 1 - 2 / rsch
            regular_potential_factor = f * (cb + 2 * PP.sigma_spin / rsch)
            romega2 = (rho - 2 * Omega * log(xsch))^2

            H = 1 - Omega^2
            H_plus_one = 1 + H
            DH = (2 / PP.rho_I) * Omega
            DH_over_1minusH = (2 / PP.rho_H) / Omega

            Q_over_omega2 = Q / Omega^2
            R_over_omega2 = R / Omega^2

            dR = Q
            dQ = DH_over_1minusH * (Q + 1im* w_mn * R) + 2im * w_mn * H * Q_over_omega2 +
                - H_plus_one * (w_mn ^ 2) * R_over_omega2 + regular_potential_factor * R_over_omega2 / romega2

        end

    elseif PP.rho_IS < rho < PP.rho_IC
        width = PP.rho_IC - PP.rho_IS
        sigma = 0.5 * pi * (rho - PP.rho_IS) / width
        jacobian = 0.5 * pi / width

        f0 = f_transition(sigma, PP.TF)
        f1 = jacobian * f_transition_1st(f0, sigma, PP.TF)
        f2 = (jacobian ^ 2) * f_transition_2nd(f0, sigma, PP.TF)

        Omega = 1 - f0 * rho / PP.rho_I
        dOmega_drho = -(f0 + rho * f1) / PP.rho_I
        d2Omega_drho2 = -(2 * f1 + rho * f2) / PP.rho_I

        LI = Omega - rho * dOmega_drho
        H = 1 - Omega^2 / LI
        one_plus_H = 1 + H
        one_minus_H = 1 - H
        DH = -(Omega / LI) * (2 * dOmega_drho + rho * (Omega / LI) * d2Omega_drho2)

        rstar = rho / Omega
        xsch = schw.r_tortoise_to_x_schwarzschild(rstar)
        rsch = 2 * (1 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch
        Vl = xsch * (2 / rsch3) * (cb + 2 * PP.sigma_spin / rsch)

        dR = Q
        dQ = (DH / one_minus_H) * (Q + 1im * w_mn * R) + 2im * w_mn * (H / one_minus_H) * Q +
            -(one_plus_H / one_minus_H) * (w_mn ^ 2) * R + (Vl / one_minus_H^2) * R

    elseif rho <= PP.rho_IS
        xsch = schw.r_tortoise_to_x_schwarzschild(rho)  # rstar = rho

        rsch = 2 * (1 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch

        Vl = xsch * (2 / rsch3) * (cb + 2 * PP.sigma_spin / rsch)

        dR = Q
        dQ = (Vl - w_mn^2) * R

    else
        print( "rho= ", rho, "  Out of Domain error during ODE Integration at the Horizon Domain")
    end

    if backward
        dR, dQ = -dR, -dQ
    end

    @SVector [real(dR), imag(dR), real(dQ), imag(dQ)]
end

# -----------------------------------------------


function solve_hd(ll, mm, nf, p, PP, save; do_plot=false)
    u0 = @SVector [1, 0, 0, -p[2]]
    rhospan = (PP.rho_H_plus, PP.rho_peri)
    prob = ODEProblem(RHS_HD, u0, rhospan, p)
    # save at whole array or only last point
    save ? saveat=PP.rho_HD : saveat=PP.rho_peri
    sol = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=saveat)

    if do_plot != false
        sol_matrix = hcat(sol.u...)'
        x_y = (sol.t, sol_matrix[:,1])
        args = (:xlab=>L"\rho", :ylab=>L"\hat{R}^-_{%$ll,%$mm,%$nf}", :label=>"")
        do_plot == 1 ? plot(x_y; args...) : plot!(x_y)
    end
    return sol
end

function solve_hod(ll, mm, nf, p, PP, save, sol_hd; do_plot=false)
    u0 = last(sol_hd.u)
    u0 = @SVector [u0[1], u0[2], u0[3], u0[4]]
    rhospan = (PP.rho_peri, PP.rho_apo)
    prob = ODEProblem(RHS_HOD, u0, rhospan, p)
    sol = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=PP.rho_HOD)
    lambda_minus = last(sol.u)[1]

    sol_matrix = hcat(sol.u...)'
    sol_matrix = sol_matrix/lambda_minus  # λ⁻ = 1

    PP.single_R_HOD = sol_matrix[:,1] + 1im * sol_matrix[:,2]
    PP.single_Q_HOD = sol_matrix[:,3] + 1im * sol_matrix[:,4]

    # save HD region
    if save
        sol_hd_matrix = hcat(sol_hd.u...)'
        sol_hd_matrix = sol_hd_matrix/lambda_minus  # λ⁻ = 1
        PP.single_R_HD = sol_hd_matrix[:,1] + 1im * sol_hd_matrix[:,2]
        PP.single_Q_HD = sol_hd_matrix[:,3] + 1im * sol_hd_matrix[:,4]
    end
    # Plot without scaling for λ⁻ = 1
    sol_matrix = hcat(sol.u...)'
    plot!(sol.t, sol_matrix[:,1], label="")
    savefig("Rm"*string(ll)*"_"*string(mm)*"_"*string(nf))
end

function solve_id(ll, mm, nf, p, PP, save; do_plot=false)
    u0 = @SVector [1, 0, 0, p[2]]

    # ε_I is bigger for modes l,0,0
    if 0 == mm == nf
        rhospan = (-PP.rho_I + 1, -PP.rho_apo)
    else
        rhospan = (-PP.rho_I_minus, -PP.rho_apo)
    end

    prob = ODEProblem(RHS_ID, u0, rhospan, p)
    # save at whole array or only last point
    save ? saveat=PP.rho_ID : saveat=PP.rho_apo
    # reverse rho -> -rho, by changing saveat sign
    sol = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=-saveat)

    sol_matrix = hcat(sol.u...)'

    if do_plot != false
        args = (:xlab=>L"\rho", :ylab=>L"\hat{R}^+_{%$ll,%$mm,%$nf}", :label=>"")
        plot(-sol.t, sol_matrix[:,1]; args...)
    end

    if 0 == mm == nf && save
        temp = zeros(length(sol.u) + 1, 4)
        temp[1,:] = sol_matrix[1,:]
        temp[2:end,:] = sol_matrix
        sol_matrix = temp
    end
    return sol_matrix
end

function solve_iod(ll, mm, nf, p, PP, save, sol_id_matrix; do_plot=false)
    u0 = sol_id_matrix[end,:]
    u0 = @SVector [u0[1], u0[2], u0[3], u0[4]]
    rhospan = (-PP.rho_apo, -PP.rho_peri)
    prob = ODEProblem(RHS_IOD, u0, rhospan, p)
    # reverse rho -> -rho, by changing saveat sign
    sol = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=-PP.rho_IOD)
    lambda_plus = last(sol.u)[1]

    sol_matrix = hcat(sol.u...)'
    sol_matrix = sol_matrix/lambda_plus  # λ⁺ = 1
    PP.single_R_IOD = sol_matrix[:,1] + 1im * sol_matrix[:,2]
    PP.single_Q_IOD = sol_matrix[:,3] + 1im * sol_matrix[:,4]

    # Save ID region
    if save
        sol_id_matrix = sol_id_matrix/lambda_plus  # λ⁺ = 1
        PP.single_R_ID = sol_id_matrix[:,1] + 1im * sol_id_matrix[:,2]
        PP.single_Q_ID = sol_id_matrix[:,3] + 1im * sol_id_matrix[:,4]
    end
    sol_matrix = hcat(sol.u...)'
    plot!(-sol.t, sol_matrix[:,1], label="")
    savefig("Rp"*string(ll)*"_"*string(mm)*"_"*string(nf))
end

"""Solve both regions, starting from the horizon and infinity
   uses static arrays to improve perfomance
   save :: keep HD and ID modes solutions"""
function compute_mode(ll, mm, nf, PP; method=METHOD, save=false)
    w_mn = nf * PP.omega_r + mm * PP.omega_phi

    # ----- HD -----
    p = @SVector [ll, w_mn, PP, false]
    sol_hd = solve_hd(ll, mm, nf, p, PP, save, do_plot=1)
    solve_hod(ll, mm, nf, p, PP, save, sol_hd, do_plot=1)

    # ----- ID (backwards) -----
    p = @SVector [ll, w_mn, PP, true]
    sol_id_matrix = solve_id(ll, mm, nf, p, PP, save, do_plot=1)
    solve_iod(ll, mm, nf, p, PP, save, sol_id_matrix, do_plot=1)
end



# Methods:
# TRBDF2() works except for n=0
# AutoVern7(Rodas5() fails at ID
# Rodas5() fails from start

"""Solve both regions, starting from the horizon and infinity
   uses static arrays to improve perfomance
   save :: keep HD and ID modes solutions"""
function compute_check(ll, mm, nf, PP; method=METHOD, save=false)
    w_mn = nf * PP.omega_r + mm * PP.omega_phi
    p = @SVector [ll, w_mn, PP, false]

    # ----- HD -----
    u0 = @SVector [1, 0, 0, -w_mn]
    rhospan = (PP.rho_H_plus, PP.rho_peri)
    prob = ODEProblem(RHS_HD, u0, rhospan, p)
    # save at whole array or only last point
    save ? saveat=PP.rho_HD : saveat=PP.rho_peri
    sol = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=saveat)

    # ----- HD check -----
    p = @SVector [ll, w_mn, PP, true]
    u0 = last(sol.u)
    u0 = @SVector [u0[1], u0[2], u0[3], u0[4]]
    rhospan = (-PP.rho_peri, -PP.rho_H_plus)
    prob = ODEProblem(RHS_HD, u0, rhospan, p)
    # save at whole array or only last point
    save ? saveat=PP.rho_HD : saveat=PP.rho_H_plus
    check = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=-saveat)

    uf = last(check.u)
    error =  [abs(uf[1]-1), abs(uf[2]), abs(uf[3]), abs(uf[4]+w_mn)]
    println(error)

    sol_matrix = hcat(sol.u...)'
    check_matrix = hcat(check.u...)'
    plot(sol.t, sol_matrix[:,1], xlab=L"\rho", ylab=L"\hat{R}^-_{%$ll,%$mm,%$nf}",
        label="forward", color=1)
    plot!(-check.t, check_matrix[:,1], label="backward", color=2)

    # ----- HOD -----
    p = @SVector [ll, w_mn, PP, false]
    u0 = last(sol.u)
    u0 = @SVector [u0[1], u0[2], u0[3], u0[4]]
    rhospan = (PP.rho_peri, PP.rho_apo)
    prob = ODEProblem(RHS_HOD, u0, rhospan, p)
    sol2 = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=PP.rho_HOD)
    lambda_minus = last(sol2.u)[1]

    u_matrix = hcat(sol2.u...)'
    u_matrix = u_matrix/lambda_minus  # λ⁻ = 1

    PP.single_R_HOD = u_matrix[:,1] + 1im * u_matrix[:,2]
    PP.single_Q_HOD = u_matrix[:,3] + 1im * u_matrix[:,4]

    # save HD region
    if save
        u_matrix = hcat(sol.u...)'
        u_matrix = u_matrix/lambda_minus  # λ⁻ = 1
        PP.single_R_HD = u_matrix[:,1] + 1im * u_matrix[:,2]
        PP.single_Q_HD = u_matrix[:,3] + 1im * u_matrix[:,4]
    end

    # ----- HOD check -----
    p = @SVector [ll, w_mn, PP, true]
    u0 = last(sol2.u)
    u0 = @SVector [u0[1], u0[2], u0[3], u0[4]]
    rhospan = (-PP.rho_apo, -PP.rho_peri)
    prob = ODEProblem(RHS_HOD, u0, rhospan, p)
    check = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=-PP.rho_IOD)

    uf = last(check.u)
    u0 = last(sol.u)
    error =  [abs(uf[1]-u0[1]), abs(uf[2]-u0[2]), abs(uf[3]-u0[3]), abs(uf[4]-u0[4])]
    println(error)

    sol_matrix = hcat(sol2.u...)'
    check_matrix = hcat(check.u...)'
    plot!(sol2.t, sol_matrix[:,1], label="", color=1)
    plot!(-check.t, check_matrix[:,1], label="", color=2)
    savefig("Rm"*string(ll)*"_"*string(mm)*"_"*string(nf))

    # ----- ID (backwards) -----
    p = @SVector [ll, w_mn, PP, true]
    u0 = @SVector [1, 0, 0, w_mn]
    rhospan = (-PP.rho_I_minus, -PP.rho_apo)
    prob = ODEProblem(RHS_ID, u0, rhospan, p)
    # save at whole array or only last point
    save ? saveat=PP.rho_ID : saveat=PP.rho_apo
    # reverse rho -> -rho, by changing saveat sign
    sol = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=-saveat)

    # ----- ID check -----
    p = @SVector [ll, w_mn, PP, false]
    u0 = last(sol.u)
    u0 = @SVector [u0[1], u0[2], u0[3], u0[4]]
    rhospan = (PP.rho_apo, PP.rho_I_minus)
    prob = ODEProblem(RHS_ID, u0, rhospan, p)
    save ? saveat=PP.rho_ID : saveat=PP.rho_I_minus
    check = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=saveat)
    uf = last(check.u)
    error =  [abs(uf[1]-1), abs(uf[2]), abs(uf[3]), abs(uf[4]-w_mn)]
    println(error)

    sol_matrix = hcat(sol.u...)'
    check_matrix = hcat(check.u...)'
    plot(-sol.t, sol_matrix[:,1], xlab=L"\rho", ylab=L"\hat{R}^+_{%$ll,%$mm,%$nf}",
        label="backward", color=1)
    plot!(check.t, check_matrix[:,1], label="forward", color=2)


    # ----- IOD (backwards) -----
    p = @SVector [ll, w_mn, PP, true]
    u0 = last(sol.u)
    u0 = @SVector [u0[1], u0[2], u0[3], u0[4]]
    rhospan = (-PP.rho_apo, -PP.rho_peri)
    prob = ODEProblem(RHS_IOD, u0, rhospan, p)
    # reverse rho -> -rho, by changing saveat sign
    sol2 = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=-PP.rho_IOD)
    lambda_plus = last(sol2.u)[1]

    u_matrix = hcat(sol2.u...)'
    u_matrix = u_matrix/lambda_plus  # λ⁺ = 1
    PP.single_R_IOD = u_matrix[:,1] + 1im * u_matrix[:,2]
    PP.single_Q_IOD = u_matrix[:,3] + 1im * u_matrix[:,4]

    # Save ID region
    if save
        u_matrix = hcat(sol.u...)'
        u_matrix = u_matrix/lambda_plus  # λ⁺ = 1
        PP.single_R_ID = u_matrix[:,1] + 1im * u_matrix[:,2]
        PP.single_Q_ID = u_matrix[:,3] + 1im * u_matrix[:,4]
    end

    # ----- IOD check -----
    p = @SVector [ll, w_mn, PP, false]
    u0 = last(sol2.u)
    u0 = @SVector [u0[1], u0[2], u0[3], u0[4]]
    rhospan = (PP.rho_peri, PP.rho_apo)
    prob = ODEProblem(RHS_IOD, u0, rhospan, p)
    check = solve(prob, METHOD, abstol=ABSTOL, rtol=RTOL, saveat=PP.rho_HOD)

    uf = last(check.u)
    u0 = last(sol.u)
    error =  [abs(uf[1]-u0[1]), abs(uf[2]-u0[2]), abs(uf[3]-u0[3]), abs(uf[4]-u0[4])]
    println(error)

    sol_matrix = hcat(sol2.u...)'
    check_matrix = hcat(check.u...)'
    plot!(-sol2.t, sol_matrix[:,1], label="", color=1)
    plot!(check.t, check_matrix[:,1], label="", color=2)
    savefig("Rp"*string(ll)*"_"*string(mm)*"_"*string(nf))
end
