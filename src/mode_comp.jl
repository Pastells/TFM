using StaticArrays
using OrdinaryDiffEq
using PyCall

# Get absolute path
FOLDER = normpath(joinpath(@__FILE__,"..",".."))

machinery = pyimport("importlib.machinery")
loader = machinery.SourceFileLoader("Schwarzschild", FOLDER * "/src/Schwarzschild.py")
schw = loader.load_module("Schwarzschild")
include("./hyperboloidal_compactification_tanh.jl")

# -----------------------------------------------

function regge_wheeler_potential(rstar, ll)
    cb = ll * (ll + 1)
    xstar = 0.5 * rstar - 1.0
    xsch = schw.r_tortoise_to_x_schwarzschild(rstar)
    rsch = 2.0 * (1.0 + xsch)
    # exp(xstar-x) = x

    Vl =  2.0 * exp(xstar - xsch) * (cb + 2.0 / rsch) / (rsch^3)

    return Vl
end

# -----------------------------------------------

"""ODE system that determines the solution to the Master Equation in the Horizon Domain"""
function RHS_HD(u, p, t)

    ll, w_mn, PP = p
    rho = t
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
        if epsilon_rho < 1.0e-6

            dR = -1im * w_mn * R * exp(-1im * w_mn * epsilon_rho)
            dQ = -(w_mn^2) * R * exp(-1im * w_mn * epsilon_rho)

            # Integration not close to the Horizon
        else
            Omega = 1.0 - rho / PP.rho_H
            rstar = rho / Omega
            xsch = schw.r_tortoise_to_x_schwarzschild(rstar)
            rsch = 2.0 * (1.0 + xsch)
            rsch2 = rsch * rsch
            rsch3 = rsch2 * rsch

            exp_rstar_over_2M = exp(0.5 * rstar)
            regular_potential_factor = ( exp(-(1.0 + xsch)) * (2.0 / rsch3)
                                        * (ll * (ll + 1) + 2.0 * (PP.sigma_spin) / rsch)
                                       )

            H = 1.0 - Omega^2
            H_plus_one = 1.0 + H
            DH = (2.0 / PP.rho_H) * Omega
            DH_over_1minusH = (2.0 / PP.rho_H) / Omega
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

        Omega = 1.0 - f0 * rho / PP.rho_H
        dOmega_drho = -(f0 + rho * f1) / PP.rho_H
        d2Omega_drho2 = -(2.0 * f1 + rho * f2) / PP.rho_H

        LH = Omega - rho * dOmega_drho
        H = 1.0 - (Omega^2) / LH
        one_plus_H = 1.0 + H
        one_minus_H = 1.0 - H
        DH = -(Omega / LH) * (2.0 * dOmega_drho + rho * (Omega / LH) * d2Omega_drho2)

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

    @SVector [real(dR), imag(dR), real(dQ), imag(dQ)]
end

# -----------------------------------------------

"""ODE system that determines the solution to the Master Equation in the Horizon Orbital Domain"""
function RHS_HOD(u, p, t)
    ll, w_mn = p
    rho = t
    re_R, im_R, re_Q, im_Q = u
    R = re_R + 1im * im_R
    Q = re_Q + 1im * im_Q

    Vl = regge_wheeler_potential(rho, ll)

    dR = Q
    dQ = (Vl - w_mn^2) * R

    @SVector [real(dR), imag(dR), real(dQ), imag(dQ)]
end

# -----------------------------------------------

"""ODE system that determines the solution to the Master Equation in the Infinity Orbital Domain
Defined backward, e.i. rho -> -rho, so can be integrated from right to left"""
function RHS_IOD(u, p, t)
    ll, w_mn = p
    rho = t
    re_R, im_R, re_Q, im_Q = u
    R = re_R + 1im * im_R
    Q = re_Q + 1im * im_Q

    Vl = regge_wheeler_potential(rho, ll)

    dR = -Q
    dQ = -(Vl - w_mn^2) * R

    @SVector [real(dR), imag(dR), real(dQ), imag(dQ)]
end

# -----------------------------------------------

"""ODE system that determines the solution to the Master Equation in the Infinity Domain
Defined backward, e.i. rho -> -rho, so can be integrated from right to left"""
function RHS_ID(u, p, t)
    ll, w_mn, PP = p
    rho = t
    re_R, im_R, re_Q, im_Q = u
    re_RI, im_RI, re_QI, im_QI = u
    R = re_R + 1im * im_R
    Q = re_Q + 1im * im_Q
    dR, dQ = 0, 0

    # Some common Definitions
    cb = ll * (ll + 1)

    # Integration through the Infinity Domain (ID)
    if PP.rho_IC <= rho <= PP.rho_I
        epsilon_rho = PP.rho_I - rho

        # Integration near (null) Infinity
        if epsilon_rho < 1.0e-6

            # Integration for the Particular Case of Zero-Frequency Modes
            if w_mn < 1.0e-8

                sigma0 = 1.0
                sigma1 = (sigma0/((ll+1)*PPC.rho_I))*( cb - (1.0/PPC.rho_I)*(cb-1) )

                sigma2 = (sigma0/(4.0*(ll+1)*(ll+3/2)*(PPC.rho_I^2)))*( 2.0*(ll^4 + 2.0*ll^3 - ll^2 - 4.0*ll-1.0)*(1.0/(PPC.rho_I^2)) \
                                                                          - 4.0*(ll+1)*(ll+3/2)*(ll*(ll+1)-1.0)*(1.0/PPC.rho_I) + 2.0*ll*(ll+1)^2*(ll+3/2) )

                sigma3 = (sigma0/(12.0*(ll+2)*(ll+1)*(ll+3/2)*(PPC.rho_I^3)))*( -2.0*(ll^2+ll-1)*(ll^4+2.0*ll^3-ll^2-8.0*ll-7.0)*(1.0/(PPC.rho_I^3)) \
                                                                                   + 6.0*(ll+2)^2*(ll^4+2.0*ll^3-ll^2-4.0*ll-1.0)*(1.0/(PPC.rho_I^2)) - 6.0*(ll+2)^2*(ll+1)*(ll+3/2)*(ll^2+ll-1.0)*(1.0/PPC.rho_I) \
                                                                                   + 2*(ll+2)^2*(ll+1)^2*ll*(ll+3/2) )


                Sigma = sigma0 + sigma1 * epsilon_rho + sigma2 * epsilon_rho ^ 2 + sigma3 * epsilon_rho ^ 3
                dSigmadepsilon = sigma1 + 2.0*sigma2*epsilon_rho + 3.0*sigma3*epsilon_rho^2
                d2Sigmadepsilon2 = 2.0*sigma2 + 6.0*sigma3*epsilon_rho

                # Particular Case: ll = 0
                if ll == 0
                    dR = -dSigmadepsilon
                    dQ = -d2Sigmadepsilon2

                # Particular Case: ll = 1
                elseif ll == 1

                    dR = -Sigma - epsilon_rho * dSigmadepsilon
                    dQ = 2.0 * dSigmadepsilon + epsilon_rho * d2Sigmadepsilon2

                # All other ll different from 0 and 1
                else
                    dR = - (epsilon_rho^ll)*( ll*Sigma + epsilon_rho*dSigmadepsilon )
                    dQ = (epsilon_rho^(ll-2))*( ll*(ll-1)*Sigma + 2.0*ll*epsilon_rho*dSigmadepsilon + (epsilon_rho^2)*d2Sigmadepsilon2 )
                end

            # Integration for non-zero Frequency Modes
            else
                qireal = 0.0
                qiimag = w_mn*re_RI*(1.0 - cb/(2.0*(PPC.rho_I^2)*w_mn^2))

                r1real = qireal
                r1imag = qiimag

                q1real = -w_mn^2*re_RI*( 1.0 - cb*( 1.0 - (ll^2+ll+2.0)/(4*(PPC.rho_I^2)*w_mn^2) )/((PPC.rho_I^2)*w_mn^2) )
                q1imag = -w_mn^2*re_RI*( 1.0 - cb*(1.0-PPC.rho_I) )/((PPC.rho_I^4)*w_mn^3)

                r2real = -0.5*q1real
                r2imag = -0.5*q1imag

                q2real = (re_RI/(4.0*(PPC.rho_I^4)))*( cb*(PPC.rho_I - 1.0) + 1.0 )*( 2.0 - (cb + 6.0)/((PPC.rho_I^2)*w_mn^2) )
                q2imag = (w_mn^3*re_RI/2.0)*( 1.0 - (3.0*cb-4.0)/(2.0*(PPC.rho_I^2)*w_mn^2) \
                                              + cb*(3.0*cb-22.0)/(4.0*(PPC.rho_I^4)*w_mn^4) \
                                              + 4.0*(1.0 + 1.5*(cb-1.0)*(PPC.rho_I))/((PPC.rho_I^6)*w_mn^4) \
                                              - cb*(ll+3.0)*(ll-2.0)*(cb+2.0)/(8.0*(PPC.rho_I^6)*w_mn^6) )

                r3real = -q2real/3.0
                r3imag = -q2imag/3.0

                q3real = (w_mn^4*re_RI/6.0)*( 1.0 - 2.0*(cb-1.0)/((PPC.rho_I^2)*w_mn^2) \
                                              + (cb*(3.0*cb+2.0)+24.0)/(2.0*(PPC.rho_I^4)*w_mn^4) \
                                              + 12.0*(cb-1.0)*(3.0/((PPC.rho_I^2)*w_mn^2) - 1.0)/((PPC.rho_I^5)*w_mn^4) \
                                              - 8.0/((PPC.rho_I^6)*w_mn^4) \
                                              - cb*(ll^4 + 2.0*ll^3 - 7.0*ll^2 - 8.0*ll + 72.0)/(2.0*(PPC.rho_I^6)*w_mn^6) \
                                              + ( cb*(3.0*cb-2.0) + 27.0 )/((PPC.rho_I^8)*w_mn^6) \
                                              + cb*(ll+3.0)*(ll+4.0)*(ll-2.0)*(ll-3.0)*(cb+2.0)/(16.0*(PPC.rho_I^8)*w_mn^8) )

                q3imag = (re_RI/6.0)*( -4.0*cb*w_mn/(PPC.rho_I^3) \
                                      + 4.0*w_mn*(cb-1.0)/(PPC.rho_I^4) \
                                      + 4.0*cb*(ll+3.0)*(ll-2.0)/((PPC.rho_I^5)*w_mn) \
                                      - 4.0*(ll+4.0)*(ll-3.0)*(cb-1.0)/((PPC.rho_I^6)*w_mn) \
                                      + 48.0/((PPC.rho_I^7)*w_mn) \
                                      - cb*(cb^2-18.0)/((PPC.rho_I^7)*w_mn^3) \
                                      + (cb-1.0)*(cb^2-18.0)/((PPC.rho_I^8)*w_mn^3) )

                r4real = -0.25*q3real
                r4imag = -0.25*q3imag

                dR_real = -r1real - 2.0*r2real*epsilon_rho - 3.0*r3real*epsilon_rho^2 - 4.0*r4real*epsilon_rho^3
                dR_imag = -r1imag - 2.0*r2imag*epsilon_rho - 3.0*r3imag*epsilon_rho^2 - 4.0*r4imag*epsilon_rho^3
                dQ_real = -q1real - 2.0*q2real*epsilon_rho - 3.0*q3real*epsilon_rho^2
                dQ_imag = -q1imag - 2.0*q2imag*epsilon_rho - 3.0*q3imag*epsilon_rho^2
                dR = dR_real + 1im * dR_imag
                dQ = dQ_real + 1im * dQ_imag

            end

        # Integration not "close" to (null) Infinity
        else
            Omega = 1.0 - rho / PP.rho_I
            rstar = rho / Omega
            xsch = schw.r_tortoise_to_x_schwarzschild(rstar)
            rsch = 2.0 * (1.0 + xsch)
            rsch2 = rsch * rsch
            rsch3 = rsch2 * rsch

            f = 1.0 - 2.0 / rsch
            regular_potential_factor = f * ( ll * (ll + 1) + 2.0 * (PP.sigma_spin) / rsch)
            romega2 = (rho - 2.0 * Omega * log(xsch)) ^ 2

            H = 1.0 - Omega ^ 2
            H_plus_one = 1.0 + H
            DH = (2.0 / PP.rho_I) * Omega
            DH_over_1minusH = (2.0 / PP.rho_H) / Omega

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

        Omega = 1.0 - f0 * rho / PP.rho_I
        dOmega_drho = -(f0 + rho * f1) / PP.rho_I
        d2Omega_drho2 = -(2.0 * f1 + rho * f2) / PP.rho_I

        LI = Omega - rho * dOmega_drho
        H = 1.0 - (Omega ^ 2) / LI
        one_plus_H = 1.0 + H
        one_minus_H = 1.0 - H
        DH = -(Omega / LI) * (2.0 * dOmega_drho + rho * (Omega / LI) * d2Omega_drho2)

        rstar = rho / Omega
        xsch = schw.r_tortoise_to_x_schwarzschild(rstar)
        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch
        Vl = xsch * (2.0 / rsch3) * (ll * (ll + 1.0) + 2.0 * (PP.sigma_spin) / rsch)

        dR = Q
        dQ = (DH / one_minus_H) * (Q + 1im * w_mn * R) + 2im * w_mn * (H / one_minus_H) * Q +
         -(one_plus_H / one_minus_H) * (w_mn ^ 2) * R + (Vl / one_minus_H^2) * R

    elseif rho <= PP.rho_IS
        #    elseif rho <= PP.rho_IS and rho >= PP.rho_apo:
        xsch = schw.r_tortoise_to_x_schwarzschild(rho)  # rstar = rho

        rsch = 2.0 * (1.0 + xsch)
        rsch2 = rsch * rsch
        rsch3 = rsch2 * rsch

        Vl = xsch * (2.0 / rsch3) * (ll * (ll + 1.0) + 2.0 * (PP.sigma_spin) / rsch)

        dR = Q
        dQ = (Vl - w_mn^2) * R

    else
        print( "rho= ", rho, "  Out of Domain error during ODE Integration at the Horizon Domain")
    end

    @SVector [real(dR), imag(dR), real(dQ), imag(dQ)]
end

# -----------------------------------------------

"""Solve both regions, starting from the horizon and infinity
   uses static arrays to improve perfomance
   save :: keep HD and ID modes solutions"""
function compute_mode(ll, mm, nf, PP; method=TRBDF2(), save=false)
    omega_mn = nf * (PP.omega_r) + mm * (PP.omega_phi)
    p = @SVector [ll,omega_mn,PP]

    # ----- HD -----
    # print("HD")
    u0 = @SVector [1, 0, 0, -p[2]]
    tspan = (PP.rho_H_plus, PP.rho_peri)
    prob = ODEProblem(RHS_HD,u0,tspan,p)
    # save at whole array or only last point
    save ? saveat=PP.rho_HD : saveat=PP.rho_peri
    sol = solve(prob, method, abstol=1e-14, rtol=1e-12, saveat=saveat)
    lambda_minus = last(sol.u)[1]

    if save
        u_matrix = hcat(sol.u...)'
        u_matrix = u_matrix/lambda_minus  # λ⁻ = 1
        PP.single_R_HD = u_matrix[:,1] + 1im * u_matrix[:,2]
        PP.single_Q_HD = u_matrix[:,3] + 1im * u_matrix[:,4]
    end

    # ----- HOD -----
    # print("HOD")
    u0 = last(sol.u)
    u0 = @SVector [u0[1], u0[2], u0[3], u0[4]]
    tspan = (PP.rho_peri, PP.rho_apo)
    prob = ODEProblem(RHS_HOD,u0,tspan,p)
    sol = solve(prob, method, abstol=1e-14, rtol=1e-12, saveat=PP.rho_HOD)

    u_matrix = hcat(sol.u...)'
    u_matrix = u_matrix/lambda_minus  # λ⁻ = 1

    PP.single_R_HOD = u_matrix[:,1] + 1im * u_matrix[:,2]
    PP.single_Q_HOD = u_matrix[:,3] + 1im * u_matrix[:,4]

    # ----- ID (backwards) -----
    # print("ID")
    u0 = @SVector [1,0,0,p[2]]
    tspan = (-PP.rho_I,-PP.rho_apo)
    prob = ODEProblem(RHS_ID,u0,tspan,p)
    # save at whole array or only last point
    save ? saveat=PP.rho_ID : saveat=PP.rho_apo
    # reverse time t -> -t, by changing saveat sign
    sol = solve(prob, method, abstol=1e-14, rtol=1e-12, saveat=-saveat)
    lambda_plus = last(sol.u)[1]

    if save
        u_matrix = hcat(sol.u...)'
        u_matrix = u_matrix/lambda_plus  # λ⁻ = 1
        PP.single_R_ID = u_matrix[:,1] + 1im * u_matrix[:,2]
        PP.single_Q_ID = u_matrix[:,3] + 1im * u_matrix[:,4]
    end

    # ----- IOD (backwards) -----
    # print("IOD")
    u0 = last(sol.u)
    u0 = @SVector [u0[1], u0[2], u0[3], u0[4]]
    tspan = (-PP.rho_apo, -PP.rho_peri)
    prob = ODEProblem(RHS_IOD,u0,tspan,p)
    # reverse time t -> -t, by changing saveat sign
    sol = solve(prob, method, abstol=1e-14, rtol=1e-12, saveat=-PP.rho_IOD)

    u_matrix = hcat(sol.u...)'
    u_matrix = u_matrix/lambda_plus  # λ⁺ = 1
    PP.single_R_IOD = u_matrix[:,1] + 1im * u_matrix[:,2]
    PP.single_Q_IOD = u_matrix[:,3] + 1im * u_matrix[:,4]
    return
end
