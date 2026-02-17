using DifferentialEquations, Plots, OrdinaryDiffEq, Sundials, LinearAlgebra, CSV, DataFrames, Statistics
# Define Functions

# Net carbon fractionation
function calc_a13C_net(reversibility, a13, a13eq)
    a13C=zeros(length(reversibility)-1) 
    # carbon isotope fractionation, 
    # keep in mind that there is no isotope fractionation for step 8
    pos=[7 6 5 4 3 2 1 8]
    a13C_net=1
    for i in 1:(length(reversibility)-1)
        a13C_net=(a13eq[pos[i]]*a13C_net-a13[pos[i]])*reversibility[pos[i]]+a13[pos[i]] 
        # A step-wise calculation of net a13C, 
        # in the order of ch3-scom, ch3-h4mpt, ch2=h4mpt, ch-=h4mpt, cho-h4mpt
        # cho-mfr, co2 (in), co2 (out)
        a13C[i]=a13C_net
    end
    return a13C, a13C_net
end

# # Net hydrogen fractionation
# function frac_hydr_ode!(ddt, C, p, t)
#     # Unpack parameters
#     phi, akf, akr, Rx = p
#     jf = phi[:, 1]
#     jr = phi[:, 2]
#     jnet = phi[1, 3]

#     # Unpack state variables
#     Rw, Rv, Ru1, RtS = C[1], C[2], C[3], C[4]
#     Rs, Rr1, Rp = C[5], C[6], C[7]
#     Ru2, Rr2, RtR = C[8], C[9], C[10]

#     # Precompute fluxes
#     jf1, jf2, jf3, jf4, jf5, jf6, jf7, jf8, jf9 = jf
#     jr1, jr2, jr3, jr4, jr5, jr6, jr7, jr8, jr9 = jr
#     akf1, akf2, akf3, akf4, akf5, akf6, akf7, akf8, akf9, akf10, akf11, akf12, akf13 = akf
#     akr1, akr2, akr3, akr4, akr5, akr6, akr7, akr8, akr9, akr10, akr11, akr12, akr13 = akr

#     # Forward and reverse fluxes
#     xw   = jf1 * Rx * akf1
#     wx   = jr1 * Rw * akr1
#     wv   = jf2 * Rw * akf2
#     vw   = jr2 * Rv * akr2
#     vu1  = jf3 * Rv * akf3
#     u1v  = jr3 * Ru1 * akr3
#     u1tS = jf4 * Ru1 * akf4
#     tSu1 = jr4 * RtS * akr4
#     tSs  = jf5 * RtS * akf5
#     tRs  = jf5 * RtR * akf6
#     stS  = jr5 * Rs * akr5
#     stR  = jr5 * Rs * akr6
#     sr1  = 3 * jf6 * Rs * akf7
#     r1s  = 3 * jr6 * Rr1 * akr7
#     r1p  = 3 * jf7 * Rr1 * akf8
#     pr1  = 3 * jr7 * Rp * akr8
#     xu2  = jf8 * Rx * akf9
#     u2x  = jr8 * Ru2 * akr9
#     u2s  = jf5 * Ru2 * akf11
#     su2  = jr5 * Rs * akr11
#     r2p  = jf7 * Rr2 * akf13
#     pr2  = jr7 * Rp * akr13
#     xr2  = jf9 * Rx * akf12
#     r2x  = jr9 * Rr2 * akr12
#     u2tR = jf4 * Ru2 * akf10
#     tRu2 = jr4 * RtR * akr10
#     pout = 4 * jnet * Rp

#     # Differential equations
#     ddt[1]  = xw + vw - wx - wv            # CHO-MFR
#     ddt[2]  = wv + u1v - vw - vu1          # CHO-H4MPT
#     ddt[3]  = vu1 + tSu1 - u1v - u1tS      # CH-H4MPT
#     ddt[4]  = u1tS + stS - tSu1 - tSs      # CH2-H4MPT (S)
#     ddt[10] = u2tR + stR - tRu2 - tRs      # CH2-H4MPT (R)
#     ddt[5]  = tSs + tRs + u2s + r1s - stS - stR - su2 - sr1 # CH3-H4MPT
#     ddt[6]  = sr1 + pr1 - r1s - r1p        # CH3-S-CoM
#     ddt[7]  = r1p + r2p - pr1 - pr2 - pout # CH4: p 
#     ddt[8]  = xu2 + tRu2 + su2 - u2x - u2tR - u2s # f420h2: u2
#     ddt[9]  = xr2 + pr2 - r2x - r2p        # hscob: r2
# end

# # Net 13CD fractionation
# # Construct ode set for 13CD
# function frac_CD_ode!(ddt, C, p, t)
#     # Unpack parameters from tuple p
#     phi, RxH, RxCin, Ru2_D, Rr2_D, Ru1_13, Rt_13, Rr1_13, Rxh2, akinf, akinr = p 
#     phif = phi[:, 1]
#     phir = phi[:, 2]
#     phinet = phi[1, 3]
#     # Only 8 relevant species containing 13CD
#     R_qh_w  = C[1] # cho-mfr
#     R_qh_v  = C[2] # cho-h4mpt
#     R_qh_u1 = C[3] # ch-h4mpt
#     R_qh_tS = C[4] # ch2-h4mpt (S)
#     R_qh_tR = C[5] # ch2-h4mpt (R)
#     R_qh_s  = C[6] # ch3-h4mpt
#     R_qh_r1 = C[7] # ch3-s-com
#     R_qh_p  = C[8] # ch4

#     xw    = phif[1] * RxH * RxCin * akinf[1]
#     wx    = phir[1] * R_qh_w * akinr[1]
#     wv    = phif[2] * R_qh_w * akinf[2]
#     vw    = phir[2] * R_qh_v * akinr[2]
#     vu1   = phif[3] * R_qh_v * akinf[3]
#     u1v   = phir[3] * R_qh_u1 * akinr[3]
#     u1tS  = phif[4] * R_qh_u1 * akinf[4]
#     tSu1  = phir[4] * R_qh_tS * akinr[4]
#     tSs   = phif[5] * R_qh_tS * akinf[5]
#     tRs   = phif[5] * R_qh_tR * akinf[6] # Secondary isotope effects
#     sr1   = 3 * phif[6] * R_qh_s * akinf[7] # Secondary isotope effect CH3-H4MPT -> CH3-SCoM
#     r1s   = 3 * phir[6] * R_qh_r1 * akinr[7] # Secondary isotope effect CH3-SCoM -> CH3-H4MPT
#     r1p   = 3 * phif[7] * R_qh_r1 * akinf[8] # Secondary isotope effect CH3-SCoM -> CH4
#     u2tR  = Ru1_13 * phif[4] * Ru2_D * akinf[9] # Primary isotope effects
#     tRu2  = R_qh_tR * phir[4] * akinr[9]
#     u2s   = phif[5] * Rt_13 * Ru2_D * akinf[11]
#     r2p   = phif[7] * Rr1_13 * Rr2_D * akinf[12]
#     pout  = 4 * phinet * R_qh_p
#     stS   = phir[5] * R_qh_s * akinr[5]
#     stR   = phir[5] * R_qh_s * akinr[6]
#     pr1   = 3*phir[7] * R_qh_p * akinr[8]
#     su2   = phir[5] * R_qh_s * akinr[11]
#     pr2   = phir[7] * R_qh_p * akinr[12]

#     ddt[1] = xw + vw - (wx + wv)                 
#     ddt[2] = wv + u1v - (vw + vu1)               
#     ddt[3] = vu1 + tSu1 - (u1v + u1tS)           
#     ddt[4] = u1tS + stS - (tSu1 + tSs)           
#     ddt[5] = u2tR + stR - (tRu2 + tRs)            
#     ddt[6] = tSs + tRs + u2s + r1s - (stS + stR + su2 + sr1)  
#     ddt[7] = sr1 + pr1 - (r1s + r1p)             
#     ddt[8] = r1p + r2p - (pr1 + pr2 + pout)      
# end

# # Calculate D12CH2D2
# function frac_DD_ode!(ddt, C, p, t)
#     # Unpack parameters
#     phi, RxH2, R_D_i, R_D_j, R_D_u, R_D_r, R_D_tS, R_D_tR, akinf, akinr = p

#     # Initialize derivative array
#     phif = phi[:, 1]
#     phir = phi[:, 2]
#     phinet = phi[1, 3]

#     R_DD_t = C[1]
#     R_DD_s = C[2]
#     R_DD_r = C[3]
#     R_DD_p = C[4]

#     u_t   = R_D_u * phif[4]*R_D_i*akinf[4]
#     t_u   = R_DD_t * phir[4]*akinr[4]
#     t_s   = phif[5]*R_DD_t*akinf[1]
#     s_t   = phir[5]*R_DD_s*akinr[1]
#     ti_s  = phif[5]*R_D_i*(R_D_tS*akinf[6] + R_D_tR*akinf[7])
#     s_ti  = phir[5]*R_DD_s*(akinr[6] + akinr[7])
#     s_r   = 3*phif[6]*R_DD_s*akinf[2]
#     r_s   = 3*phir[6]*R_DD_r*akinr[2]
#     rj_p  = 3*phif[7]*R_D_j*R_D_r*akinf[8]
#     p_rj  = 3*phir[7]*R_DD_p*akinr[8]
#     r_p   = 3*phif[7]*R_DD_r*akinf[3]
#     p_r   = 3*phir[7]*R_DD_p*akinr[3]
#     p_out = 6*R_DD_p*phinet

#     ddt[1] = u_t + s_t - (t_u + t_s)
#     ddt[2] = t_s + ti_s + r_s - (s_t + s_ti + s_r)
#     ddt[3] = s_r + p_r - (r_s + r_p)
#     ddt[4] = rj_p + r_p - (p_rj + p_r + p_out)
# end

# # Define a function to implement the change of H2,CO2,CH4 concentrations
# function calc_isotope_effect(H2,CO2,CH4, pin)
#     # unpack parameters
#     R, Tk, Vm, km, dGstd, c0, Yatp, Vcell, dz, D, rad, A, R_13CCO2, R_H2O, R_H2, a13kff, a13kf, a13kr, a13eq, a2kf, a2kr, a2eq,a132kf,a132kr,a132eq,a22kf,a22kr,a22eq = pin
#     dat=zeros(length(H2)*length(CO2)*length(CH4),7)
#     counter=1
#     for i in 1:length(H2)
#         c0[18]=H2[i]
#         for j in 1:length(CO2)
#             c0[19]=CO2[j]
#             for k in 1:length(CH4)
#                 c0[17]=CH4[k]

#                 tspan = (0.0,1e4)
#                 prob = ODEProblem(flux_ode!, c0, tspan)
#                 sol = solve(prob, CVODE_BDF(),saveat=0.1, reltol=1e-4, abstol=1e-18)
#                 # dGr_net
#                 dGr_net=calc_dG_net(Tk,c0[19],c0[17],c0[18])
#                 print(dGr_net, "\n")
#                 # Calculate reversibility of each step
#                 reversibility, flux_mtx=calc_rev(sol)
#                 # Calculate isotope fractionations and compound-specific isotope values
#                 # Carbon
#                 a13C_step, a13C_net=calc_a13C_net(reversibility,a13kff,a13eq)
#                 print("13αCO2-CH4: ", a13C_net, "\n")
#                 # Calculate carbon isotope values of each intermediates
#                 R_p_13C   = R_13CCO2 ./ a13C_net          # CH4
#                 R_r1_13C  = R_p_13C.*a13C_step[1] # CH3-S-CoM
#                 R_s_13C  = R_p_13C.*a13C_step[2] # CH3-H4MPT
#                 R_t_13C   = R_p_13C.*a13C_step[3] # CH2-H4MPT
#                 R_u1_13C = R_p_13C.*a13C_step[4] # CH-H4MPT
#                 R_v_13C  = R_p_13C.*a13C_step[5] # CHO-H4MPT
#                 R_w_13C  = R_p_13C.*a13C_step[6] # CHO-MFR
#                 R_i_13C   = R_p_13C.*a13C_step[7] # intracellular CO2

#                 # Calculate net hydrogen fractionation factors
#                 params = (flux_mtx, a2kf, a2kr, R_H2O)
#                 ch=[1,1,1,1,1,1,1,1,1,1]*1.0e-4
#                 tspan = (0.0,5e7)
#                 prob2H = ODEProblem(frac_hydr_ode!, ch, tspan, params)
#                 sol2H = solve(prob2H, CVODE_BDF(),saveat=10, reltol=1e-2, abstol=1e-5)
#                 a2H_net=sol2H[7,end]/R_H2O
#                 print("2αCH4-H2O: ", a2H_net, "\n")
#                 # Derive compound specific isotope ratios
#                 R_w_D = sol2H[1,end] # CHO-MFR
#                 R_v_D = sol2H[2,end] # CHO-H4MPT
#                 R_u1_D = sol2H[3,end] # CH-H4MPT
#                 R_tS_D = sol2H[4,end] # CH2-H4MPT (S)
#                 R_s_D = sol2H[5,end] # CH3-H4MPT
#                 R_r1_D = sol2H[6,end] # CH3-SCoM
#                 R_p_D = sol2H[7,end]  # CH4
#                 R_u2_D = sol2H[8,end] # F420
#                 R_r2_D = sol2H[9,end] # HS-CoB
#                 R_tR_D = sol2H[10,end] # CH2-H4MPT (R)

#                 # plot(sol2H.t[1:100:end],sol2H[7,1:100:end]./R_H2O) # Plot to check if the isotope has reached steady state

#                 # D13CH3D
#                 params = (flux_mtx, R_H2O, R_i_13C, R_u2_D, R_r2_D, R_u1_13C, R_t_13C, R_r1_13C, R_H2, a132kf, a132kr)
#                 cqh=[1,1,1,1,1,1,1,1].*1.0e-6
#                 probCD = ODEProblem(frac_CD_ode!, cqh, tspan, params)
#                 solCD = solve(probCD, CVODE_BDF(),saveat=10, reltol=1e-2, abstol=1e-5)
#                 # Convert the results to Delta13CD
#                 RCD=solCD[:,end]
#                 D13CH3D=1000*(RCD[8]/(R_p_13C*R_p_D)-1)
#                 print("D13CH3D:", D13CH3D, "\n")
#                 # plot(solCD.t[1:100:end],solCD[8,1:100:end]) # Plot to check if the isotope has reached steady state

#                 # D12CH2D2
#                 params= (flux_mtx, R_H2, R_u2_D, R_r2_D, R_u1_D, R_r1_D, R_tS_D, R_tR_D, a22kf, a22kr)
#                 chh=[1,1,1,1]*1.0e-8
#                 probDD = ODEProblem(frac_DD_ode!, chh, tspan, params)
#                 solDD = solve(probDD, CVODE_BDF(),saveat=10, reltol=1e-2, abstol=1e-5)
#                 # Convert the results to Delta12CH2D2
#                 RDD=solDD[:,end]
#                 D12CH2D2=1000 .* (RDD[4] ./ (R_p_D .^ 2)-1)
#                 print("D12CH2D2:", D12CH2D2, "\n")
#                 # plot(solDD.t[1:100:end],solDD[4,1:100:end]) # Plot to check if the isotope has reached steady state
#                 dat[counter,:]=[c0[18],c0[19],c0[17],a13C_net,a2H_net,D13CH3D,D12CH2D2]
#                 counter+=1
#             end
#         end
#     end
#     return dat
# end

# Constants
const R = 8.31446
Tk = 35 + 273.15

# Model parameters
Vm = [6.09, 28.53, 4.03, 12.92, 6.09, 2.38, 10.41, 14.15, 19.02] 
# Maximum enzyme capacity in mM/s
Vm .= Vm./1000 # Convert to M/s
Vm .= Vm.*10.6 # Correcting factor in Gropp et al., 2022
km = [
    6.800, 6.800, 6.800, 0.030, 0.010,
    0.050, 0.060, 0.005, 0.005,
    0.148, 0.167,
    0.05, 0.016, 0.033, 0.065,
    0.3, 0.003, 0.024, 0.040,
    0.135, 0.277, 0.559, 0.098,
    0.821, 0.204, 0.110, 0.110,
    0.036, 0.012, 0.012,
    0.03, 0.145, 0.01, 0.2, 0.2, 0.075
]
# Saturation coefficient in M-M equations, in mM
km.=km./1000 # Convert to M
dGstd = [10.3, -3.5, -4.2, 2.0, -1.7, -26.9, -17.9, -28, -90.8] # in kJ

# Initial concentrations, in M
# In the order of 
# CHO-MFR, MFR, Fdox,Fdred, H4MPT,CHO-H4MPT,CH=-H4MPT,CH2=H4MPT,
# F420, F420H2, CH3H4MPT, HS-CoM, CH3-SCoM, HS-CoB, CoM-S-S-CoB,
# CO2_in, CH4_in,H2_in, CO2_out
c0 = [
    1.0e-7, 1.8e-3, 5.0e-3, 1e-7, 1.2e-3, 1.5e-5, 7.5e-5, 2.5e-2,
    5.8e-4, 1e-7, 1.5e-2, 6.0e-3, 1.0e-7, 6.0e-3, 1.0e-7,
    1e-2, 1e-5, 1e-8, 0.01
]

# ATP scaling factor
Yatp=0.2
# Cell parameters
Vcell=2.00e-15 # in Liters
dz=0.5e-9 # in m, ~0.5 nm
D=2.9e-9 # Diifusivity constant, m2/s
rad=(Vcell*1e-3*3/4/pi)^(1/3) # in meters
A=4*pi*rad^2 # in m2

# Set the isotope values of substrates 
d13CCO2 = -36.0 # permil d13C_CO2
dDH2O = -50.0 # permil dD_H2O

RVPDB = 0.011202 # Standard carbon isotope ratio (VPDB)
RVSMOW = 1.5576e-4 # Standard hydrogen isotope ratio (VSMOW)
R_13CCO2 = (d13CCO2 ./ 1000 + 1) .* RVPDB # Carbon isotope ratio of CO2
R_H2O = (dDH2O ./ 1000 + 1) .* RVSMOW # Hydrogen isotope ratio of H2O

# Find temperature dependent equilibrium isotope fractionation between H2O and H2
aH2O_H2_eq = 0.0334 .* 1e12 ./ Tk.^4 - 0.2513 .* 1e9 ./ Tk.^3 + 1.0267 .* 1e6 ./ Tk.^2 - 1.2166 .* 1e3 ./ Tk + 1.7321
R_H2 = R_H2O/aH2O_H2_eq # Hydrogen isotope ratio of H2, assuming rapid equilibration between H2 and H2O. 

#------------------------------ Fractionation Factors of each step -----------------------------------
# Read the csv dataframe for the KFFs of 13C and d2H
kff_13C_dat = CSV.read("kff13C_for.csv", DataFrame)
a13kff=[median(kff_13C_dat[!, col]) for col in names(kff_13C_dat)] # Extract the median values for KFF
a13kf=1 ./ a13kff # Fractionation factors
# EQUILIBRIUM Isotope Fractionation Factors from Gropp et al., GCA (2020)
# b_vals: 7×5 matrix of polynomial coefficients, for the carbon isotope fractionation of each step
# Each row are the coefficients for calculating the EFF of each reaction involving carbon
b_vals = [
    -0.03507   0.89189  -5.78478  16.97954  -7.27610;
     0.02543  -0.28251   0.78966   1.26127  -0.16561;
     0.02412  -0.23329   0.55401  -1.22821   0.26467;
     0.11085  -1.18789   4.54994  -0.98362   0.14324;
     0.02240  -0.57894   4.73613  -6.21676   2.17747;
     0.06685  -0.84498   4.16633  -1.80360   0.51959;
    -0.04853   0.32124   0.93815  -4.85941   1.76778
]

# Compute 1000 * ln(α) values (klna13eq)
klna13eq = b_vals[:,1] .* 1e12 ./ Tk^4 .+ b_vals[:,2] .* 1e9 ./ Tk^3 .+
           b_vals[:,3] .* 1e6 ./ Tk^2 .+ b_vals[:,4] .* 1e3 ./ Tk .+
           b_vals[:,5]

# Convert to α values, this is alpha_CO2-CH4, or 1/EFF
a13eq = exp.(klna13eq ./ 1000)

# Add an 8th value manually for the diffusivity effect
a13eq = vcat(a13eq, 1.0)  # a13eq[8] = 1
a13kr = a13eq .* a13kf # carbon fractionation in the reverse direction

# Hydrogen fractionation Factors
a2H_dat = CSV.read("a2H_for.csv", DataFrame)
a2kf=[median(a2H_dat[!, col]) for col in names(a2H_dat)] # Extract the median values for KFF
# Define the b_vals matrix (14×5 matrix), from Gropp et al., GCA (2020), 
# containing 14 kff2H values, the last one is for Hmd, can ignore for now
b_vals_h = [
     5.2327  -49.0324  166.1621 -245.9147  252.9625;
     0.0797   -1.0541    3.6232   11.9088    0.6043;
     0.8225   -9.7259   39.5466  -82.2848   23.5085;
    -1.1204   13.8748  -61.0661   73.3352  -18.9078;
     0.2246   -4.0886   28.8637  -38.4722   14.1934;
     0.3687   -5.7787   36.1315  -51.4672   18.8685;
     0.3490   -4.2149   18.4743  -16.3863    4.8821;
    -0.1854    0.8537    9.8527  -22.0439    9.7272;
     5.3661  -50.6782  171.3311 -248.3334  259.0032;
    -0.4436    5.9171  -28.5163   15.6731   -4.1018;
    -0.1139    0.5054    6.3783  -34.0472   13.9050;
     6.1915  -63.5921  249.1940 -174.4374  228.2042;
    -0.7896   10.2625  -44.1298 -144.5034   58.0480;
    -3.4212   46.8356 -255.0463   38.4767   46.9858
]

# Compute klna2eq for each reaction
klna2eq = b_vals_h[:,1] .* 1e12 ./ Tk.^4 .+ 
          b_vals_h[:,2] .* 1e9  ./ Tk.^3 .+ 
          b_vals_h[:,3] .* 1e6  ./ Tk.^2 .+ 
          b_vals_h[:,4] .* 1e3  ./ Tk    .+ 
          b_vals_h[:,5]

# Convert to equilibrium fractionation factors
a2eq = exp.(klna2eq ./ 1000)

# Calculate kinetic reverse fractionation
a2kr = a2eq .* a2kf

# EQUILIBRIUM Isotope Fractionation Factors, taken from Gropp et
# al., GCA, doi.org/10.1016/j.gca.2020.10.018
b_vals_qh = [5.04286 -46.55646  154.26902 -220.44125 241.20163
           0.11987  -1.46945    4.88531   12.53630   0.74283
           0.84641  -9.99803   40.49548  -84.72305  24.53775
          -1.04439  13.08131  -58.23150   75.42109 -20.49841
           0.27840  -4.97688   34.63646  -46.41678  17.36500
           0.38561  -6.30570   40.61156  -57.46741  20.95643
           0.36154  -4.50907   20.64966  -15.06848   3.60230
          -0.17407   0.60937   12.69865  -29.73599  13.05395
          -0.39285   5.37175  -26.59094   17.44264  -4.97376
          -3.35961  46.21746 -253.01593   40.37976  45.84330
          -0.07497  -0.18489   11.09870  -41.64583  17.35250
          -0.83925  10.64855  -43.85178 -149.76316  60.55202]

klna132eq = b_vals_qh[:,1].*1e12 ./Tk.^4 .+ b_vals_qh[:,2].*1e9 ./Tk.^3 .+ 
          b_vals_qh[:,3].*1e6 ./Tk.^2  .+ b_vals_qh[:,4].*1e3 ./Tk .+ 
          b_vals_qh[:,5]

a132eq = exp.(klna132eq./1000);

# Define the gamma values
# Define minimal primary (p) and secondary (s) gamma range.
gammap = 0.997 # Primary clumped isotope effect varies between 0.997-1
gammas = 1

gammaf = ones(1,12)
gammaf[2:8] .= gammas
gammaf[[1 ; 9:12]] .= gammap
# Random numbers between 0.997 and 1 for primary isotope effects
# KFFs from gamma and optimized C and H KFFs
a132kf=zeros(12)
a132kf[1]  = gammaf[1].*a2kf[1].*a13kf[1]   # Fmd, p
a132kf[2]  = gammaf[2].*a2kf[2].*a13kf[2]   # Ftr, s
a132kf[3]  = gammaf[3].*a2kf[3].*a13kf[3]   # Mch, s
a132kf[4]  = gammaf[4].*a2kf[4].*a13kf[4]   # Mtd, s (u1->tS)
a132kf[5]  = gammaf[5].*a2kf[5].*a13kf[5]   # Mer, s (tS->s)
a132kf[6]  = gammaf[6].*a2kf[6].*a13kf[5];   # Mer, s (tR->s)
a132kf[7]  = gammaf[7].*a2kf[7].*a13kf[6];   # Mtr, s
a132kf[8]  = gammaf[8].*a2kf[8].*a13kf[7];   # Mcr, s
a132kf[9]  = gammaf[9].*a2kf[10].*a13kf[4];  # Mtd, p
a132kf[10] = gammaf[10].*a2kf[14].*a13kf[4]; # Hmd, p
a132kf[11] = gammaf[11].*a2kf[11].*a13kf[5]; # Mer, p
a132kf[12] = gammaf[12].*a2kf[13].*a13kf[7]; # Mcr, p

a132kr = a132eq.*a132kf;
#-----------------------------------------------------------------
# DD isotopologue fractionation factors
# Define minimal primary (p) and secondary (s) gamma. gamma < 0 is normal. 
gammap = 0.994
gammas = 1

gammaDDf = ones(1,8);
gammaDDf[1:3] .= gammas
gammaDDf[4:8] .= gammap
# EQUILIBRIUM Isotope Fractionation Factors (from Gropp et al., 2020)
b = [
    9.96831E-18  -4.26734E-14  7.60881E-11  -7.28978E-08  4.01300E-05  -1.23247E-02  2.70904E+00;  # Mer s
    1.55513E-18  -6.74282E-15  1.23156E-11  -1.23331E-08  7.35249E-06  -2.59468E-03  1.45398E+00;  # Mtr s
    1.15066E-17  -4.88781E-14  8.61571E-11  -8.10885E-08  4.33637E-05  -1.26788E-02  2.61499E+00;  # Mcr s
   -5.35819E-19   2.68382E-15 -5.95919E-12   7.66964E-09 -6.15692E-06   2.97095E-03  3.20812E-01;  # Mtd p
    6.58835E-18  -2.85197E-14  5.05845E-11  -4.60941E-08  2.11152E-05  -2.90011E-03  1.06681E-01;  # Hmd p
    5.13100E-18  -2.19709E-14  3.91503E-11  -3.74090E-08  2.04360E-05  -6.14045E-03  1.79016E+00;  # Mer pS
    5.02110E-18  -2.15090E-14  3.83592E-11  -3.67124E-08  2.01145E-05  -6.07080E-03  1.78194E+00;  # Mer pR
   -2.38222E-19   1.40757E-15 -3.65466E-12   5.44542E-09 -5.09692E-06   3.04268E-03 -2.31867E-02   # Mcr p
]

# Calculate a22eq
a22eq = b[:,1] .* Tk^6 + b[:,2].* Tk^5 + b[:,3].*Tk^4 + b[:,4].*Tk^3 +
    b[:,5].*Tk^2 + b[:,6].*Tk .+ b[:,7]

# Calculate KFFs
a22kf = zeros(length(gammaDDf))  # Preallocate 8-element array

a22kf[1] = gammaDDf[1] * a2kf[5] * a2kf[6]       # Mer, s (tS->s)
a22kf[2] = gammaDDf[2] * a2kf[7]^2                 # Mtr, s
a22kf[3] = gammaDDf[3] * a2kf[8]^2                 # Mcr, s
a22kf[4] = gammaDDf[4] * a2kf[4] * a2kf[10]      # Mtd, p
a22kf[5] = gammaDDf[5] * a2kf[4] * a2kf[14]      # Hmd, p
a22kf[6] = gammaDDf[6] * a2kf[5] * a2kf[11]      # Mer, pS
a22kf[7] = gammaDDf[7] * a2kf[6] * a2kf[11]      # Mer, pR
a22kf[8] = gammaDDf[8] * a2kf[8] * a2kf[13]      # Mcr, p

# Calculate reverse KFFs
a22kr = a22eq .* a22kf

# Define H2,CO2,CH4 concentrations
# a dictionary for parameters
p = (R, Tk, Vm, km, dGstd, c0, Yatp, Vcell, dz, D, rad, A, R_13CCO2, R_H2O, R_H2, a13kff, a13kf, a13kr, a13eq, a2kf, a2kr, a2eq, a132kf, a132kr, a132eq, a22kf, a22kr, a22eq)
exp_range=range(-8,stop=-3,length=6)
H2_range=10 .^ exp_range
CO2_range=[1e-3]
CH4_range=[1e-5]
output=calc_isotope_effect(H2_range,CO2_range,CH4_range,p)