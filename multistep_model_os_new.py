# This is a model to model the reversibility of AOM
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

output=False # True if you want to save the plots as pdf
ff="ab initio" # Select the KIEs in the model: 
model="no INT"
F420=True # equilibirum with F420. Direct equilibrium with water is set at False
os=True
rev1_list=[0.0,0.4,0.5,0.6,0.8] # First-step reversibility list, CH4 <-> CH3-SCoM; each value ranges from 0 to <1
rev2_list=[0.0,0.99,0.8,0.8,0.8] # Second-step reversibility list, CH3-SCoM <-> CHO-MFR; each value ranges from 0 to <1
rev3_list=[0.0,0.3,0.5,0.6,0.8] # Third-step reversibility list, CHO-MFR <-> CO2; each value ranges from 0 to <1
t_lower=0.000 # minimum time for time interval
time_list=[35.0,50.0,70.0,90.0,70.0] # Maximum time for time interval, relevant to the final fraction of methane left
num=100000 # Number of tim steps
# "experiment": data from Scheller et al., 2013;
# "Ab initio": ab initio calculation in this study
dDH2O = -50.0 # permil dD_H2O
d13C_DIC=-15.0 # permil
RVPDB = 0.0112372 # Standard carbon isotope ratio (VPDB)
RVSMOW = 1.5576e-4 # Standard hydrogen isotope ratio (VSMOW)
RH2O=RVSMOW*(dDH2O/1000+1) # D/H ratio in water
FH2O=RH2O/(1+RH2O) # D/(D+H) ratio in water
R13C_DIC=RVPDB*(d13C_DIC/1000+1)
F13C_DIC=R13C_DIC/(1+R13C_DIC)
# For open system, define a phi value, phi=Jnet/Jadvin (net oxidation rate/advection in rate)  
phi=0.75
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Equlibrium isotope effect of the first step, from Gropp et al., 2021 (50 degree C)
a1cfeq=1/np.exp(0.8/1000)
a1dfeqp=1/np.exp(-580.0/1000) # Primary equilibrium fractionation
a1dfeqs=1/np.exp(44.2/1000) # Secondary equilibrium fractionation
a1cdfeqp=1/np.exp(-584.2/1000) # Primary
a1cdfeqs=1/np.exp(44.7/1000) # Secondary
a1ddfeqp=1/np.exp(-550.9/1000) # Primary
a1ddfeqs=1/np.exp(86.1/1000) # Secondary
# Step 1: MCR-catalyzed step
# The experimental results from Scheller 2013, methane activation by mcr
if ff=="experiment":    
    a1cfb= 1/1.039 # Carbon isotope effect KIE=1.039, methane formation dirction
    a1cff=a1cfb/a1cfeq
    a1dffp= 1/2.44 # primary hydrogen isotope fractionation, KIE 2.44+-0.22
    a1dffs= 1/1.17 # secondary hydrogen isotope fractionation, KIE 1.17+-0.05
    a1dfnet=1/4*a1dffp+3/4*a1dffs # Net hydrogen isotope fractionation
    gammaCD=0.991
    gammaCDp=0.875
    a1cdffnet=gammaCD*a1cff*a1dfnet
    a1cdffp=a1dffp*a1cff*gammaCDp # primary clumped isotopologue 13CD fractionation factor, net is 0.7559
    a1cdffs= (a1cdffnet-a1cdffp/4)*(4/3) # secondary clumped isotopologue 13CD fractionation factor
    gammaDD=0.968
    p=2.45 # alpha_s/alpha_p=2.09 +- 0.15, from Scheller et al.
    a1ddffnet=gammaDD*a1dfnet**2 # net DD fractionation factor
    a1ddffp=a1ddffnet/(p/2+1/2)
    a1ddffs=p*a1ddffp 

    # Calculate the fractionation factors of the back reactions for mcr
    a1dfbp=a1dffp*a1dfeqp # primary isotope effect backwards
    a1dfbs=a1dffs*a1dfeqs # secondary isotope effect backwards
    a1cdfbp=a1cdffp*a1cdfeqp
    a1cdfbs=a1cdffs*a1cdfeqs
    a1ddfbp=a1ddffp*a1ddfeqp
    a1ddfbs=a1ddffs*a1ddfeqs
    print("Net fractionation factors of the first step (forward):")
    print(a1cff, "\n", a1dfnet, "\n", a1cdffnet, "\n", a1ddffnet)
# Ab initio calculation using the DFT model from Wognate et al., 50 degree C
if ff=="ab initio": 
    if model=="no INT": # Not considering the equilibrium isotope effect between methane and INT
        a1cff=0.9387
        a1dffp=0.5416
        a1dffs=0.8432
        a1cdffp=0.5058
        a1cdffs=0.7919
        a1ddffp=0.4491
        a1ddffs=0.7069
    if model=="INT":
        a1cff=0.9387
        a1dffp=0.5416
        a1dffs=0.8483
        a1cdffp=0.5058 # 0.3459 for experimental observations
        a1cdffs=0.7967 # 0.8322
        a1ddffp=0.4554 # 0.3102
        a1ddffs=0.7208 # 0.7600
    # Calculate the fractionation factors of the backward reactions for mcr
    a1cfb=a1cff*a1cfeq
    a1dfbp=a1dffp*a1dfeqp # primary isotope effect backwards
    a1dfbs=a1dffs*a1dfeqs # secondary isotope effect backwards
    a1cdfbp=a1cdffp*a1cdfeqp
    a1cdfbs=a1cdffs*a1cdfeqs
    a1ddfbp=a1ddffp*a1ddfeqp
    a1ddfbs=a1ddffs*a1ddfeqs
    # Calculate net ff
    a1dfnet=a1dffp/4+3/4*a1dffs
    a1cdffnet=a1cdffp/4+3/4*a1cdffs
    a1ddffnet=a1ddffp/2+a1ddffs/2
    print("Net fractionation factors:")
    print(a1cff, "\n", a1dfnet, "\n", a1cdffnet, "\n", a1ddffnet)

# Isotope fractionation factors for the two downstream reversible steps.
# Step 2: CH3-SCoM <-> CHO-MFR
# Equilibrium fractionation factors (forward-direction convention).
a2ceq=0.9540 # 13C EIE, CH3-SCoM -> CHO-MFR
a2deqp=0.8748 # primary D EIE
a2deqs=0.9764 # secondary D EIE

# Forward kinetic fractionation factors (CH3-SCoM -> CHO-MFR).
a2cff=0.979 # 13C effect
a2dffp=0.888 # primary D effect: D is removed from the methyl group
a2dffs=1.000 # secondary D effect: D is retained in the formyl group
a2dffnet=(2.0/3.0)*a2dffp+(1.0/3.0)*a2dffs # diagnostic only

# Clumped-isotopologue gamma factors.
# p = primary D position; s = secondary D position.
# pp = two primary D positions; ps = one primary and one secondary D.
gammaCDff2p=0.9749
gammaCDff2s=1.0
gammaDDff2pp=0.9256
gammaDDff2ps=0.9255
gammaCDfb2p=1.0
gammaCDfb2s=1.0
gammaDDfb2pp=1.0
gammaDDfb2ps=1.0

# Pathway-specific forward clumped fractionation factors.
a2cdffp=gammaCDff2p*a2cff*a2dffp
a2cdffs=gammaCDff2s*a2cff*a2dffs
a2ddffpp=gammaDDff2pp*a2dffp**2
a2ddffps=gammaDDff2ps*a2dffp*a2dffs

# Backward KIEs calculated from KIE_backward = KIE_forward * EIE.
a2cfb=a2cff*a2ceq
a2dfbp=a2dffp*a2deqp
a2dfbs=a2dffs*a2deqs
a2dfbnet=(2.0/3.0)*a2dfbp+(1.0/3.0)*a2dfbs # diagnostic only

# Pathway-specific backward clumped fractionation factors.
a2cdfbp=gammaCDfb2p*a2cfb*a2dfbp
a2cdfbs=gammaCDfb2s*a2cfb*a2dfbs
a2ddfbpp=gammaDDfb2pp*a2dfbp**2
a2ddfbps=gammaDDfb2ps*a2dfbp*a2dfbs

# Step 3: CHO-MFR <-> CO2
a3ceq=0.9825 #1/np.exp(17.5/1000)
a3deq=0.8959 #1/np.exp(153.2/1000)
a3cff=0.985 # 13C effect, CHO-MFR -> CO2
a3dff=0.709 # D effect, CHO-MFR -> CO2
gammaCDff3=1.0
gammaCDfb3=1.0
a3cdff=gammaCDff3*a3cff*a3dff # 13C-D clumped effect, CHO-MFR -> CO2
a3cfb=a3ceq*a3cff # 13C effect, CO2 -> CHO-MFR
a3dfb=a3deq*a3dff # D effect, CO2 -> CHO-MFR
a3cdfb=gammaCDfb3*a3cfb*a3dfb # 13C-D clumped effect, CO2 -> CHO-MFR

# The hydrogen of the first step is assumed to be in equilibrium with HS-COB, equilibrium fractionation from Wegener et al.
ahscobeq=0.4686 # 1/np.exp(831.1/1000) # aHSCOB-H2O=R_H2O/R_HSCOB HSCOB-->H2O, equilibrium value
ahscobf=1.0 # Forward fractionation HS-CoB --> H2O, best-fit value in Wegener et al.
ahscobb=ahscobf*ahscobeq
RHSCOB=RH2O/ahscobeq # D/H ratio in HS-CoB, assuming equilibrium with water
rev_hscob=0.99

# F420H2 and water is also assumed to be in equilibrium
af420h2eq=1.0/np.exp(121.4/1000)
if F420==False: 
      af420h2eq=1.0 # Toggle the use of F420 or not  
RF420H2=RH2O/af420h2eq
XF420H2=1/(1+RF420H2)
XF420D=1-XF420H2


# reversibility of cross-membrane transport, assuming highly reversibile methane exchange inside and outside the cells, without any isotope fractionation.
rev_tr=0.99

# Set up initial conditions
# Abundance of all relevant methane isotopologues 12CH4, 13CH4, 12CH3D, 13CH3D, 12CH2D2
abundance=[
    9.89089799e-01, 
    1.05053242e-02, 
    4.00577619e-04, 
    4.24165463e-06,
    5.78643181e-08
]

# Typical freshwater methanogenesis
# abundance=[
#     9.89214661e-01, 
#     1.03960311e-02, 
#     3.85195116e-04, 
#     4.05835145e-06,
#     5.43142016e-08
# ]
# The abundance of hydrogenotrophic methanogenesis by M barkeri
# abundance=[
#     9.89445854e-01, 
#     1.02657303e-02, 
#     2.85430170e-04, 
#     2.95541119e-06,
#     3.05685049e-08
# ]

# Species index map:
# 0:5   intracellular CH4 isotopologues: 12CH4, 13CH4, 12CH3D, 13CH3D, 12CH2D2
# 5:10  CH3-SCoM isotopologues:         12CH3, 13CH3, 12CH2D, 13CH2D, 12CHD2
# 10:12 HS-CoB isotopologues:           H, D
# 12:17 extracellular CH4 isotopologues, same order as 0:5
# 17:21 CHO-MFR isotopologues:          12CHO, 13CHO, 12CDO, 13CDO
# 21:23 CO2 isotopologues:              12CO2, 13CO2
R=np.zeros(23) # abundance of each species involved in the reaction network

# Normalize the abundance, assuming the chemicals inside the cell (CH4,CH3-SCoM, HS-Cob) has a total abundance of 1 for each.
for i in range(5):    
    R[i]=abundance[i]/sum(abundance)
# Abundance of all relevant CH3-SCoM isotopologues 12CH3, 13CH3, 12CH2D, 13CH2D, 12CHD2
# Assume the same abundance as CH4
R[5:10]=R[0:5]
# D and H are in equilibrium with water
R[10]=1.0/(1.0+RHSCOB) # H
R[11]=(1-R[10]) # D
R[12:17]=R[0:5]*50 # Methane isotopologue abundances outside, assuming the reservoir is 10 times larger

# Initialize CHO-MFR and CO2 with small, isotopically consistent pools so that
# reversible downstream fluxes can be partitioned from t=0. This mirrors the
# original model's initialization of CH3-SCoM rather than starting the
# intermediate/product pools exactly at zero.
XH2O_init=1.0-FH2O
XD2O_init=FH2O
R[17]=(R[0]+R[2]+R[4])*XH2O_init # 12CHO-MFR
R[18]=(R[1]+R[3])*XH2O_init      # 13CHO-MFR
R[19]=(R[0]+R[2]+R[4])*XD2O_init # 12CDO-MFR
R[20]=(R[1]+R[3])*XD2O_init      # 13CDO-MFR
R[21]=1-F13C_DIC             # 12CO2
R[22]=F13C_DIC                  # 13CO2
R0=R.copy()
# Construct ode
def dfdt(t,R,k,rev2,rev3):
    k1f,k1b=k     # unpack first-step rate constants; rev2 and rev3 are passed separately
    # Calculate total forward and backward fluxes for the first step.
    Jf = (k1f*R[0] + k1f*R[1]*a1cff
          + (1/4*k1f*R[2]*a1dffp + 3/4*k1f*R[2]*a1dffs)
          + (1/4*k1f*R[3]*a1cdffp + 3/4*k1f*R[3]*a1cdffs)
          + (1/2*k1f*R[4]*a1ddffp + 1/2*k1f*R[4]*a1ddffs))

    Jb = (k1b*R[5]*R[10] + k1b*R[6]*a1cfb*R[10]
          + (k1b*R[5]*R[11]*a1dfbp + k1b*R[7]*R[10]*a1dfbs)
          + (k1b*R[6]*R[11]*a1cdfbp + k1b*R[8]*R[10]*a1cdfbs)
          + (k1b*R[7]*R[11]*a1ddfbp + k1b*R[9]*R[10]*a1ddfbs))

    # Apply steady state for HS-CoB species
    fhscobf=R[10]+R[11]*ahscobf
    fhscobb=(1-FH2O)+FH2O*ahscobb
    Jfhscob=(Jf-Jb)/(1-rev_hscob)
    Jbhscob=(Jf-Jb)*rev_hscob/(1-rev_hscob)

    dRdt=np.zeros_like(R)
    Jnet = Jf - Jb
    if rev2 >= 1.0 or rev3 >= 1.0:
        raise ValueError("rev2 and rev3 must be < 1.0 because Jf=Jnet/(1-rev).")

    # ------------------------------------------------------------------
    # Step 2: CH3-SCoM <-> CHO-MFR with constant flux reversibility
    # J2b/J2f = rev2 and J2f - J2b = Jnet.
    # ------------------------------------------------------------------
    J2f = Jnet/(1.0-rev2)
    J2b = Jnet*rev2/(1.0-rev2)
    # Forward CH3-SCoM -> CHO-MFR partition with explicit primary/secondary
    # H-isotope effects. If D is removed from CH3-SCoM, the primary effect
    # applies; if D is retained as the formyl D in CDO-MFR, the secondary
    # effect applies. Statistical coefficients count the possible H/D sites.
    f2_12CH3_to_12CHO = R[5]
    f2_13CH3_to_13CHO = R[6]*a2cff

    f2_12CH2D_to_12CHO = R[7]*(2.0/3.0)*a2dffp
    f2_12CH2D_to_12CDO = R[7]*(1.0/3.0)*a2dffs
    f2_13CH2D_to_13CHO = R[8]*(2.0/3.0)*a2cdffp
    f2_13CH2D_to_13CDO = R[8]*(1.0/3.0)*a2cdffs

    # For CHD2-SCoM, formation of CHO removes both D atoms (primary-primary),
    # whereas formation of CDO removes one D and retains one D
    # (primary-secondary).
    f2_12CHD2_to_12CHO = R[9]*(1.0/3.0)*a2ddffpp
    f2_12CHD2_to_12CDO = R[9]*(2.0/3.0)*a2ddffps

    fch3f = (f2_12CH3_to_12CHO + f2_13CH3_to_13CHO
             + f2_12CH2D_to_12CHO + f2_12CH2D_to_12CDO
             + f2_13CH2D_to_13CHO + f2_13CH2D_to_13CDO
             + f2_12CHD2_to_12CHO + f2_12CHD2_to_12CDO)

    if fch3f <= 0.0:
        F2_12CH3 = F2_13CH3 = F2_12CH2D = F2_13CH2D = F2_12CHD2 = 0.0
        F2_12CHO = F2_13CHO = F2_12CDO = F2_13CDO = 0.0
    else:
        j2_12CH3_to_12CHO = J2f*f2_12CH3_to_12CHO/fch3f
        j2_13CH3_to_13CHO = J2f*f2_13CH3_to_13CHO/fch3f
        j2_12CH2D_to_12CHO = J2f*f2_12CH2D_to_12CHO/fch3f
        j2_12CH2D_to_12CDO = J2f*f2_12CH2D_to_12CDO/fch3f
        j2_13CH2D_to_13CHO = J2f*f2_13CH2D_to_13CHO/fch3f
        j2_13CH2D_to_13CDO = J2f*f2_13CH2D_to_13CDO/fch3f
        j2_12CHD2_to_12CHO = J2f*f2_12CHD2_to_12CHO/fch3f
        j2_12CHD2_to_12CDO = J2f*f2_12CHD2_to_12CDO/fch3f

        # Depletion of CH3-SCoM isotopologue pools.
        F2_12CH3 = j2_12CH3_to_12CHO
        F2_13CH3 = j2_13CH3_to_13CHO
        F2_12CH2D = j2_12CH2D_to_12CHO + j2_12CH2D_to_12CDO
        F2_13CH2D = j2_13CH2D_to_13CHO + j2_13CH2D_to_13CDO
        F2_12CHD2 = j2_12CHD2_to_12CHO + j2_12CHD2_to_12CDO

        # Production of CHO-MFR isotopologue pools.
        F2_12CHO = j2_12CH3_to_12CHO + j2_12CH2D_to_12CHO + j2_12CHD2_to_12CHO
        F2_13CHO = j2_13CH3_to_13CHO + j2_13CH2D_to_13CHO
        F2_12CDO = j2_12CH2D_to_12CDO + j2_12CHD2_to_12CDO
        F2_13CDO = j2_13CH2D_to_13CDO

    # Backward CHO-MFR -> CH3-SCoM partition. The formyl H/D is retained,
    # while two additional methyl H/D atoms are supplied by F420H2. D added
    # from F420H2 carries the backward primary effect; formyl D carries the
    # backward secondary effect.
    b2_12CHO_to_12CH3 = R[17]*(XF420H2**2)
    b2_12CHO_to_12CH2D = R[17]*(2.0*XF420H2*XF420D)*a2dfbp
    b2_12CHO_to_12CHD2 = R[17]*(XF420D**2)*a2ddfbpp
    b2_12CDO_to_12CH2D = R[19]*(XF420H2**2)*a2dfbs
    b2_12CDO_to_12CHD2 = R[19]*(2.0*XF420H2*XF420D)*a2ddfbps

    b2_13CHO_to_13CH3 = R[18]*a2cfb*(XF420H2**2)
    b2_13CHO_to_13CH2D = R[18]*(2.0*XF420H2*XF420D)*a2cdfbp
    b2_13CDO_to_13CH2D = R[20]*(XF420H2**2)*a2cdfbs

    fcho_b = (b2_12CHO_to_12CH3 + b2_12CHO_to_12CH2D
              + b2_12CHO_to_12CHD2 + b2_12CDO_to_12CH2D
              + b2_12CDO_to_12CHD2 + b2_13CHO_to_13CH3
              + b2_13CHO_to_13CH2D + b2_13CDO_to_13CH2D)

    if J2b == 0.0 or fcho_b <= 0.0:
        B2_12CH3 = B2_12CH2D = B2_12CHD2 = B2_13CH3 = B2_13CH2D = 0.0
        B2_12CHO = B2_13CHO = B2_12CDO = B2_13CDO = 0.0
    else:
        j2b_12CHO_to_12CH3 = J2b*b2_12CHO_to_12CH3/fcho_b
        j2b_12CHO_to_12CH2D = J2b*b2_12CHO_to_12CH2D/fcho_b
        j2b_12CHO_to_12CHD2 = J2b*b2_12CHO_to_12CHD2/fcho_b
        j2b_12CDO_to_12CH2D = J2b*b2_12CDO_to_12CH2D/fcho_b
        j2b_12CDO_to_12CHD2 = J2b*b2_12CDO_to_12CHD2/fcho_b
        j2b_13CHO_to_13CH3 = J2b*b2_13CHO_to_13CH3/fcho_b
        j2b_13CHO_to_13CH2D = J2b*b2_13CHO_to_13CH2D/fcho_b
        j2b_13CDO_to_13CH2D = J2b*b2_13CDO_to_13CH2D/fcho_b

        # Production of CH3-SCoM isotopologue pools.
        B2_12CH3 = j2b_12CHO_to_12CH3
        B2_12CH2D = j2b_12CHO_to_12CH2D + j2b_12CDO_to_12CH2D
        B2_12CHD2 = j2b_12CHO_to_12CHD2 + j2b_12CDO_to_12CHD2
        B2_13CH3 = j2b_13CHO_to_13CH3
        B2_13CH2D = j2b_13CHO_to_13CH2D + j2b_13CDO_to_13CH2D

        # Depletion of CHO-MFR isotopologue pools.
        B2_12CHO = (j2b_12CHO_to_12CH3 + j2b_12CHO_to_12CH2D
                    + j2b_12CHO_to_12CHD2)
        B2_12CDO = j2b_12CDO_to_12CH2D + j2b_12CDO_to_12CHD2
        B2_13CHO = j2b_13CHO_to_13CH3 + j2b_13CHO_to_13CH2D
        B2_13CDO = j2b_13CDO_to_13CH2D
    
    # ------------------------------------------------------------------
    # Step 3: CHO-MFR <-> CO2 with constant flux reversibility
    # J3b/J3f = rev3 and J3f - J3b = Jnet.
    # ------------------------------------------------------------------
    J3f = Jnet/(1.0-rev3)
    J3b = Jnet*rev3/(1.0-rev3)
    XH2O = 1.0-FH2O
    XD2O = FH2O
    # Forward CHO-MFR -> CO2 partition. CO2 retains carbon but not H/D.
    fcho_f = R[17] + R[18]*a3cff + R[19]*a3dff + R[20]*a3cdff
    if fcho_f <= 0.0:
        F3_12CHO = F3_13CHO = F3_12CDO = F3_13CDO = 0.0
    else:
        F3_12CHO = J3f*R[17]/fcho_f
        F3_13CHO = J3f*R[18]*a3cff/fcho_f
        F3_12CDO = J3f*R[19]*a3dff/fcho_f
        F3_13CDO = J3f*R[20]*a3cdff/fcho_f
    F3_12CO2 = F3_12CHO + F3_12CDO
    F3_13CO2 = F3_13CHO + F3_13CDO

    # Backward CO2 -> CHO-MFR partition. Carbon comes from CO2; formyl H/D
    # comes from water.
    b3_12CHO_rate = (1-F13C_DIC)*XH2O
    b3_12CDO_rate = (1-F13C_DIC)*XD2O*a3dfb
    b3_13CHO_rate = F13C_DIC*a3cfb*XH2O
    b3_13CDO_rate = F13C_DIC*XD2O*a3cdfb
    fco2_b = b3_12CHO_rate + b3_12CDO_rate + b3_13CHO_rate + b3_13CDO_rate
    if J3b == 0.0 or fco2_b <= 0.0:
        B3_12CHO = B3_12CDO = B3_13CHO = B3_13CDO = 0.0
    else:
        B3_12CHO = J3b*b3_12CHO_rate/fco2_b
        B3_12CDO = J3b*b3_12CDO_rate/fco2_b
        B3_13CHO = J3b*b3_13CHO_rate/fco2_b
        B3_13CDO = J3b*b3_13CDO_rate/fco2_b
    B3_12CO2 = B3_12CHO + B3_12CDO
    B3_13CO2 = B3_13CHO + B3_13CDO

    Jin=(Jf-Jb)/(1-rev_tr)
    Jout=(Jf-Jb)*rev_tr/(1-rev_tr)
    if os==True:
        Jadvin=Jnet/phi # Advection rate for in and out in open system
        Jadvout=Jadvin-Jnet
    if os==False:
        Jadvin=0.0
        Jadvout=0.0
    fch4in=sum(R[12:17]) # Total methane abundance outside
    fch4out=sum(R[0:5]) # Total methane abundance inside
    dRdt[0]=Jin*R[12]/fch4in-Jout*R[0]/fch4out-k1f*R[0]+k1b*R[5]*R[10] # 12CH4
    dRdt[1]=Jin*R[13]/fch4in-Jout*R[1]/fch4out-k1f*R[1]*a1cff+k1b*R[6]*a1cfb*R[10] # 13CH4
    dRdt[2]=Jin*R[14]/fch4in-Jout*R[2]/fch4out-1/4*k1f*R[2]*a1dffp-3/4*k1f*R[2]*a1dffs+k1b*R[5]*R[11]*a1dfbp+k1b*R[7]*R[10]*a1dfbs # 12CH3D
    dRdt[3]=Jin*R[15]/fch4in-Jout*R[3]/fch4out-1/4*k1f*R[3]*a1cdffp-3/4*k1f*R[3]*a1cdffs+k1b*R[6]*R[11]*a1cdfbp+k1b*R[8]*R[10]*a1cdfbs # 13CH3D
    dRdt[4]=Jin*R[16]/fch4in-Jout*R[4]/fch4out-1/2*k1f*R[4]*a1ddffp-1/2*k1f*R[4]*a1ddffs+k1b*R[7]*R[11]*a1ddfbp+k1b*R[9]*R[10]*a1ddfbs #12CH2D2
    dRdt[5]=k1f*R[0]+1/4*k1f*R[2]*a1dffp-k1b*R[5]*R[11]*a1dfbp-k1b*R[5]*R[10]-F2_12CH3+B2_12CH3 # 12CH3
    dRdt[6]=k1f*R[1]*a1cff+1/4*k1f*R[3]*a1cdffp-k1b*R[6]*R[11]*a1cdfbp-k1b*R[6]*R[10]*a1cfb-F2_13CH3+B2_13CH3 # 13CH3
    dRdt[7]=3/4*k1f*R[2]*a1dffs+1/2*k1f*R[4]*a1ddffp-k1b*R[7]*R[10]*a1dfbs-k1b*R[7]*R[11]*a1ddfbp-F2_12CH2D+B2_12CH2D # 12CH2D
    dRdt[8]=3/4*k1f*R[3]*a1cdffs-k1b*R[8]*R[10]*a1cdfbs-F2_13CH2D+B2_13CH2D # 13CH2D
    dRdt[9]=1/2*k1f*R[4]*a1ddffs-k1b*R[9]*R[10]*a1ddfbs-F2_12CHD2+B2_12CHD2 # 12CHD2
    dRdt[10]=(k1f*R[0]+k1f*R[1]*a1cff
    +3/4*k1f*R[2]*a1dffs+3/4*k1f*R[3]*a1cdffs+1/2*k1f*R[4]*a1ddffs
    -k1b*R[5]*R[10]-k1b*R[6]*R[10]*a1cfb
    -k1b*R[7]*R[10]*a1dfbs-k1b*R[8]*R[10]*a1cdfbs-k1b*R[9]*R[10]*a1ddfbs
    -Jfhscob*R[10]/fhscobf+Jbhscob*(1-FH2O)/fhscobb) # HS-CoB sink to H2O
    dRdt[11]=(1/4*k1f*R[2]*a1dffp+1/4*k1f*R[3]*a1cdffp+1/2*k1f*R[4]*a1ddffp
    -k1b*R[5]*R[11]*a1dfbp-k1b*R[6]*R[11]*a1cdfbp-k1b*R[7]*R[11]*a1ddfbp
    -Jfhscob*R[11]*ahscobf/fhscobf+Jbhscob*FH2O*ahscobb/fhscobb)
    dRdt[12]=-Jin*R[12]/fch4in+Jout*R[0]/fch4out+Jadvin*abundance[0]/sum(abundance)-Jadvout*R[12]/fch4in
    dRdt[13]=-Jin*R[13]/fch4in+Jout*R[1]/fch4out+Jadvin*abundance[1]/sum(abundance)-Jadvout*R[13]/fch4in
    dRdt[14]=-Jin*R[14]/fch4in+Jout*R[2]/fch4out+Jadvin*abundance[2]/sum(abundance)-Jadvout*R[14]/fch4in
    dRdt[15]=-Jin*R[15]/fch4in+Jout*R[3]/fch4out+Jadvin*abundance[3]/sum(abundance)-Jadvout*R[15]/fch4in
    dRdt[16]=-Jin*R[16]/fch4in+Jout*R[4]/fch4out+Jadvin*abundance[4]/sum(abundance)-Jadvout*R[16]/fch4in
    dRdt[17]=F2_12CHO-B2_12CHO-F3_12CHO+B3_12CHO # 12CHO-MFR
    dRdt[18]=F2_13CHO-B2_13CHO-F3_13CHO+B3_13CHO # 13CHO-MFR
    dRdt[19]=F2_12CDO-B2_12CDO-F3_12CDO+B3_12CDO # 12CDO-MFR
    dRdt[20]=F2_13CDO-B2_13CDO-F3_13CDO+B3_13CDO # 13CDO-MFR
    dRdt[21]=F3_12CO2-B3_12CO2 # 12CO2
    dRdt[22]=F3_13CO2-B3_13CO2 # 13CO2
    return dRdt

# Define a separate function to calculate the evolution of reversibility
def jb_jf_ratio(t, R, k):
    k1f, k1b = k

    Jf = (k1f*R[0] + k1f*R[1]*a1cff
          + (1/4*k1f*R[2]*a1dffp + 3/4*k1f*R[2]*a1dffs)
          + (1/4*k1f*R[3]*a1cdffp + 3/4*k1f*R[3]*a1cdffs)
          + (1/2*k1f*R[4]*a1ddffp + 1/2*k1f*R[4]*a1ddffs))

    Jb = (k1b*R[5]*R[10] + k1b*R[6]*a1cfb*R[10]
          + (k1b*R[5]*R[11]*a1dfbp + k1b*R[7]*R[10]*a1dfbs)
          + (k1b*R[6]*R[11]*a1cdfbp + k1b*R[8]*R[10]*a1cdfbs)
          + (k1b*R[7]*R[11]*a1ddfbp + k1b*R[9]*R[10]*a1ddfbs))

    return Jb / Jf

# Define separate functions to calculate the second- and third-step flux reversibility.
# These are constant by construction and should equal rev2/rev3 at every time point.
def j2b_j2f_ratio(t, R, rev2):
    return rev2

def j3b_j3f_ratio(t, R, rev3):
    return rev3

# Process the data
def process(sol):
    tot=sum(sol.y[12:17]) # Total isotopologue abundance
    tot0=sum(R0[12:17]) # initial isotopologue abundance
    fCH4=tot/tot0 # Fraction of residual methane
    d13C_t=(sol.y[13]/sol.y[12]/RVPDB-1)*1000
    dD_t=(sol.y[14]/(4*sol.y[12])/RVSMOW-1)*1000
    # Calculate the stochastic distributions
    X13C=(sol.y[13]+sol.y[15])/tot # X13C 
    XD=(sol.y[14]+sol.y[15]+2*sol.y[16])/(4*tot) # XD
    X12C=1-X13C
    XH=1-XD
    sto_13CD=4*X13C*XD*XH**3/(X12C*XH**4)
    sto_DD=6*X12C*XD**2*XH**2/(X12C*XH**4)
    D13CH3D_t=(sol.y[15]/sol.y[12]/sto_13CD-1)*1000
    D12CH2D2_t=(sol.y[16]/sol.y[12]/sto_DD-1)*1000
    return tot,fCH4,d13C_t,dD_t,D13CH3D_t,D12CH2D2_t

# Define a function to model the change of isotope fractionation with reversibility
def model_rev(rev1, rev2, rev3, tmax, num):
    k1f_input=1.0
    k1b_input=k1f_input*rev1
    k=[k1f_input,k1b_input]
    tint=np.linspace(t_lower,tmax,num) 
    solution=solve_ivp(dfdt,(t_lower,tmax),R0,args=(k,rev2,rev3),t_eval=tint,atol=1.0e-12,rtol=1.0e-9)  # use method='BDF' if kex is very large
    tot_sol,fCH4_sol,d13C_sol,dD_sol,D13CH3D_sol,D12CH2D2_sol=process(solution)
    rev1_sol = np.array([jb_jf_ratio(t, R, k) for t, R in zip(solution.t, solution.y.T)])
    rev2_sol = np.array([j2b_j2f_ratio(t, R, rev2) for t, R in zip(solution.t, solution.y.T)])
    rev3_sol = np.array([j3b_j3f_ratio(t, R, rev3) for t, R in zip(solution.t, solution.y.T)])
    return tot_sol,fCH4_sol,rev1_sol,rev2_sol,rev3_sol,d13C_sol,dD_sol,D13CH3D_sol,D12CH2D2_sol

if len(rev1_list) != len(rev2_list) or len(rev1_list) != len(rev3_list):
    raise ValueError("rev1_list, rev2_list, and rev3_list must have the same length for pairwise plotting.")
if len(time_list) != len(rev1_list):
    raise ValueError("time_list must have the same length as rev1_list/rev2_list/rev3_list.")

n_models=len(rev1_list)
tot=np.zeros([n_models,num])
fCH4=np.zeros([n_models,num])
rev1=np.zeros([n_models,num])
rev2=np.zeros([n_models,num])
rev3=np.zeros([n_models,num])
d13C=np.zeros([n_models,num])
dD=np.zeros([n_models,num])
D13CH3D=np.zeros([n_models,num])
D12CH2D2=np.zeros([n_models,num])
for i in range(n_models):
    tot[i,:],fCH4[i,:],rev1[i,:],rev2[i,:],rev3[i,:],d13C[i,:],dD[i,:],D13CH3D[i,:],D12CH2D2[i,:]=model_rev(rev1_list[i],rev2_list[i],rev3_list[i],time_list[i],num)

# AeOM fractionation by sMMO
a13C_sMMO=0.9872
aD_sMMO=0.7292
a13CD_sMMO=0.7196
aDD_sMMO=0.4877
# Experimental results of MCR exchange by Scheller et al.
a13C_mcr_exp=1/1.039/a1cfeq
aD_mcr_exp=1/2.44/4+1/1.17*3/4
# Calculate the fractionation trajectory
f=np.linspace(0.3,1,20)
normC_sMMO=(a13C_sMMO-1)*(np.log(f))
normD_sMMO=(aD_sMMO-1)*(np.log(f))
normCD_sMMO=1000*(a13CD_sMMO-a13C_sMMO-aD_sMMO+1)*np.log(f)
normDD_sMMO=1000*(aDD_sMMO-2*aD_sMMO+1)*np.log(f)
dC_sMMO=f**(a13C_sMMO-1)*(-41.108+1000)-1000
dD_sMMO=f**(aD_sMMO-1)*(-169.120+1000)-1000
CD_sMMO=normCD_sMMO+3.131
DD_sMMO=normDD_sMMO+7.038
normC_mcr_exp=(a13C_mcr_exp-1)*np.log(f)
normD_mcr_exp=(aD_mcr_exp-1)*np.log(f)
dC_mcr_exp=f**(a13C_mcr_exp-1)*(-41.108+1000)-1000
dD_mcr_exp=f**(aD_mcr_exp-1)*(-169.12+1000)-1000

# Plotting
# Import data
data = pd.read_csv('isotope_data.csv')
equib = pd.read_csv('equib.csv')
ANME2d=data[data['label']=='3'] # ANME-2d
NC10=data[data['label']=='1'] # NC10
# Previous AOM data
AOM_P=data[data['label']=='AOM_P']
AOM_P_LS=data[data['label']=='AOM_P_LS']
AOM_ono=data[data['label']=='AOM_ono']
AOM_wegener=data[data['label']=='AOM_Wegener']
AOM_wegener_LS=data[data['label']=='AOM_Wegener_LS']
Oct=data[data['label']=='4']
# Previous AeOM
AeOM_P=data[data['label']=='P'] # Previous data from Li et al., 2024
AeOM_P1=data[data['label']=='P1'] # Previous data from Krause et al., 2022

# Plot
# Font dictionary
font = {'family': 'sans serif',
        'color':  'black',
        'weight': 'normal',
        'size': 36,
        }
font_labels = {'family': 'sans serif',
               'color':  'black',
               'weight': 'normal',
               'size': 36,
                }
font_ticks = {'family': 'sans serif',
               'color':  'black',
               'weight': 'normal',
               'size': 32,
                }
def set_axis(ax,xlabel,ylabel):
    ax.set_ylabel(ylabel, fontdict = font_labels)
    ax.set_xlabel(xlabel, fontdict = font_labels)
    ax.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
    ax.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

def normalize_dat(data): # Get rid of the influence of T0 values
    norm=np.zeros([len(time_list),data.shape[1]])
    for i in range(norm.shape[0]):
        norm[i,:]=np.log((data[i,:]+1000)/(data[i,0]+1000))
    return norm

def normalize_clumped_dat(data): # Get rid of the influence of T0 values
    norm=np.zeros([len(time_list),data.shape[1]])
    for i in range(norm.shape[0]):
        norm[i,:]=data[i,:]-data[i,0]
    return norm

def make_label():
    labellist={}
    for i in range(len(rev1_list)):
        labellist[str(i)]=r"R$_1$="+str(rev1_list[i])+r", R$_2$="+str(rev2_list[i])+r", R$_3$="+str(rev3_list[i])
    return labellist

Data_to_fit1=[-62.75,-336.78,16.48,-26.06]
Data_to_fit2=[-63.05,-336.94,6.01,-26.08]

fig_bulk,ax_bulk=plt.subplots(figsize=(12,12))
ax_bulk.scatter(Data_to_fit1[0],Data_to_fit1[1], s=80)
ax_bulk.scatter(Data_to_fit2[0],Data_to_fit2[1], s=80)
ax_bulk.plot(d13C[0,:],dD[0,:], linewidth=2.5, color="black", linestyle="-.", alpha=1.0, label="Model, R1="+str(rev1_list[0])+", R2="+str(rev2_list[0])+", R3="+str(rev3_list[0]),zorder=1)
ax_bulk.plot(d13C[1,:],dD[1,:], linewidth=2.5, linestyle=":", color="purple", alpha=1.0,label="Model, R1="+str(rev1_list[1])+", R2="+str(rev2_list[1])+", R3="+str(rev3_list[1]),zorder=1)
ax_bulk.plot(d13C[2,:],dD[2,:], linewidth=2.5, linestyle="--", color="blue", alpha=1.0,label="Model, R1="+str(rev1_list[2])+", R2="+str(rev2_list[2])+", R3="+str(rev3_list[2]),zorder=1)
ax_bulk.plot(d13C[3,:],dD[3,:], linewidth=2.5, linestyle=(5,(10,3)), color="orange", alpha=1.0, label="Model, R1="+str(rev1_list[3])+", R2="+str(rev2_list[3])+", R3="+str(rev3_list[3]), zorder=1)
ax_bulk.plot(d13C[4,:],dD[4,:], linewidth=2.5, color="red", alpha=1.0, label="Model, R1="+str(rev1_list[4])+", R2="+str(rev2_list[4])+", R3="+str(rev3_list[4]), zorder=1)

set_axis(ax_bulk,'$\delta^{13}$C (\u2030)','$\delta$D (\u2030)')
ax_bulk.legend(fontsize=15)

fig_clump,ax_clump=plt.subplots(figsize=(12,12))
ax_clump.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax_clump.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)

ax_clump.scatter(Data_to_fit1[2],Data_to_fit1[3], s=80)
ax_clump.scatter(Data_to_fit2[2],Data_to_fit2[3], s=80)
ax_clump.plot(D13CH3D[0,:],D12CH2D2[0,:], linewidth=2.5, color="black",linestyle="-.",alpha=1.0,zorder=1)
ax_clump.plot(D13CH3D[1,:],D12CH2D2[1,:], linewidth=2.5, linestyle=":", color="purple", alpha=1.0,zorder=1)
ax_clump.plot(D13CH3D[2,:],D12CH2D2[2,:], linewidth=2.5, linestyle="--", color="blue", alpha=1.0,zorder=1)
ax_clump.plot(D13CH3D[3,:],D12CH2D2[3,:], linewidth=2.5, linestyle=(5,(10,3)),color="orange", alpha=1.0,zorder=1)
ax_clump.plot(D13CH3D[4,:],D12CH2D2[4,:], linewidth=2.5, color="red", alpha=1.0,zorder=1)

plt.show()
