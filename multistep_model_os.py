# This is a model to model the reversibility of AOM
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

output=True # True if you want to save the plots as pdf
ff="ab initio" # Select the KIEs in the model: 
model="no INT"
rev1_list=[0.8,0.8,0.9,0.7,0.5] # First-step reversibility list, CH4 <-> CH3-SCoM; each value ranges from 0 to <1
rev2_list=[0.8,0.8,0.9,0.5,0.5] # Second-step reversibility list, CH3-SCoM <-> CHO-MFR; each value ranges from 0 to <1
rev3_list=[0.5,0.8,0.9,0.1,0.5] # Third-step reversibility list, CHO-MFR <-> CO2; each value ranges from 0 to <1
t_lower=0.000 # minimum time for time interval
time_list=[5000.0,50.0,100.0,200.0,500.0] # Maximum time for time interval, relevant to the final fraction of methane left
num=500000 # Number of tim steps
# "experiment": data from Scheller et al., 2013;
# "Ab initio": ab initio calculation in this study
dDH2O = -50.0 # permil dD_H2O
RVPDB = 0.0112372 # Standard carbon isotope ratio (VPDB)
RVSMOW = 1.5576e-4 # Standard hydrogen isotope ratio (VSMOW)
RH2O=RVSMOW*(dDH2O/1000+1) # D/H ratio in water
FH2O=RH2O/(1+RH2O) # D/(D+H) ratio in water
# For open system, define a phi value, phi=Jnet/Jadvin (net oxidation rate/advection in rate)  
phi=0.15
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Equlibrium isotope effect of the first step, from Gropp et al., 2021 (25 degree C)
a1cfeq=1/np.exp(2.1/1000)
a1dfeqp=1/np.exp(-635.8/1000) # Primary equilibrium fractionation
a1dfeqs=1/np.exp(55.4/1000) # Secondary equilibrium fractionation
a1cdfeqp=1/np.exp(-639.5/1000) # Primary
a1cdfeqs=1/np.exp(57.1/1000) # Secondary
a1ddfeqp=1/np.exp(-598.7/1000) # Primary
a1ddfeqs=1/np.exp(107.9/1000) # Secondary
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
# Ab initio calculation using the DFT model from Wognate et al.
if ff=="ab initio": 
    if model=="no INT": # Not considering the equilibrium isotope effect between methane and INT
        a1cff=0.9359
        a1dffp=0.5185
        a1dffs=0.8312
        a1cdffp=0.4825
        a1cdffs=0.7784
        a1ddffp=0.4227
        a1ddffs=0.6865
    if model=="INT":
        a1cff=0.9359
        a1dffp=0.5185
        a1dffs=0.8370
        a1cdffp=0.4825 # 0.3459 for experimental observations
        a1cdffs=0.7937 # 0.8322
        a1ddffp=0.4292 # 0.3102
        a1ddffs=0.7013 # 0.7600
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
# These are placeholders set to 1.0 for now; edit them later when you want
# to add step-specific kinetic/equilibrium isotope effects.
#
# Step 2: CH3-SCoM <-> CHO-MFR
# EFFs from Gropp et al., 2021, GCA
a2ceq=1/np.exp(18.1/1000)*1/np.exp(15.8/1000)*1/np.exp(16.9/1000)*1/np.exp(-3.3/1000)*1/np.exp(1.9/1000)
a2deq=1/np.exp(42.9/1000)*((1/np.exp(81.3/1000)+1/np.exp(84.0/1000))/2)*1/np.exp(-78.2/1000)*1/np.exp(-70.5/1000)*1/np.exp(8.5/1000)
# a2cdeqs=1/np.exp(61.0/1000)*(1/np.exp(98.8)+1/np.exp(96.0))/2*1/np.exp()
# KFFs from Wegener et al., 2022, Sci Adv
a2cff=0.979 # 13C effect, CH3-SCoM -> CHO-MFR
a2dff=0.999 # D effect, CH3-SCoM -> CHO-MFR
gammaCDff2=0.978
gammaDDff2=0.930
gammaCDfb2=1.0
gammaDDfb2=1.0
a2cdff=gammaCDff2*a2cff*a2dff # 13C-D clumped effect, CH3-SCoM -> CHO-MFR
a2ddff=gammaDDff2*a2dff**2 # D-D clumped effect, CH3-SCoM -> CHO-MFR
a2cfb=a2cff*a2ceq # 13C effect, CHO-MFR -> CH3-SCoM
a2dfb=a2dff*a2deq # D effect, CHO-MFR -> CH3-SCoM
a2cdfb=gammaCDfb2*a2cfb*a2dfb # 13C-D clumped effect, CHO-MFR -> CH3-SCoM
a2ddfb=gammaDDfb2*a2dfb**2 # D-D clumped effect, CHO-MFR -> CH3-SCoM
# Step 3: CHO-MFR <-> CO2
a3ceq=1/np.exp(17.5/1000)
a3deq=1/np.exp(153.2/1000)
a3cff=0.985 # 13C effect, CHO-MFR -> CO2
a3dff=0.709 # D effect, CHO-MFR -> CO2
gammaCDff3=1.0
gammaCDfb3=1.0
a3cdff=gammaCDff3*a3cff*a3dff # 13C-D clumped effect, CHO-MFR -> CO2
a3cfb=a3ceq*a3cff # 13C effect, CO2 -> CHO-MFR
a3dfb=a3deq*a3dff # D effect, CO2 -> CHO-MFR
a3cdfb=gammaCDfb3*a3cfb*a3dfb # 13C-D clumped effect, CO2 -> CHO-MFR

# The hydrogen of the first step is assumed to be in equilibrium with HS-COB, equilibrium fractionation from Wegener et al.
ahscobeq=1/np.exp(831.1/1000) # aHSCOB-H2O=R_H2O/R_HSCOB HSCOB-->H2O, equilibrium value
ahscobf=1.0 # Forward fractionation HS-CoB --> H2O, best-fit value in Wegener et al.
ahscobb=ahscobf*ahscobeq
RHSCOB=RH2O/ahscobeq # D/H ratio in HS-CoB, assuming equilibrium with water
rev_hscob=0.99

# F420H2 and water is also assumed to be in equilibrium
af420h2eq=1/np.exp(120.9/1000)
RF420H2=RH2O/af420h2eq
XF420H2=1/(1+RF420H2)
XF420D=1-XF420H2

# ---------------------------------------------------------------------------
# Direct H/D exchange between the methyl hydrogens of CH3-SCoM and water.
#
# alphaD_ch3scom_h2o is defined as
#     (D/H)_CH3-SCoM / (D/H)_H2O
# at isotopic equilibrium. Replace the placeholder value with the equilibrium
# fractionation factor appropriate for your temperature/model.
#
# kex_ch3scom_h2o is a pseudo-first-order exchange-rate scale in the same time
# units as k1f. It controls only how quickly equilibrium is approached; the
# forward/backward rate ratio controls the equilibrium composition. Set it to
# 0.0 to recover the original model exactly. Values much larger than k1f
# approximate rapid equilibration but may make the ODE stiff.
kex_ch3scom_h2o = 0.0
alphaD_ch3scom_h2o = 1.0/0.8748       # R_CH3-SCoM / R_H2O at equilibrium
gammaCD_ch3scom_h2o = 1.000      # 13C-D equilibrium clumping multiplier
gammaDD_ch3scom_h2o = 1.000      # multiplier for the second D substitution


# reversibility of cross-membrane transport, assuming highly reversibile methane exchange inside and outside the cells, without any isotope fractionation.
rev_tr=0.99

# Set up initial conditions
# Abundance of all relevant methane isotopologues 12CH4, 13CH4, 12CH3D, 13CH3D, 12CH2D2
# The abundance of tank gas
abundance=[
    0.9895430920,
    0.01006760015,
    3.85339203*10**-4,
    3.91444488*10**-6,
    5.42236786*10**-8
]

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
R[21]=R[0]+R[2]+R[4]             # 12CO2
R[22]=R[1]+R[3]                  # 13CO2
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

    # Forward CH3-SCoM -> CHO-MFR partition. Fractionation factors are all
    # initialized to 1.0 above. The H/D branching is statistical: CH2D has
    # 2/3 probability to make CHO and 1/3 to make CDO; CHD2 has 1/3 CHO
    # and 2/3 CDO.
    fch3f = (R[5] + R[6]*a2cff + R[7]*a2dff + R[8]*a2cdff + R[9]*a2ddff)
    if fch3f <= 0.0:
        F2_12CH3 = F2_13CH3 = F2_12CH2D = F2_13CH2D = F2_12CHD2 = 0.0
        F2_12CHO = F2_13CHO = F2_12CDO = F2_13CDO = 0.0
    else:
        F2_12CH3  = J2f*R[5]/fch3f
        F2_13CH3  = J2f*R[6]*a2cff/fch3f
        F2_12CH2D = J2f*R[7]*a2dff/fch3f
        F2_13CH2D = J2f*R[8]*a2cdff/fch3f
        F2_12CHD2 = J2f*R[9]*a2ddff/fch3f
        F2_12CHO = F2_12CH3 + (2.0/3.0)*F2_12CH2D + (1.0/3.0)*F2_12CHD2
        F2_13CHO = F2_13CH3 + (2.0/3.0)*F2_13CH2D
        F2_12CDO = (1.0/3.0)*F2_12CH2D + (2.0/3.0)*F2_12CHD2
        F2_13CDO = (1.0/3.0)*F2_13CH2D

    # Backward CHO-MFR -> CH3-SCoM partition. The formyl H/D is retained,
    # and the other two methyl H/D positions are supplied from F420H2.
    b2_12CH3_rate  = R[17]*(XF420H2**2)
    b2_12CH2D_rate = (R[17]*(2*XF420H2*XF420D)*a2dfb + R[19]*(XF420H2**2)*a2dfb)
    b2_12CHD2_rate = (R[17]*(XF420D**2)*a2ddfb + R[19]*(2*XF420H2*XF420D)*a2ddfb)
    b2_13CH3_rate  = R[18]*a2cfb*(XF420H2**2)
    b2_13CH2D_rate = (R[18]*(2*XF420H2*XF420D)*a2cdfb + R[20]*(XF420H2**2)*a2cdfb)
    fcho_b = (b2_12CH3_rate + b2_12CH2D_rate + b2_12CHD2_rate
              + b2_13CH3_rate + b2_13CH2D_rate)
    if J2b == 0.0 or fcho_b <= 0.0:
        B2_12CH3 = B2_12CH2D = B2_12CHD2 = B2_13CH3 = B2_13CH2D = 0.0
    else:
        B2_12CH3  = J2b*b2_12CH3_rate/fcho_b
        B2_12CH2D = J2b*b2_12CH2D_rate/fcho_b
        B2_12CHD2 = J2b*b2_12CHD2_rate/fcho_b
        B2_13CH3  = J2b*b2_13CH3_rate/fcho_b
        B2_13CH2D = J2b*b2_13CH2D_rate/fcho_b

    # Depletion of CHO-MFR pools by the backward step, grouped by source pool.
    if J2b == 0.0 or fcho_b <= 0.0:
        B2_12CHO = B2_13CHO = B2_12CDO = B2_13CDO = 0.0
    else:
        B2_12CHO = J2b*(R[17]*(XF420H2**2) + R[17]*(2*XF420H2*XF420D)*a2dfb + R[17]*(XF420D**2)*a2ddfb)/fcho_b
        B2_13CHO = J2b*(R[18]*a2cfb*(XF420H2**2) + R[18]*(2*XF420H2*XF420D)*a2cdfb)/fcho_b
        B2_12CDO = J2b*(R[19]*(XF420H2**2)*a2dfb + R[19]*(2*XF420H2*XF420D)*a2ddfb)/fcho_b
        B2_13CDO = J2b*(R[20]*(XF420H2**2)*a2cdfb)/fcho_b
    
    # ------------------------------------------------------------------
    # Direct H/D exchange: CH3-SCoM <-> CH2D-SCoM <-> CHD2-SCoM.
    # Water is treated as a fixed, effectively infinite reservoir. Therefore,
    # exchange conserves carbon and the total CH3-SCoM pool, but not hydrogen
    # isotopes within the explicitly modeled metabolite pools.
    #
    # For three equivalent methyl-H positions, the statistical factors are:
    #   0D -> 1D: 3 available H sites; 1D -> 0D: 1 available D site
    #   1D -> 2D: 2 available H sites; 2D -> 1D: 2 available D sites
    # At equilibrium this gives
    #   [CH2D]/(3[CH3]) = alphaD_ch3scom_h2o * (D/H)_H2O.
    # ------------------------------------------------------------------
    XH2O_ex = 1.0-FH2O
    XD2O_ex = FH2O

    # Symmetric parameterization keeps the geometric mean of kf and kb equal
    # to kex while imposing kf/kb = the requested equilibrium fractionation.
    alpha_ex_12_01 = alphaD_ch3scom_h2o
    alpha_ex_12_12 = alphaD_ch3scom_h2o*gammaDD_ch3scom_h2o
    alpha_ex_13_01 = alphaD_ch3scom_h2o*gammaCD_ch3scom_h2o

    kex12_01_f = kex_ch3scom_h2o*np.sqrt(alpha_ex_12_01)
    kex12_01_b = kex_ch3scom_h2o/np.sqrt(alpha_ex_12_01)
    kex12_12_f = kex_ch3scom_h2o*np.sqrt(alpha_ex_12_12)
    kex12_12_b = kex_ch3scom_h2o/np.sqrt(alpha_ex_12_12)
    kex13_01_f = kex_ch3scom_h2o*np.sqrt(alpha_ex_13_01)
    kex13_01_b = kex_ch3scom_h2o/np.sqrt(alpha_ex_13_01)

    Jex12_01_f = 3.0*kex12_01_f*XD2O_ex*R[5]
    Jex12_01_b = 1.0*kex12_01_b*XH2O_ex*R[7]
    Jex12_12_f = 2.0*kex12_12_f*XD2O_ex*R[7]
    Jex12_12_b = 2.0*kex12_12_b*XH2O_ex*R[9]

    # The current state vector contains 13CH3-SCoM and 13CH2D-SCoM, but not
    # 13CHD2-SCoM. The omitted 13C+2D pool is negligible at natural D/H.
    Jex13_01_f = 3.0*kex13_01_f*XD2O_ex*R[6]
    Jex13_01_b = 1.0*kex13_01_b*XH2O_ex*R[8]

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
    b3_12CHO_rate = R[21]*XH2O
    b3_12CDO_rate = R[21]*XD2O*a3dfb
    b3_13CHO_rate = R[22]*a3cfb*XH2O
    b3_13CDO_rate = R[22]*XD2O*a3cdfb
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
    Jadvin=Jnet/phi # Advection rate for in and out in open system
    Jadvout=Jadvin-Jnet
    fch4in=sum(R[12:17]) # Total methane abundance outside
    fch4out=sum(R[0:5]) # Total methane abundance inside
    dRdt[0]=Jin*R[12]/fch4in-Jout*R[0]/fch4out-k1f*R[0]+k1b*R[5]*R[10] # 12CH4
    dRdt[1]=Jin*R[13]/fch4in-Jout*R[1]/fch4out-k1f*R[1]*a1cff+k1b*R[6]*a1cfb*R[10] # 13CH4
    dRdt[2]=Jin*R[14]/fch4in-Jout*R[2]/fch4out-1/4*k1f*R[2]*a1dffp-3/4*k1f*R[2]*a1dffs+k1b*R[5]*R[11]*a1dfbp+k1b*R[7]*R[10]*a1dfbs # 12CH3D
    dRdt[3]=Jin*R[15]/fch4in-Jout*R[3]/fch4out-1/4*k1f*R[3]*a1cdffp-3/4*k1f*R[3]*a1cdffs+k1b*R[6]*R[11]*a1cdfbp+k1b*R[8]*R[10]*a1cdfbs # 13CH3D
    dRdt[4]=Jin*R[16]/fch4in-Jout*R[4]/fch4out-1/2*k1f*R[4]*a1ddffp-1/2*k1f*R[4]*a1ddffs+k1b*R[7]*R[11]*a1ddfbp+k1b*R[9]*R[10]*a1ddfbs #12CH2D2
    dRdt[5]=(k1f*R[0]+1/4*k1f*R[2]*a1dffp-k1b*R[5]*R[11]*a1dfbp-k1b*R[5]*R[10]
              -F2_12CH3+B2_12CH3-Jex12_01_f+Jex12_01_b) # 12CH3
    dRdt[6]=(k1f*R[1]*a1cff+1/4*k1f*R[3]*a1cdffp-k1b*R[6]*R[11]*a1cdfbp-k1b*R[6]*R[10]*a1cfb
              -F2_13CH3+B2_13CH3-Jex13_01_f+Jex13_01_b) # 13CH3
    dRdt[7]=(3/4*k1f*R[2]*a1dffs+1/2*k1f*R[4]*a1ddffp-k1b*R[7]*R[10]*a1dfbs-k1b*R[7]*R[11]*a1ddfbp
              -F2_12CH2D+B2_12CH2D+Jex12_01_f-Jex12_01_b-Jex12_12_f+Jex12_12_b) # 12CH2D
    dRdt[8]=(3/4*k1f*R[3]*a1cdffs-k1b*R[8]*R[10]*a1cdfbs-F2_13CH2D+B2_13CH2D
              +Jex13_01_f-Jex13_01_b) # 13CH2D
    dRdt[9]=(1/2*k1f*R[4]*a1ddffs-k1b*R[9]*R[10]*a1ddfbs-F2_12CHD2+B2_12CHD2
              +Jex12_12_f-Jex12_12_b) # 12CHD2
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


fig_bulk,ax_bulk=plt.subplots(figsize=(12,12))
ax_bulk.plot(d13C[0,:],dD[0,:], linewidth=2.5, color="black", linestyle="-.", alpha=1.0, label="Model, R1="+str(rev1_list[0])+", R2="+str(rev2_list[0])+", R3="+str(rev3_list[0]),zorder=1)
ax_bulk.plot(d13C[1,:],dD[1,:], linewidth=2.5, linestyle=":", color="purple", alpha=1.0,label="Model, R1="+str(rev1_list[1])+", R2="+str(rev2_list[1])+", R3="+str(rev3_list[1]),zorder=1)
ax_bulk.plot(d13C[2,:],dD[2,:], linewidth=2.5, linestyle="--", color="blue", alpha=1.0,label="Model, R1="+str(rev1_list[2])+", R2="+str(rev2_list[2])+", R3="+str(rev3_list[2]),zorder=1)
ax_bulk.plot(d13C[3,:],dD[3,:], linewidth=2.5, linestyle=(5,(10,3)), color="orange", alpha=1.0, label="Model, R1="+str(rev1_list[3])+", R2="+str(rev2_list[3])+", R3="+str(rev3_list[3]), zorder=1)
ax_bulk.plot(d13C[4,:],dD[4,:], linewidth=2.5, color="red", alpha=1.0, label="Model, R1="+str(rev1_list[4])+", R2="+str(rev2_list[4])+", R3="+str(rev3_list[4]), zorder=1)


# ax_bulk.errorbar(ANME2d["d13C"],ANME2d["dD"],xerr=ANME2d["cse"],yerr=ANME2d["dse"], markersize=16,label=r'N-AOM (this study)', fmt='o', 
#         markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
# ax_bulk.errorbar(Oct["d13C"],Oct["dD"],xerr=Oct["cse"],yerr=Oct["dse"], markersize=16,label=r'NC10+ANME+Oct', fmt='D', 
#         markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
# ax_bulk.errorbar(AOM_P["d13C"],AOM_P["dD"],xerr=AOM_P["cse"],yerr=AOM_P["dse"], markersize=12,label=r'S-AOM (High sulfate)', fmt='s', 
#         markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
# ax_bulk.errorbar(AOM_P_LS["d13C"],AOM_P_LS["dD"],xerr=AOM_P_LS["cse"],yerr=AOM_P_LS["dse"], markersize=12,label=r'S-AOM (Low sulfate, Liu)', fmt='s', 
#         markerfacecolor='gray', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
# ax_bulk.errorbar(AOM_wegener["d13C"],AOM_wegener["dD"],xerr=AOM_wegener["cse"],yerr=AOM_wegener["dse"], markersize=14, label=r"S-AOM (Wegener et al.)", 
#         fmt='^', markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
# ax_bulk.errorbar(AOM_ono["d13C"],AOM_ono["dD"],xerr=AOM_ono["cse"],yerr=AOM_ono["dse"], markersize=14, label=r"S-AOM (Ono et al.)",
#                  fmt='v', markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
# ax_bulk.errorbar(AOM_wegener_LS["d13C"],AOM_wegener_LS["dD"],xerr=AOM_wegener_LS["cse"],yerr=AOM_wegener_LS["dse"], markersize=14, label=r"S-AOM (Low sulfate, Wegener)",
#                  fmt='^', markerfacecolor='gray', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
# ax_bulk.errorbar(NC10["d13C"], NC10["dD"],xerr=NC10["cse"],yerr=NC10["dse"], markersize=18,label=r'NC10', fmt='o', 
#         markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
# ax_bulk.errorbar(AeOM_P["d13C"], AeOM_P["dD"],xerr=AeOM_P["cse"],yerr=AeOM_P["dse"], markersize=12,label=r'AeOM', fmt='o', 
#         markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
set_axis(ax_bulk,'$\delta^{13}$C (\u2030)','$\delta$D (\u2030)')
# ax_bulk.set_xlim([-68,10])
# ax_bulk.set_ylim([-180,260])
# ax_bulk.xaxis.set_minor_locator(MultipleLocator(4))
# ax_bulk.yaxis.set_minor_locator(MultipleLocator(10))
ax_bulk.legend(fontsize=15)

fig_clump,ax_clump=plt.subplots(figsize=(12,12))
ax_clump.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax_clump.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)


ax_clump.plot(D13CH3D[0,:],D12CH2D2[0,:], linewidth=2.5, color="black",linestyle="-.",alpha=1.0,zorder=1)
ax_clump.plot(D13CH3D[1,:],D12CH2D2[1,:], linewidth=2.5, linestyle=":", color="purple", alpha=1.0,zorder=1)
ax_clump.plot(D13CH3D[2,:],D12CH2D2[2,:], linewidth=2.5, linestyle="--", color="blue", alpha=1.0,zorder=1)
ax_clump.plot(D13CH3D[3,:],D12CH2D2[3,:], linewidth=2.5, linestyle=(5,(10,3)),color="orange", alpha=1.0,zorder=1)
ax_clump.plot(D13CH3D[4,:],D12CH2D2[4,:], linewidth=2.5, color="red", alpha=1.0,zorder=1)
# ax_clump.errorbar(ANME2d["D13CH3D"],ANME2d["D12CH2D2"],xerr=ANME2d["cdse"],yerr=ANME2d["ddse"], markersize=18,label=r'N-AOM (this study)', fmt='o', 
#         markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
# ax_clump.errorbar(Oct["D13CH3D"],Oct["D12CH2D2"],xerr=Oct["cdse"],yerr=Oct["ddse"], markersize=18,label=r'NC10+ANME+Oct', fmt='D', 
#         markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
# ax_clump.errorbar(AOM_P["D13CH3D"],AOM_P["D12CH2D2"],xerr=AOM_P["cdse"],yerr=AOM_P["ddse"], markersize=12,label=r'S-AOM (High sulfate)', fmt='s', 
#         markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
# ax_clump.errorbar(AOM_P_LS["D13CH3D"],AOM_P_LS["D12CH2D2"],xerr=AOM_P_LS["cdse"],yerr=AOM_P_LS["ddse"], markersize=12,label=r'S-AOM (Low sulfate)', fmt='s', 
#         markerfacecolor='gray', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
# # ax_clump.errorbar(NC10["D13CH3D"], NC10["D12CH2D2"],xerr=NC10["cdse"],yerr=NC10["ddse"], markersize=18,label=r'NC10 (this study)', fmt='o', 
# #         markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
# ax_clump.errorbar(AeOM_P["D13CH3D"], AeOM_P["D12CH2D2"],xerr=AeOM_P["cdse"],yerr=AeOM_P["ddse"], markersize=12,label=r'AeOM', fmt='o', 
#         markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-2)
set_axis(ax_clump,'$\Delta^{13}$CH$_3$D (\u2030)','$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
# ax_clump.set_ylim([-55,80])
# ax_clump.set_xlim([-20,24])
# ax_clump.xaxis.set_minor_locator(MultipleLocator(2))
# ax_clump.yaxis.set_minor_locator(MultipleLocator(4))
ax_clump.legend(fontsize=18)


if output==True:
    fig_bulk.savefig("bulk_model.pdf",bbox_inches="tight")
    fig_clump.savefig("clump_model.pdf",bbox_inches="tight")
    fig_f1.savefig("model_f1.pdf",bbox_inches="tight")
    fig_f2.savefig("model_f2.pdf",bbox_inches="tight")
    fig_f3.savefig("model_f3.pdf",bbox_inches="tight")
    fig_f4.savefig("model_f4.pdf",bbox_inches="tight")
