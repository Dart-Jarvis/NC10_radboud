# This is a model to model the reversibility of AOM
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

output=True # True if you want to save the plots as pdf
ff="ab initio" # Select the KIEs in the model: 
model="INT"
# Fixed AOM endmember reversibility values.
# R1: CH4 <-> CH3-SCoM; R2: CH3-SCoM <-> CHO-MFR; R3: CHO-MFR <-> CO2.
rev1_AOM=0.32
rev2_AOM=0.97
rev3_AOM=0.86

# Fraction of total methane oxidation rate contributed by irreversible AeOM/NC10.
# Edit this list to choose the AeOM contribution. Each value must be between 0 and 1.
phi_NC10_list=[0.0,0.25,0.50,0.75,1.0]

# Keep the AOM endmember fixed while varying phi_NC10.
rev1_list=[rev1_AOM]*len(phi_NC10_list)
rev2_list=[rev2_AOM]*len(phi_NC10_list)
rev3_list=[rev3_AOM]*len(phi_NC10_list)

t_lower=0.000 # minimum time for time interval
time_list=[50.0]*len(phi_NC10_list) # Maximum time for each phi_NC10 model
num=100000 # Number of tim steps
# "experiment": data from Scheller et al., 2013;
# "Ab initio": ab initio calculation in this study
dDH2O = -50.0 # permil dD_H2O
RVPDB = 0.0112372 # Standard carbon isotope ratio (VPDB)
RVSMOW = 1.5576e-4 # Standard hydrogen isotope ratio (VSMOW)
RH2O=RVSMOW*(dDH2O/1000+1) # D/H ratio in water
FH2O=RH2O/(1+RH2O) # D/(D+H) ratio in water

# Irreversible aerobic methane oxidation (AeOM/NC10) isotope fractionation factors.
# These factors are applied as a direct irreversible sink on the extracellular CH4 pool.
aC_aeom=0.9758
aD_aeom=0.7754
aCD_aeom=0.7564
aDD_aeom=0.5764

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
        a1cff=0.9364
        a1dffp=0.5203
        a1dffs=0.8341
        a1cdffp=0.4844
        a1cdffs=0.7814
        a1ddffp=0.4293
        a1ddffs=0.6972
    if model=="INT":
        a1cff=0.9368
        a1dffp=0.5276
        a1dffs=0.8517
        a1cdffp=0.4915 # 0.3459 for experimental observations
        a1cdffs=0.7983 # 0.8322
        a1ddffp=0.4447 # 0.3102
        a1ddffs=0.7267 # 0.7600
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
gammaCDff2=0.979
gammaDDff2=0.92
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
ahscobeq=0.4686 # aHSCOB-H2O=R_H2O/R_HSCOB HSCOB-->H2O, equilibrium value
ahscobf=1.0 # Forward fractionation HS-CoB --> H2O, best-fit value in Wegener et al.
ahscobb=ahscobf*ahscobeq
RHSCOB=RH2O/ahscobeq # D/H ratio in HS-CoB, assuming equilibrium with water
rev_hscob=0.99

# reversibility of cross-membrane transport, assuming highly reversibile methane exchange inside and outside the cells, without any isotope fractionation.
rev_tr=0.99

# Set up initial conditions
# Abundance of all relevant methane isotopologues 12CH4, 13CH4, 12CH3D, 13CH3D, 12CH2D2
# The abundance of tank gas
abundance=[
    9.8883E-01,
    1.0655E-02,
    5.1189E-04,
    5.5330E-06,
    1.0007E-07
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
def dfdt(t,R,k,rev2,rev3,phi_NC10):
    k1f,k1b=k     # unpack first-step rate constants; rev2 and rev3 are passed separately
    if phi_NC10 < 0.0 or phi_NC10 > 1.0:
        raise ValueError("phi_NC10 must be between 0 and 1.")
    aom_fraction = 1.0 - phi_NC10
    # Calculate total forward and backward fluxes for the first step.
    Jf = (k1f*R[0] + k1f*R[1]*a1cff
          + (1/4*k1f*R[2]*a1dffp + 3/4*k1f*R[2]*a1dffs)
          + (1/4*k1f*R[3]*a1cdffp + 3/4*k1f*R[3]*a1cdffs)
          + (1/2*k1f*R[4]*a1ddffp + 1/2*k1f*R[4]*a1ddffs))

    Jb = (k1b*R[5]*R[10] + k1b*R[6]*a1cfb*R[10]
          + (k1b*R[5]*R[11]*a1dfbp + k1b*R[7]*R[10]*a1dfbs)
          + (k1b*R[6]*R[11]*a1cdfbp + k1b*R[8]*R[10]*a1cdfbs)
          + (k1b*R[7]*R[11]*a1ddfbp + k1b*R[9]*R[10]*a1ddfbs))

    # Split the methane oxidation rate between fixed-endmember AOM and irreversible AeOM.
    # Jnet_total is the oxidation-rate potential of the original pure-AOM model.
    # AOM receives (1 - phi_NC10) of this rate, while AeOM receives phi_NC10.
    # Both forward and backward AOM fluxes are scaled by the same factor, so the
    # fixed AOM endmember reversibility values stay at R1=0.32, R2=0.96, R3=0.84.
    Jnet_total = Jf - Jb
    Jaeom = phi_NC10*max(Jnet_total,0.0)
    Jf *= aom_fraction
    Jb *= aom_fraction
    k1f *= aom_fraction
    k1b *= aom_fraction

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
    # and the other two methyl H/D positions are supplied from water.
    XH2O = 1.0-FH2O
    XD2O = FH2O
    b2_12CH3_rate  = R[17]*(XH2O**2)
    b2_12CH2D_rate = (R[17]*(2*XH2O*XD2O)*a2dfb + R[19]*(XH2O**2)*a2dfb)
    b2_12CHD2_rate = (R[17]*(XD2O**2)*a2ddfb + R[19]*(2*XH2O*XD2O)*a2ddfb)
    b2_13CH3_rate  = R[18]*a2cfb*(XH2O**2)
    b2_13CH2D_rate = (R[18]*(2*XH2O*XD2O)*a2cdfb + R[20]*(XH2O**2)*a2cdfb)
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
        B2_12CHO = J2b*(R[17]*(XH2O**2) + R[17]*(2*XH2O*XD2O)*a2dfb + R[17]*(XD2O**2)*a2ddfb)/fcho_b
        B2_13CHO = J2b*(R[18]*a2cfb*(XH2O**2) + R[18]*(2*XH2O*XD2O)*a2cdfb)/fcho_b
        B2_12CDO = J2b*(R[19]*(XH2O**2)*a2dfb + R[19]*(2*XH2O*XD2O)*a2ddfb)/fcho_b
        B2_13CDO = J2b*(R[20]*(XH2O**2)*a2cdfb)/fcho_b

    # ------------------------------------------------------------------
    # Step 3: CHO-MFR <-> CO2 with constant flux reversibility
    # J3b/J3f = rev3 and J3f - J3b = Jnet.
    # ------------------------------------------------------------------
    J3f = Jnet/(1.0-rev3)
    J3b = Jnet*rev3/(1.0-rev3)

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
    fch4in=sum(R[12:17]) # Total methane abundance outside
    fch4out=sum(R[0:5]) # Total methane abundance inside

    # Irreversible AeOM/NC10 sink directly removes extracellular methane.
    # The total AeOM rate is Jaeom and its isotopologue partitioning is weighted
    # by the AeOM fractionation factors supplied above.
    faeom = (R[12] + R[13]*aC_aeom + R[14]*aD_aeom
             + R[15]*aCD_aeom + R[16]*aDD_aeom)
    if Jaeom <= 0.0 or faeom <= 0.0:
        AeOM_12CH4 = AeOM_13CH4 = AeOM_12CH3D = AeOM_13CH3D = AeOM_12CH2D2 = 0.0
    else:
        AeOM_12CH4   = Jaeom*R[12]/faeom
        AeOM_13CH4   = Jaeom*R[13]*aC_aeom/faeom
        AeOM_12CH3D  = Jaeom*R[14]*aD_aeom/faeom
        AeOM_13CH3D  = Jaeom*R[15]*aCD_aeom/faeom
        AeOM_12CH2D2 = Jaeom*R[16]*aDD_aeom/faeom
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
    dRdt[12]=-Jin*R[12]/fch4in+Jout*R[0]/fch4out-AeOM_12CH4
    dRdt[13]=-Jin*R[13]/fch4in+Jout*R[1]/fch4out-AeOM_13CH4
    dRdt[14]=-Jin*R[14]/fch4in+Jout*R[2]/fch4out-AeOM_12CH3D
    dRdt[15]=-Jin*R[15]/fch4in+Jout*R[3]/fch4out-AeOM_13CH3D
    dRdt[16]=-Jin*R[16]/fch4in+Jout*R[4]/fch4out-AeOM_12CH2D2
    dRdt[17]=F2_12CHO-B2_12CHO-F3_12CHO+B3_12CHO # 12CHO-MFR
    dRdt[18]=F2_13CHO-B2_13CHO-F3_13CHO+B3_13CHO # 13CHO-MFR
    dRdt[19]=F2_12CDO-B2_12CDO-F3_12CDO+B3_12CDO # 12CDO-MFR
    dRdt[20]=F2_13CDO-B2_13CDO-F3_13CDO+B3_13CDO # 13CDO-MFR
    dRdt[21]=F3_12CO2-B3_12CO2+AeOM_12CH4+AeOM_12CH3D+AeOM_12CH2D2 # 12CO2
    dRdt[22]=F3_13CO2-B3_13CO2+AeOM_13CH4+AeOM_13CH3D # 13CO2
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
def model_rev(rev1, rev2, rev3, phi_NC10, tmax, num):
    k1f_input=1.0
    k1b_input=k1f_input*rev1
    k=[k1f_input,k1b_input]
    tint=np.linspace(t_lower,tmax,num) 
    solution=solve_ivp(dfdt,(t_lower,tmax),R0,args=(k,rev2,rev3,phi_NC10),t_eval=tint,atol=1.0e-12,rtol=1.0e-9)
    tot_sol,fCH4_sol,d13C_sol,dD_sol,D13CH3D_sol,D12CH2D2_sol=process(solution)
    rev1_sol = np.array([jb_jf_ratio(t, R, k) for t, R in zip(solution.t, solution.y.T)])
    rev2_sol = np.array([j2b_j2f_ratio(t, R, rev2) for t, R in zip(solution.t, solution.y.T)])
    rev3_sol = np.array([j3b_j3f_ratio(t, R, rev3) for t, R in zip(solution.t, solution.y.T)])
    return tot_sol,fCH4_sol,rev1_sol,rev2_sol,rev3_sol,d13C_sol,dD_sol,D13CH3D_sol,D12CH2D2_sol

if len(rev1_list) != len(rev2_list) or len(rev1_list) != len(rev3_list):
    raise ValueError("rev1_list, rev2_list, and rev3_list must have the same length for pairwise plotting.")
if len(phi_NC10_list) != len(rev1_list):
    raise ValueError("phi_NC10_list must have the same length as rev1_list/rev2_list/rev3_list.")
if len(time_list) != len(rev1_list):
    raise ValueError("time_list must have the same length as rev1_list/rev2_list/rev3_list.")
if any(phi < 0.0 or phi > 1.0 for phi in phi_NC10_list):
    raise ValueError("All phi_NC10 values must be between 0 and 1.")

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
    tot[i,:],fCH4[i,:],rev1[i,:],rev2[i,:],rev3[i,:],d13C[i,:],dD[i,:],D13CH3D[i,:],D12CH2D2[i,:]=model_rev(rev1_list[i],rev2_list[i],rev3_list[i],phi_NC10_list[i],time_list[i],num)

# Standalone irreversible AeOM Rayleigh trajectory using the same AeOM factors
# as the ODE sink above.
a13C_sMMO=aC_aeom
aD_sMMO=aD_aeom
a13CD_sMMO=aCD_aeom
aDD_sMMO=aDD_aeom
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

T0=data[data['label']=='0'] # T0, tank gas and controls
NC10=data[data['label']=='1'] # NC10
NC10_ANME=data[data['label']=='2'] # NC10+ANME
ANME2d=data[data['label']=='3'] # ANME-2d
Oct=data[data['label']=='4'] # pMMO inhibited experiments
BES=data[data['label']=='5'] # MCR inhibited experiments
AeOM_P=data[data['label']=='P'] # Previous data from Li et al., 2024
AeOM_P1=data[data['label']=='P1'] # Previous data from Krause et al., 2022
AeOM_P2=data[data['label']=='P2'] # wang et al., 2016
AOM_P=data[data['label']=='AOM_P'] # Previous data from Liu et al., 2023, Kinetic Sulfate AOM
AOM_P_LS=data[data['label']=='AOM_P_LS'] # AOM data from Liu et al., 2023, equilibrium S-AOM
AOM_ono=data[data['label']=='AOM_ono'] # AOM data from Ono et al., 2021 (high sulfate series)
AeOM_m=AeOM_P[AeOM_P['f']>=0.1]
AOM_wegener=data[data['label']=='AOM_Wegener']
AOM_wegener_LS=data[data['label']=='AOM_Wegener_LS']

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
    norm=np.zeros([n_models,data.shape[1]])
    for i in range(norm.shape[0]):
        norm[i,:]=np.log((data[i,:]+1000)/(data[i,0]+1000))
    return norm

def normalize_clumped_dat(data): # Get rid of the influence of T0 values
    norm=np.zeros([n_models,data.shape[1]])
    for i in range(norm.shape[0]):
        norm[i,:]=data[i,:]-data[i,0]
    return norm

def make_label():
    labellist={}
    for i in range(len(phi_NC10_list)):
        labellist[str(i)]=r"$\phi_{\rm NC10}$="+str(phi_NC10_list[i])
    return labellist

llist=make_label()
normC=normalize_dat(d13C)
normD=normalize_dat(dD)
normCD=normalize_clumped_dat(D13CH3D)
normDD=normalize_clumped_dat(D12CH2D2)

# -----------------------------------------------------------------------------
# Data/model plotting helpers
# -----------------------------------------------------------------------------
# Toggle datasets here. All enabled datasets are plotted as hollow symbols.
DATA_SERIES = [
    {
        "name": "NC10",
        "data": NC10,
        "enabled": True,
        "fmt": "o",                 # hollow circle
        "markerfacecolor": "none",
        "markeredgecolor": "red",
        "ecolor": "red",
    },
    {
        "name": "NC10+ANME",
        "data": NC10_ANME,
        "enabled": True,
        "fmt": "^",                 # hollow upward triangle
        "markerfacecolor": "none",
        "markeredgecolor": "blue",
        "ecolor": "blue",
    },
    {
        "name": "ANME",
        "data": ANME2d,
        "enabled": True,
        "fmt": "s",                 # hollow square
        "markerfacecolor": "none",
        "markeredgecolor": "black",
        "ecolor": "black",
    },
    {
        "name": "NC10+ANME+Oct",
        "data": Oct,
        "enabled": True,
        "fmt": "D",                 # hollow diamond
        "markerfacecolor": "none",
        "markeredgecolor": "black",
        "ecolor": "black",
    },
    {
        "name": "NC10+ANME+BES",
        "data": BES,
        "enabled": True,
        "fmt": "v",                 # hollow downward triangle
        "markerfacecolor": "none",
        "markeredgecolor": "red",
        "ecolor": "red",
    },
]

MODEL_GRAY_LEVELS = np.linspace(0.15, 0.70, n_models)
MODEL_LINESTYLES = ["-", "--", "-.", ":", (0, (8, 3))]


def _error_values(df, col):
    """Return an error column if it exists and contains useful values; otherwise None."""
    if col is None or col not in df.columns:
        return None
    vals = df[col]
    if vals.empty:
        return None
    return vals


def plot_selected_data(ax, xcol, ycol, xerr_col=None, yerr_col=None, markersize=14, zorder=3):
    """Plot the currently enabled experimental series on one axis."""
    for style in DATA_SERIES:
        if not style["enabled"]:
            continue
        df = style["data"]
        if df.empty or xcol not in df.columns or ycol not in df.columns:
            continue
        ax.errorbar(
            df[xcol], df[ycol],
            xerr=_error_values(df, xerr_col),
            yerr=_error_values(df, yerr_col),
            markersize=markersize,
            fmt=style["fmt"],
            linestyle="none",
            label=style["name"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["markeredgecolor"],
            markeredgewidth=2.5,
            ecolor=style["ecolor"],
            elinewidth=2.5,
            capsize=0,
            zorder=zorder,
        )


def plot_model_curves(ax, x, y, label=True, linewidth=2.8, zorder=1):
    """Plot all model curves using different gray scales."""
    for i in range(n_models):
        ax.plot(
            x[i, :], y[i, :],
            linewidth=linewidth,
            linestyle=MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)],
            color=str(MODEL_GRAY_LEVELS[i]),
            alpha=1.0,
            label=llist[str(i)] if label else None,
            zorder=zorder,
        )


def autofit_axes(ax, invert_x=False, xmargin=0.07, ymargin=0.07):
    """Autoscale the view after plotting, with optional reversed x-axis for f plots."""
    ax.relim()
    ax.autoscale_view()
    ax.margins(x=xmargin, y=ymargin)
    if invert_x:
        left, right = ax.get_xlim()
        if left < right:
            ax.set_xlim(right, left)


# -----------------------------------------------------------------------------
# Normalized bulk isotope plot
# -----------------------------------------------------------------------------
fig_bulk0, ax_bulk0 = plt.subplots(figsize=(12, 12))
plot_model_curves(ax_bulk0, normC, normD, label=True)
plot_selected_data(ax_bulk0, "ln(c/c0)", "ln(d/d0)", "lncse", "lndse", markersize=14)
set_axis(ax_bulk0, r'ln$\frac{\delta^{13}{\rm C}+1000}{\delta^{13}{\rm C}_{\rm init}+1000}$',
         r'ln$\frac{\delta{\rm D}+1000}{\delta{\rm D}_{\rm init}+1000}$')
autofit_axes(ax_bulk0)
ax_bulk0.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_bulk0.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_bulk0.legend(fontsize=16)


# -----------------------------------------------------------------------------
# Normalized clumped isotope plot
# -----------------------------------------------------------------------------
fig_clump0, ax_clump0 = plt.subplots(figsize=(12, 12))
plot_model_curves(ax_clump0, normCD, normDD, label=True)
plot_selected_data(ax_clump0, "DeltaCD", "DeltaDD", "se", "se", markersize=14)
set_axis(ax_clump0, r'$\Delta \Delta^{13}$CH$_3$D'+'(\u2030)',
         r'$\Delta \Delta^{12}$CH$_2$D$_2$'+'(\u2030)')
autofit_axes(ax_clump0)
ax_clump0.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_clump0.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_clump0.legend(fontsize=16)


# -----------------------------------------------------------------------------
# Absolute bulk isotope plot
# -----------------------------------------------------------------------------
fig_bulk, ax_bulk = plt.subplots(figsize=(12, 12))
plot_model_curves(ax_bulk, d13C, dD, label=True)
plot_selected_data(ax_bulk, "d13C", "dD", "cse", "dse", markersize=14)
set_axis(ax_bulk, r'$\delta^{13}$C (\u2030)', r'$\delta$D (\u2030)')
autofit_axes(ax_bulk)
ax_bulk.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_bulk.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_bulk.legend(fontsize=15)


# -----------------------------------------------------------------------------
# Absolute clumped isotope plot
# -----------------------------------------------------------------------------
fig_clump, ax_clump = plt.subplots(figsize=(12, 12))
ax_clump.plot(equib['D13CH3D'], equib['D12CH2D2'], '-', color='0.2',
              label='Equilibrium', linewidth=2.5, markersize=15)
for i in range(len(equib)):
    if equib['p'].iloc[i] == 1:
        ax_clump.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],
                         color='0.2', s=60)
plot_model_curves(ax_clump, D13CH3D, D12CH2D2, label=True)
plot_selected_data(ax_clump, "D13CH3D", "D12CH2D2", "cdse", "ddse", markersize=14)
set_axis(ax_clump, r'$\Delta^{13}$CH$_3$D (\u2030)', r'$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
autofit_axes(ax_clump)
ax_clump.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_clump.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_clump.legend(fontsize=16)


# -----------------------------------------------------------------------------
# Publication-style bulk isotope plot
# -----------------------------------------------------------------------------
fig_bulk1, ax_bulk1 = plt.subplots(figsize=(12, 12))
plot_model_curves(ax_bulk1, d13C, dD, label=True)
plot_selected_data(ax_bulk1, "d13C", "dD", "cse", "dse", markersize=16)
set_axis(ax_bulk1, r'$\delta^{13}$C (\u2030)', r'$\delta$D (\u2030)')
autofit_axes(ax_bulk1)
ax_bulk1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_bulk1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_bulk1.legend(fontsize=15)


# -----------------------------------------------------------------------------
# Publication-style clumped isotope plot
# -----------------------------------------------------------------------------
fig_clump1, ax_clump1 = plt.subplots(figsize=(12, 12))
ax_clump1.plot(equib['D13CH3D'], equib['D12CH2D2'], '-', color='0.2',
               label='Equilibrium', linewidth=2.5, markersize=15)
for i in range(len(equib)):
    if equib['p'].iloc[i] == 1:
        ax_clump1.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],
                          color='0.2', s=60)
plot_model_curves(ax_clump1, D13CH3D, D12CH2D2, label=True)
plot_selected_data(ax_clump1, "D13CH3D", "D12CH2D2", "cdse", "ddse", markersize=16)
set_axis(ax_clump1, r'$\Delta^{13}$CH$_3$D (\u2030)', r'$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
autofit_axes(ax_clump1)
ax_clump1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_clump1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_clump1.legend(fontsize=16)


# -----------------------------------------------------------------------------
# Residual methane fraction plots
# -----------------------------------------------------------------------------
fig_f1, ax_f1 = plt.subplots(figsize=(12, 6))
fig_f2, ax_f2 = plt.subplots(figsize=(12, 6))
fig_f3, ax_f3 = plt.subplots(figsize=(12, 6))
fig_f4, ax_f4 = plt.subplots(figsize=(12, 6))


def plotf_norm(x, n, ne, ax):
    plot_model_curves(ax, fCH4, x, label=True)
    plot_selected_data(ax, "f", n, "fse", ne, markersize=14)
    autofit_axes(ax, invert_x=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))


plotf_norm(normC, "ln(c/c0)", "lncse", ax_f1)
set_axis(ax_f1, r"Residual methane fraction, f", r'ln$\frac{\delta^{13}{\rm C}+1000}{\delta^{13}{\rm C}_{\rm init}+1000}$')

plotf_norm(normD, "ln(d/d0)", "lndse", ax_f2)
set_axis(ax_f2, r"Residual methane fraction, f", r'ln$\frac{\delta{\rm D}+1000}{\delta{\rm D}_{\rm init}+1000}$')

plotf_norm(normCD, "DeltaCD", "se", ax_f3)
set_axis(ax_f3, r"Residual methane fraction, f", r'$\Delta \Delta^{13}$CH$_3$D'+'(\u2030)')

plotf_norm(normDD, "DeltaDD", "se", ax_f4)
set_axis(ax_f4, r"Residual methane fraction, f", r'$\Delta \Delta^{12}$CH$_2$D$_2$'+'(\u2030)')

ax_f2.legend(fontsize=18)

if output==True:
    fig_bulk.savefig("bulk_mixing_model.pdf",bbox_inches="tight")
    fig_clump.savefig("clump_mixing_model.pdf",bbox_inches="tight")
    fig_bulk0.savefig("bulk0_mixing_model.pdf",bbox_inches="tight")
    fig_clump0.savefig("clump0_mixing_model.pdf",bbox_inches="tight")
    fig_f1.savefig("model_f1.pdf",bbox_inches="tight")
    fig_f2.savefig("model_f2.pdf",bbox_inches="tight")
    fig_f3.savefig("model_f3.pdf",bbox_inches="tight")
    fig_f4.savefig("model_f4.pdf",bbox_inches="tight")
