# This is a model to model the reversibility of AOM
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

output=False # True if you want to save the plots as pdf
ff="ab initio" # Select the KIEs in the model: 
model="INT"
rev1_list=[0.0,0.2,0.5,0.7,0.99] # First-step reversibility list, CH4 <-> CH3-SCoM; each value ranges from 0 to <1
rev2_list=[0.0,0.1,0.3,0.4,0.99] # Second-step reversibility list, CH3-SCoM <-> CO2; each value ranges from 0 to <1
t_lower=0.000 # minimum time for time interval
time_list=[8.5,11.0,13.0,20.0,150.0] # Maximum time for time interval, relevant to the final fraction of methane left
num=100000 # Number of tim steps
# "experiment": data from Scheller et al., 2013;
# "Ab initio": ab initio calculation in this study
dDH2O = -50.0 # permil dD_H2O
RVPDB = 0.0112372 # Standard carbon isotope ratio (VPDB)
RVSMOW = 1.5576e-4 # Standard hydrogen isotope ratio (VSMOW)
RH2O=RVSMOW*(dDH2O/1000+1) # D/H ratio in water
FH2O=RH2O/(1+RH2O) # D/(D+H) ratio in water

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

if ff=="ab initio": # Ab initio calculation gets the fractionation in the backward dirction
    if model=="no INT":
        a1cff=0.9364
        a1dffp=0.5203
        a1dffs=0.8341
        a1cdffp=0.4844
        a1cdffs=0.7814
        a1ddffp=0.4293
        a1ddffs=0.6972
    # # Calculate the fractionation factors of the forward reactions for mcr
    # a1cff=a1cfb/a1cfeq
    # a1dffp=a1dfbp/a1dfeqp # primary isotope effect backwards
    # a1dffs=a1dfbs/a1dfeqs # secondary isotope effect backwards
    # a1cdffp=a1cdfbp/a1cdfeqp
    # a1cdffs=a1cdfbs/a1cdfeqs
    # a1ddffp=a1ddfbp/a1ddfeqp
    # a1ddffs=a1ddfbs/a1ddfeqs
    if model=="INT":
        a1cff=0.9368
        a1dffp=0.5276
        a1dffs=0.8517
        a1cdffp=0.4915 # 0.3459 for experimental observations
        a1cdffs=0.7983 # 0.8322
        a1ddffp=0.4447 # 0.3102
        a1ddffs=0.7267 # 0.7600
    # Calculate the fractionation factors of the forward reactions for mcr
    a1cfb=a1cff*a1cfeq
    a1dfbp=a1dffp*a1dfeqp # primary isotope effect backwards
    a1dfbs=a1dffs*a1dfeqs # secondary isotope effect backwards
    a1cdfbp=a1cdffp/a1cdfeqp
    a1cdfbs=a1cdffs/a1cdfeqs
    a1ddfbp=a1ddffp/a1ddfeqp
    a1ddfbs=a1ddffs/a1ddfeqs
    # Calculate net ff
    a1dfnet=a1dffp/4+3/4*a1dffs
    a1cdffnet=a1cdffp/4+3/4*a1cdffs
    a1ddffnet=a1ddffp/2+a1ddffs/2
    print("Net fractionation factors:")
    print(a1cff, "\n", a1dfnet, "\n", a1cdffnet, "\n", a1ddffnet)

# The kinetic isotope effect of the second step (CH3-SCoM --> CHO-MFR), from the best-fit values in Wegener et al., 2021, Table S6
# Gamma values are set at 1
gamma2cd=1.000
gamma2dd=1.000
a2cff=0.979
a2dff=1.000
a2cdff=gamma2cd*a2cff*a2dff
a2ddff=gamma2dd*a2dff**2
a2ceq=1/np.exp(18.1/1000)*1/np.exp(15.8/1000)*1/np.exp(16.9/1000)*1/np.exp(-3.3/1000)*1/np.exp(1.9/1000)*1/np.exp(17.5/1000)
a2deq=1/np.exp(42.9/1000)*((1/np.exp(81.3/1000)+1/np.exp(84.0/1000))/2)*1/np.exp(-78.2/1000)*1/np.exp(-70.5/1000)*1/np.exp(8.5/1000)*1/np.exp(153.2/1000)
a2cdeq=1/np.exp(61.0/1000)*((1/np.exp(98.8/1000)+1/np.exp(96.0/1000))/2)*1/np.exp(-17.6/1000)*((1/np.exp(-61.2/1000)+1/np.exp(-92.6/1000))/2)*1/np.exp(-74.2/1000)*1/np.exp(10.6/1000)*1/np.exp(166.2/1000)
a2ddeq=1/np.exp(85.8/1000)*((2/3*1/np.exp(44.7/1000)+1/3*1/np.exp(163.2/1000)))*1/np.exp(-196.7/1000)
a2cfb=a2cff*a2ceq
a2dfb=a2dff*a2deq
# Back-reaction clumped fractionation factors for the lumped CH3-SCoM <-> CO2 step.
a2cdfb=a2cdeq*a2cdff # Alternatively could use a gamma: a2cdfb=gamma2cd*a2cfb*a2dfb, values are similar
a2ddfb=a2ddeq*a2ddff

print("Net fractionation factors of the second step (forward):")
print(a2cff, "\n", a2dff, "\n", a2cdff, "\n", a2ddff)

# Account for hydrogen isotope exchange between CH3-SCoM and water
ach3scom_h2of=0.888 # KFF from CH3-SCoM -> H2O, best-fit value from Wegener et al., 2022
ach3scom_h2oeq=0.8748 # EFF CH3-SCoM <-> H2O
rev_ch3scom_h2o=0.99

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
abundance=[
    9.8883E-01,
    1.0655E-02,
    5.1189E-04,
    5.5330E-06,
    1.0007E-07
]

# Tank gas
# 9.8883E-01,
# 1.0654E-02,
# 5.1193E-04,
# 5.5301E-06,
# 1.0029E-07

# Species index map:
# 0:5   intracellular CH4 isotopologues: 12CH4, 13CH4, 12CH3D, 13CH3D, 12CH2D2
# 5:10  CH3-SCoM isotopologues:         12CH3, 13CH3, 12CH2D, 13CH2D, 12CHD2
# 10:12 HS-CoB isotopologues:           H, D
# 12:17 extracellular CH4 isotopologues, same order as 0:5
# 17:19 CO2 isotopologues:              12CO2, 13CO2
R=np.zeros(19) # abundance of each species involved in the reaction network

# Normalize the abundance, assuming the chemicals inside the cell (CH4,CH3-SCoM, HS-Cob) has a total abundance of 1 for each.
for i in range(5):    
    R[i]=abundance[i]/sum(abundance)
# Abundance of all relevant CH3-SCoM isotopologues 12CH3, 13CH3, 12CH2D, 13CH2D, 12CHD2
# Assume the same abundance as CH4
R[5:10]=R[0:5]
# D and H are in equilibrium with water
R[10]=1.0/(1.0+RHSCOB) # H
R[11]=(1-R[10]) # D
R[12:17]=R[0:5]*10 # Methane isotopologue abundances outside, assuming the reservoir is 10 times larger
R[17]=0.0 # 12CO2 pool produced by oxidation
R[18]=0.0 # 13CO2 pool produced by oxidation
R0=R.copy()
# Construct ode
def dfdt(t,R,k,rev2):
    k1f,k1b=k     # unpack first-step rate constants; rev2 is passed separately
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
    # Explicit reversible CH3-SCoM <-> CO2 step with constant flux reversibility.
    # As in the original first-step implementation, the total forward and backward
    # second-step fluxes are calculated from the net flux through the pathway:
    #     J2b / J2f = rev2
    #     J2f - J2b = Jf - Jb
    # Therefore J2f=(Jf-Jb)/(1-rev2) and J2b=rev2*(Jf-Jb)/(1-rev2).
    # Individual isotopologue fluxes are then partitioned according to their
    # relative reaction rates.
    dRdt=np.zeros_like(R)
    J2net = Jf - Jb
    if rev2 >= 1.0:
        raise ValueError("rev2 must be < 1.0 because J2f=(J2net)/(1-rev2).")
    J2f = J2net/(1.0-rev2)
    J2b = J2net*rev2/(1.0-rev2)

    # Forward CH3-SCoM -> CO2 flux partition.
    fch3f = (R[5] + R[6]*a2cff + R[7]*a2dff + R[8]*a2cdff + R[9]*a2ddff)
    F2_12CH3  = J2f*R[5]/fch3f
    F2_13CH3  = J2f*R[6]*a2cff/fch3f
    F2_12CH2D = J2f*R[7]*a2dff/fch3f
    F2_13CH2D = J2f*R[8]*a2cdff/fch3f
    F2_12CHD2 = J2f*R[9]*a2ddff/fch3f

    # Backward CO2 -> CH3-SCoM flux partition. The carbon isotope is supplied
    # by the CO2 pool, while H/D sites in the regenerated methyl group are
    # supplied from water. The back-reaction isotope effects weight the relative
    # rates before normalizing to the total J2b flux.
    XH2O = 1.0-FH2O
    XD2O = FH2O
    b2_12CH3_rate  = R[17]*(XH2O**3)
    b2_12CH2D_rate = R[17]*(3*XH2O**2*XD2O)*a2dfb
    b2_12CHD2_rate = R[17]*(3*XH2O*XD2O**2)*a2ddfb
    b2_13CH3_rate  = R[18]*a2cfb*(XH2O**3)
    b2_13CH2D_rate = R[18]*(3*XH2O**2*XD2O)*a2cdfb
    fco2b = (b2_12CH3_rate + b2_12CH2D_rate + b2_12CHD2_rate
             + b2_13CH3_rate + b2_13CH2D_rate)

    # If rev2=0, J2b is zero and the CO2 pool can initially be zero. Avoid a
    # 0/0 normalization while preserving zero back flux.
    if J2b == 0.0 or fco2b <= 0.0:
        B2_12CH3 = B2_12CH2D = B2_12CHD2 = B2_13CH3 = B2_13CH2D = 0.0
    else:
        B2_12CH3  = J2b*b2_12CH3_rate/fco2b
        B2_12CH2D = J2b*b2_12CH2D_rate/fco2b
        B2_12CHD2 = J2b*b2_12CHD2_rate/fco2b
        B2_13CH3  = J2b*b2_13CH3_rate/fco2b
        B2_13CH2D = J2b*b2_13CH2D_rate/fco2b

    # Total carbon fluxes into/out of each CO2 isotopologue pool.
    F2_12CO2 = F2_12CH3 + F2_12CH2D + F2_12CHD2
    F2_13CO2 = F2_13CH3 + F2_13CH2D
    B2_12CO2 = B2_12CH3 + B2_12CH2D + B2_12CHD2
    B2_13CO2 = B2_13CH3 + B2_13CH2D

    Jin=(Jf-Jb)/(1-rev_tr)
    Jout=(Jf-Jb)*rev_tr/(1-rev_tr)
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
    dRdt[12]=-Jin*R[12]/fch4in+Jout*R[0]/fch4out
    dRdt[13]=-Jin*R[13]/fch4in+Jout*R[1]/fch4out
    dRdt[14]=-Jin*R[14]/fch4in+Jout*R[2]/fch4out
    dRdt[15]=-Jin*R[15]/fch4in+Jout*R[3]/fch4out
    dRdt[16]=-Jin*R[16]/fch4in+Jout*R[4]/fch4out
    dRdt[17]=F2_12CO2-B2_12CO2 # 12CO2
    dRdt[18]=F2_13CO2-B2_13CO2 # 13CO2
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

# Define a separate function to calculate the second-step flux reversibility.
# This is constant by construction and should equal rev2 at every time point.
def j2b_j2f_ratio(t, R, rev2):
    return rev2

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
def model_rev(rev1, rev2, tmax, num):
    k1f_input=1.0
    k1b_input=k1f_input*rev1
    k=[k1f_input,k1b_input]
    tint=np.linspace(t_lower,tmax,num) 
    solution=solve_ivp(dfdt,(t_lower,tmax),R0,args=(k,rev2),t_eval=tint,atol=1.0e-12,rtol=1.0e-9)
    tot_sol,fCH4_sol,d13C_sol,dD_sol,D13CH3D_sol,D12CH2D2_sol=process(solution)
    rev1_sol = np.array([jb_jf_ratio(t, R, k) for t, R in zip(solution.t, solution.y.T)])
    rev2_sol = np.array([j2b_j2f_ratio(t, R, rev2) for t, R in zip(solution.t, solution.y.T)])
    return tot_sol,fCH4_sol,rev1_sol,rev2_sol,d13C_sol,dD_sol,D13CH3D_sol,D12CH2D2_sol

if len(rev1_list) != len(rev2_list):
    raise ValueError("rev1_list and rev2_list must have the same length for pairwise plotting.")
if len(time_list) != len(rev1_list):
    raise ValueError("time_list must have the same length as rev1_list/rev2_list.")

n_models=len(rev1_list)
tot=np.zeros([n_models,num])
fCH4=np.zeros([n_models,num])
rev1=np.zeros([n_models,num])
rev2=np.zeros([n_models,num])
d13C=np.zeros([n_models,num])
dD=np.zeros([n_models,num])
D13CH3D=np.zeros([n_models,num])
D12CH2D2=np.zeros([n_models,num])
for i in range(n_models):
    tot[i,:],fCH4[i,:],rev1[i,:],rev2[i,:],d13C[i,:],dD[i,:],D13CH3D[i,:],D12CH2D2[i,:]=model_rev(rev1_list[i],rev2_list[i],time_list[i],num)

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

fig_bulk,ax_bulk=plt.subplots(figsize=(12,12))
ax_bulk.plot(d13C[0,:],dD[0,:], linewidth=2.5, color="black", linestyle="-.", alpha=1.0, label="Model, R1="+str(rev1_list[0])+", R2="+str(rev2_list[0]),zorder=-2)
ax_bulk.plot(d13C[1,:],dD[1,:], linewidth=2.5, linestyle=":", color="purple", alpha=1.0,label="Model, R1="+str(rev1_list[1])+", R2="+str(rev2_list[1]),zorder=-2)
ax_bulk.plot(d13C[2,:],dD[2,:], linewidth=2.5, linestyle="--", color="blue", alpha=1.0,label="Model, R1="+str(rev1_list[2])+", R2="+str(rev2_list[2]),zorder=-2)
ax_bulk.plot(d13C[3,:],dD[3,:], linewidth=2.5, linestyle=(5,(10,3)), color="orange", alpha=1.0, label="Model, R1="+str(rev1_list[3])+", R2="+str(rev2_list[3]), zorder=-2)
ax_bulk.plot(d13C[4,:],dD[4,:], linewidth=2.5, color="red", alpha=1.0, label="Model, R1="+str(rev1_list[4])+", R2="+str(rev2_list[4]), zorder=-2)


ax_bulk.errorbar(ANME2d["d13C"],ANME2d["dD"],xerr=ANME2d["cse"],yerr=ANME2d["dse"], markersize=18,label=r'N-AOM (this study)', fmt='o', 
        markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_bulk.errorbar(AOM_P["d13C"],AOM_P["dD"],xerr=AOM_P["cse"],yerr=AOM_P["dse"], markersize=14,label=r'S-AOM (Liu et al.)', fmt='s', 
        markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_bulk.errorbar(AOM_wegener["d13C"],AOM_wegener["dD"],xerr=AOM_wegener["cse"],yerr=AOM_wegener["dse"], markersize=14, label=r"S-AOM (Wegener et al.)", 
        fmt='o', markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_bulk.errorbar(AOM_ono["d13C"],AOM_ono["dD"],xerr=AOM_ono["cse"],yerr=AOM_ono["dse"], markersize=14, label=r"S-AOM (Ono et al.)",
                 fmt='^', markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_bulk.errorbar(NC10["d13C"], NC10["dD"],xerr=NC10["cse"],yerr=NC10["dse"], markersize=18,label=r'NC10', fmt='o', 
        markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_bulk.errorbar(AeOM_P["d13C"], AeOM_P["dD"],xerr=AeOM_P["cse"],yerr=AeOM_P["dse"], markersize=18,label=r'AeOM', fmt='o', 
        markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
set_axis(ax_bulk,'$\delta^{13}$C (\u2030)','$\delta$D (\u2030)')
ax_bulk.set_xlim([-60,20])
ax_bulk.set_ylim([-220,200])
ax_bulk.xaxis.set_minor_locator(MultipleLocator(2))
ax_bulk.yaxis.set_minor_locator(MultipleLocator(20))
ax_bulk.legend(fontsize=18)

fig_clump,ax_clump=plt.subplots(figsize=(12,12))
ax_clump.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax_clump.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)


ax_clump.plot(D13CH3D[0,:],D12CH2D2[0,:], linewidth=2.5, color="black",linestyle="-.",alpha=1.0,zorder=-2)
ax_clump.plot(D13CH3D[1,:],D12CH2D2[1,:], linewidth=2.5, linestyle=":", color="purple", alpha=1.0,zorder=-2)
ax_clump.plot(D13CH3D[2,:],D12CH2D2[2,:], linewidth=2.5, linestyle="--", color="blue", alpha=1.0,zorder=-2)
ax_clump.plot(D13CH3D[3,:],D12CH2D2[3,:], linewidth=2.5, linestyle=(5,(10,3)),color="orange", alpha=1.0,zorder=-2)
ax_clump.plot(D13CH3D[4,:],D12CH2D2[4,:], linewidth=2.5, color="red", alpha=1.0,zorder=-2)
ax_clump.errorbar(ANME2d["D13CH3D"],ANME2d["D12CH2D2"],xerr=ANME2d["cdse"],yerr=ANME2d["ddse"], markersize=18,label=r'Nitrate-dependent AOM (this study)', fmt='o', 
        markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_clump.errorbar(AOM_P["D13CH3D"],AOM_P["D12CH2D2"],xerr=AOM_P["cdse"],yerr=AOM_P["ddse"], markersize=14,label=r'Sulfate-dependent AOM', fmt='s', 
        markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_clump.errorbar(NC10["D13CH3D"], NC10["D12CH2D2"],xerr=NC10["cdse"],yerr=NC10["ddse"], markersize=18,label=r'NC10 (this study)', fmt='o', 
        markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax_clump.errorbar(AeOM_P["D13CH3D"], AeOM_P["D12CH2D2"],xerr=AeOM_P["cdse"],yerr=AeOM_P["ddse"], markersize=18,label=r'AeOM', fmt='o', 
        markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-2)
set_axis(ax_clump,'$\Delta^{13}$CH$_3$D (\u2030)','$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
ax_clump.xaxis.set_minor_locator(MultipleLocator(2))
ax_clump.yaxis.set_minor_locator(MultipleLocator(4))
ax_clump.legend(fontsize=18)

fig_f1,ax_f1=plt.subplots(figsize=(12,6))
fig_f2,ax_f2=plt.subplots(figsize=(12,6))
fig_f3,ax_f3=plt.subplots(figsize=(12,6))
fig_f4,ax_f4=plt.subplots(figsize=(12,6))
def plotf(x,n,ne,ax):
    ax.plot(fCH4[0,:],x[0,:],linewidth=2.5,linestyle="-.", color="black")
    ax.plot(fCH4[1,:],x[1,:],linewidth=2.5,linestyle=":",color="purple")
    ax.plot(fCH4[2,:],x[2,:],linewidth=2.5,linestyle="--",color="blue")
    ax.plot(fCH4[3,:],x[3,:],linewidth=2.5,linestyle=(5,(10,3)),color="orange")
    ax.plot(fCH4[4,:],x[4,:],linewidth=2.5,color="red")
    ax.errorbar(ANME2d["f"],ANME2d[n],xerr=ANME2d["fse"],yerr=ANME2d[ne],markerfacecolor="yellow",
               markersize=18,fmt="o",markeredgecolor="black",markeredgewidth=2.5)
    ax.errorbar(AOM_P["f"],AOM_P[n],xerr=AOM_P["fse"],yerr=AOM_P[ne],markerfacecolor="white",
               markersize=14,fmt="s",markeredgecolor="black",markeredgewidth=2.5)
    ax.errorbar(AOM_wegener["f"],AOM_wegener[n],xerr=AOM_wegener["fse"],yerr=AOM_wegener[ne],markerfacecolor="white",
               markersize=14,fmt="o",markeredgecolor="black",markeredgewidth=2.5)
    ax.errorbar(AOM_ono["f"],AOM_ono[n],xerr=AOM_ono["fse"],yerr=AOM_ono[ne],markerfacecolor="white",
               markersize=14,fmt="^",markeredgecolor="black",markeredgewidth=2.5)
    ax.invert_xaxis()

plotf(d13C,"d13C","cse",ax_f1)
set_axis(ax_f1,r"$f$",'$\delta^{13}$C (\u2030)')
ax_f1.set_xlim([1.04,-0.04])
ax_f1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax_f1.yaxis.set_minor_locator(MultipleLocator(4))
plotf(dD,"dD","dse",ax_f2)
set_axis(ax_f2,r"$f$",'$\delta$D (\u2030)')
ax_f2.set_ylim([-220,350])
ax_f2.set_xlim([1.04,-0.04])
ax_f2.xaxis.set_minor_locator(MultipleLocator(0.1))
ax_f2.yaxis.set_minor_locator(MultipleLocator(40))
plotf(D13CH3D,"D13CH3D","cdse",ax_f3)
set_axis(ax_f3,r"$f$",'$\Delta^{13}$CH$_3$D (\u2030)')
ax_f3.set_xlim([1.04,-0.04])
ax_f3.xaxis.set_minor_locator(MultipleLocator(0.1))
ax_f3.yaxis.set_minor_locator(MultipleLocator(2))
plotf(D12CH2D2,"D12CH2D2","ddse",ax_f4)
set_axis(ax_f4,r"$f$",'$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
ax_f4.set_xlim([1.04,-0.04])
ax_f4.xaxis.set_minor_locator(MultipleLocator(0.1))
ax_f4.yaxis.set_minor_locator(MultipleLocator(10))

if output==True:
    fig_bulk.savefig("bulk_model.pdf",bbox_inches="tight")
    fig_clump.savefig("clump_model.pdf",bbox_inches="tight")
    fig_f1.savefig("model_f1.pdf",bbox_inches="tight")
    fig_f2.savefig("model_f2.pdf",bbox_inches="tight")
    fig_f3.savefig("model_f3.pdf",bbox_inches="tight")
    fig_f4.savefig("model_f4.pdf",bbox_inches="tight")
