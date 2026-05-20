# This is a model to model the reversibility of AOM
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

output=False # True if you want to save the plots as pdf
ff="ab initio" # Select the KIEs in the model: 
model="no INT"
# Constant rate constants for each model run.
# Reversibility is not prescribed in this version; it is calculated dynamically as
# J1b/J1f and J2b/J2f from the evolving isotope/species abundances.
# Each tuple is: (k1f, k1b, k2f, k2b), where
# step 1 = CH4 <-> CH3-SCoM and step 2 = CH3-SCoM <-> CO2.
rate_constant_list=[
    (1.0, 0.0, 1.0, 0.0),
    (1.0, 0.2, 1.0, 0.2),
    (1.0, 0.3, 1.0, 0.3),
    (1.0, 0.5, 1.0, 0.5),
    (1.0, 0.9, 1.0, 0.9),
]
t_lower=0.000 # minimum time for time interval
time_list=[8.5,11.0,13.0,16.0,18.0] # Maximum time for time interval, relevant to the final fraction of methane left
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
    print("Net fractionation factors:")
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
        a1cdffp=0.4915
        a1cdffs=0.7983
        a1ddffp=0.4447
        a1ddffs=0.7267
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
a2dff=1.00
a2cdff=gamma2cd*a2cff*a2dff
a2ddff=gamma2dd*a2dff**2
a2ceq=1/np.exp(18.1/1000)*1/np.exp(15.8/1000)*1/np.exp(16.9/1000)*1/np.exp(-3.3/1000)*1/np.exp(1.9/1000)
a2deq=1/np.exp(42.9/1000)*((1/np.exp(81.3/1000)+1/np.exp(84.0/1000))/2)*1/np.exp(-78.2/1000)*1/np.exp(-70.5/1000)*1/np.exp(8.5/1000)
a2cfb=a2cff*a2ceq
a2dfb=a2dff*a2deq
# Back-reaction clumped fractionation factors for the lumped CH3-SCoM <-> CO2 step.
# These mirror the forward definitions and assume gamma values are unchanged.
a2cdfb=gamma2cd*a2cfb*a2dfb
a2ddfb=gamma2dd*a2dfb**2

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
# Helper functions for flux calculations
def step1_fluxes(R,k):
    k1f,k1b,k2f,k2b=k
    J1f = (k1f*R[0] + k1f*R[1]*a1cff
           + (1/4*k1f*R[2]*a1dffp + 3/4*k1f*R[2]*a1dffs)
           + (1/4*k1f*R[3]*a1cdffp + 3/4*k1f*R[3]*a1cdffs)
           + (1/2*k1f*R[4]*a1ddffp + 1/2*k1f*R[4]*a1ddffs))

    J1b = (k1b*R[5]*R[10] + k1b*R[6]*a1cfb*R[10]
           + (k1b*R[5]*R[11]*a1dfbp + k1b*R[7]*R[10]*a1dfbs)
           + (k1b*R[6]*R[11]*a1cdfbp + k1b*R[8]*R[10]*a1cdfbs)
           + (k1b*R[7]*R[11]*a1ddfbp + k1b*R[9]*R[10]*a1ddfbs))
    return J1f,J1b

def step2_fluxes(R,k):
    k1f,k1b,k2f,k2b=k

    # Forward CH3-SCoM -> CO2 absolute fluxes.
    F2_12CH3  = k2f*R[5]
    F2_13CH3  = k2f*R[6]*a2cff
    F2_12CH2D = k2f*R[7]*a2dff
    F2_13CH2D = k2f*R[8]*a2cdff
    F2_12CHD2 = k2f*R[9]*a2ddff

    # Backward CO2 -> CH3-SCoM absolute fluxes. Carbon comes from the CO2 pool;
    # H/D sites in the regenerated methyl group are supplied from water.
    XH2O = 1.0-FH2O
    XD2O = FH2O
    B2_12CH3  = k2b*R[17]*(XH2O**3)
    B2_12CH2D = k2b*R[17]*(3*XH2O**2*XD2O)*a2dfb
    B2_12CHD2 = k2b*R[17]*(3*XH2O*XD2O**2)*a2ddfb
    B2_13CH3  = k2b*R[18]*a2cfb*(XH2O**3)
    B2_13CH2D = k2b*R[18]*(3*XH2O**2*XD2O)*a2cdfb

    F2_12CO2 = F2_12CH3 + F2_12CH2D + F2_12CHD2
    F2_13CO2 = F2_13CH3 + F2_13CH2D
    B2_12CO2 = B2_12CH3 + B2_12CH2D + B2_12CHD2
    B2_13CO2 = B2_13CH3 + B2_13CH2D

    J2f = F2_12CO2 + F2_13CO2
    J2b = B2_12CO2 + B2_13CO2

    return {
        "J2f": J2f, "J2b": J2b,
        "F2_12CH3": F2_12CH3, "F2_13CH3": F2_13CH3,
        "F2_12CH2D": F2_12CH2D, "F2_13CH2D": F2_13CH2D,
        "F2_12CHD2": F2_12CHD2,
        "B2_12CH3": B2_12CH3, "B2_13CH3": B2_13CH3,
        "B2_12CH2D": B2_12CH2D, "B2_13CH2D": B2_13CH2D,
        "B2_12CHD2": B2_12CHD2,
        "F2_12CO2": F2_12CO2, "F2_13CO2": F2_13CO2,
        "B2_12CO2": B2_12CO2, "B2_13CO2": B2_13CO2,
    }

# Construct ode
def dfdt(t,R,k):
    k1f,k1b,k2f,k2b=k
    Jf,Jb=step1_fluxes(R,k)

    # Apply steady state for HS-CoB species
    fhscobf=R[10]+R[11]*ahscobf
    fhscobb=(1-FH2O)+FH2O*ahscobb
    Jfhscob=(Jf-Jb)/(1-rev_hscob)
    Jbhscob=(Jf-Jb)*rev_hscob/(1-rev_hscob)

    # Explicit reversible CH3-SCoM <-> CO2 step with fixed forward and backward
    # rate constants. Reversibility is therefore variable and is diagnosed as
    # J2b/J2f at each time point.
    flux2=step2_fluxes(R,k)

    dRdt=np.zeros_like(R)
    Jin=(Jf-Jb)/(1-rev_tr)
    Jout=(Jf-Jb)*rev_tr/(1-rev_tr)
    fch4in=sum(R[12:17]) # Total methane abundance outside
    fch4out=sum(R[0:5]) # Total methane abundance inside
    dRdt[0]=Jin*R[12]/fch4in-Jout*R[0]/fch4out-k1f*R[0]+k1b*R[5]*R[10] # 12CH4
    dRdt[1]=Jin*R[13]/fch4in-Jout*R[1]/fch4out-k1f*R[1]*a1cff+k1b*R[6]*a1cfb*R[10] # 13CH4
    dRdt[2]=Jin*R[14]/fch4in-Jout*R[2]/fch4out-1/4*k1f*R[2]*a1dffp-3/4*k1f*R[2]*a1dffs+k1b*R[5]*R[11]*a1dfbp+k1b*R[7]*R[10]*a1dfbs # 12CH3D
    dRdt[3]=Jin*R[15]/fch4in-Jout*R[3]/fch4out-1/4*k1f*R[3]*a1cdffp-3/4*k1f*R[3]*a1cdffs+k1b*R[6]*R[11]*a1cdfbp+k1b*R[8]*R[10]*a1cdfbs # 13CH3D
    dRdt[4]=Jin*R[16]/fch4in-Jout*R[4]/fch4out-1/2*k1f*R[4]*a1ddffp-1/2*k1f*R[4]*a1ddffs+k1b*R[7]*R[11]*a1ddfbp+k1b*R[9]*R[10]*a1ddfbs #12CH2D2
    dRdt[5]=k1f*R[0]+1/4*k1f*R[2]*a1dffp-k1b*R[5]*R[11]*a1dfbp-k1b*R[5]*R[10]-flux2["F2_12CH3"]+flux2["B2_12CH3"] # 12CH3
    dRdt[6]=k1f*R[1]*a1cff+1/4*k1f*R[3]*a1cdffp-k1b*R[6]*R[11]*a1cdfbp-k1b*R[6]*R[10]*a1cfb-flux2["F2_13CH3"]+flux2["B2_13CH3"] # 13CH3
    dRdt[7]=3/4*k1f*R[2]*a1dffs+1/2*k1f*R[4]*a1ddffp-k1b*R[7]*R[10]*a1dfbs-k1b*R[7]*R[11]*a1ddfbp-flux2["F2_12CH2D"]+flux2["B2_12CH2D"] # 12CH2D
    dRdt[8]=3/4*k1f*R[3]*a1cdffs-k1b*R[8]*R[10]*a1cdfbs-flux2["F2_13CH2D"]+flux2["B2_13CH2D"] # 13CH2D
    dRdt[9]=1/2*k1f*R[4]*a1ddffs-k1b*R[9]*R[10]*a1ddfbs-flux2["F2_12CHD2"]+flux2["B2_12CHD2"] # 12CHD2
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
    dRdt[17]=flux2["F2_12CO2"]-flux2["B2_12CO2"] # 12CO2
    dRdt[18]=flux2["F2_13CO2"]-flux2["B2_13CO2"] # 13CO2
    return dRdt

# Define separate functions to calculate the evolution of reversibility.
def jb_jf_ratio(t, R, k):
    Jf,Jb=step1_fluxes(R,k)
    return Jb/Jf if Jf != 0 else np.nan

def j2b_j2f_ratio(t, R, k):
    flux2=step2_fluxes(R,k)
    return flux2["J2b"]/flux2["J2f"] if flux2["J2f"] != 0 else np.nan

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

# Define a function to model isotope evolution with fixed rate constants.
def model_rate_constants(k1f, k1b, k2f, k2b, tmax, num):
    k=[k1f,k1b,k2f,k2b]
    tint=np.linspace(t_lower,tmax,num) 
    solution=solve_ivp(dfdt,(t_lower,tmax),R0,args=(k,),t_eval=tint,atol=1.0e-12,rtol=1.0e-9)
    tot_sol,fCH4_sol,d13C_sol,dD_sol,D13CH3D_sol,D12CH2D2_sol=process(solution)
    rev1_sol = np.array([jb_jf_ratio(t, R, k) for t, R in zip(solution.t, solution.y.T)])
    rev2_sol = np.array([j2b_j2f_ratio(t, R, k) for t, R in zip(solution.t, solution.y.T)])
    return tot_sol,fCH4_sol,rev1_sol,rev2_sol,d13C_sol,dD_sol,D13CH3D_sol,D12CH2D2_sol

if len(time_list) != len(rate_constant_list):
    raise ValueError("time_list must have the same length as rate_constant_list.")

n_models=len(rate_constant_list)
tot=np.zeros([n_models,num])
fCH4=np.zeros([n_models,num])
rev1=np.zeros([n_models,num])
rev2=np.zeros([n_models,num])
d13C=np.zeros([n_models,num])
dD=np.zeros([n_models,num])
D13CH3D=np.zeros([n_models,num])
D12CH2D2=np.zeros([n_models,num])
for i in range(n_models):
    k1f_i,k1b_i,k2f_i,k2b_i=rate_constant_list[i]
    tot[i,:],fCH4[i,:],rev1[i,:],rev2[i,:],d13C[i,:],dD[i,:],D13CH3D[i,:],D12CH2D2[i,:]=model_rate_constants(k1f_i,k1b_i,k2f_i,k2b_i,time_list[i],num)

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
ax_bulk.plot(d13C[0,:],dD[0,:], linewidth=2.5, color="black", linestyle="-.", alpha=1.0, label="Model, k="+str(rate_constant_list[0]),zorder=-2)
ax_bulk.plot(d13C[1,:],dD[1,:], linewidth=2.5, linestyle=":", color="purple", alpha=1.0,label="Model, k="+str(rate_constant_list[1]),zorder=-2)
ax_bulk.plot(d13C[2,:],dD[2,:], linewidth=2.5, linestyle="--", color="blue", alpha=1.0,label="Model, k="+str(rate_constant_list[2]),zorder=-2)
ax_bulk.plot(d13C[3,:],dD[3,:], linewidth=2.5, linestyle=(5,(10,3)), color="orange", alpha=1.0, label="Model, k="+str(rate_constant_list[3]), zorder=-2)
ax_bulk.plot(d13C[4,:],dD[4,:], linewidth=2.5, color="red", alpha=1.0, label="Model, k="+str(rate_constant_list[4]), zorder=-2)


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
