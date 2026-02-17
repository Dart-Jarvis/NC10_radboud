# This is a model to model the reversibility of AOM
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

dDH2O = -50.0 # permil dD_H2O
RVPDB = 0.0112372 # Standard carbon isotope ratio (VPDB)
RVSMOW = 1.5576e-4 # Standard hydrogen isotope ratio (VSMOW)
RH2O=RVSMOW*(dDH2O/1000+1) # D/H ratio in water
FH2O=RH2O/(1+RH2O)

# Define KFF and EFFs from Gropp et al., 2021; Wegner et al., 2022; and this study
# The forward KFF of the first step is defined by the experimental data (this might change after we get the ab initio result)
a1cff= 0.9745 # Carbon isotope fractionation, use the net isotope fractionation of carbon in ANME experiment， 0.9745
a1dffp= 0.4098 # primary hydrogen isotope fractionation, net is 0.7819, this value is adopted from Scheller et al., 2013
a1dffs= (0.7819-a1dffp/4)*(4/3) # secondary hydrogen isotope fractionation
gammaCD=0.72 # 0.7
a1cdffp=a1dffp*a1cff*gammaCD # primary clumped isotopologue 13CD fractionation factor, net is 0.7559
a1cdffs= (0.7559-a1cdffp/4)*(4/3) # secondary clumped isotopologue 13CD fractionation factor
gammaDD=0.92 # 0.87
a1ddffp=a1dffp*a1dffs*gammaDD # primary DD fractionation factor, net is 0.5925
a1ddffs= (0.5925-a1ddffp/2)*2 # secondary DD fractionation factor
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# The experimental results from Scheller 2013
# a1cff= 0.9636 # Derived Carbon isotope effect on methane formation
# a1dffp= 1/2.44 # primary hydrogen isotope fractionation, net is 0.7819, this value is adopted from Scheller et al., 2013
# a1dffs= 1/1.17 # secondary hydrogen isotope fractionation
# a1dfnet=1/4*a1dffp+3/4*a1dffs
# gammaCD=1.00
# gammaCDp=1.00
# a1cdffp=a1dffp*a1cff*gammaCDp # primary clumped isotopologue 13CD fractionation factor, net is 0.7559
# a1cdffs= (gammaCD*a1cff*a1dfnet-a1cdffp/4)*(4/3) # secondary clumped isotopologue 13CD fractionation factor
# gammaDDp=1.00
# a1ddffp=a1dffp*a1dffs*gammaDDp # primary DD fractionation factor
# a1ddffs= (1/2.09-a1ddffp/2)*2 # Determined from the experiment

# Define reversibility and fluxes of the reactions
# Use partial reversibility for the first step
# For N-DAMO, we assume the first step is irreversible (this might change after we get the ab initio results)
J1f_n=1.0
r1_n=0.0
J1b_n=J1f_n*r1_n
# Second step is assumed to be irreversible, and the flux is calculated assuming a steady state
J2f_n=J1f_n-J1b_n
J_n=[J1f_n,J1b_n,J2f_n] # assuming rapid exchange of H between water and methyl group
# Sulfate is defined to have less reversibility, to get the best fit of experimental results
J1f_s=1.0
r1_s=0.5
J1b_s=J1f_s*r1_s
# Second step
J2f_s=J1f_s-J1b_s
J_s=[J1f_s,J1b_s,J2f_s]

num=100000  #number of time steps
t_lower=0.000 # minimum time for time interval
t_upper= 0.9 # maximum time for time interval, dimensionless
tint_n=np.linspace(t_lower,t_upper,num)  # num is number of time steps specified at top of program
t_lower=0.000 # minimum time for time interval
t_upper= 4.0 # maximum time for time interval, dimensionless
tint_s=np.linspace(t_lower,t_upper,num) 

# Equlibrium isotope effect of the first step, from Gropp et al., 2021 (25 degree C)
a1cfeq=1/np.exp(2.1/1000)
a1dfeqp=1/np.exp(-635.8/1000) # Primary equilibrium fractionation
a1dfeqs=1/np.exp(55.4/1000) # Secondary equilibrium fractionation
a1cdfeqp=1/np.exp(-639.5/1000) # Primary
a1cdfeqs=1/np.exp(57.1/1000) # Secondary
a1ddfeqp=1/np.exp(-598.7/1000) # Primary
a1ddfeqs=1/np.exp(107.9/1000) # Secondary

# The hydrogen of the first step is assumed to be in equilibrium with HS-COB, equilibrium fractionation from Wegener et al.
ahscobeq=0.4686 # aHSCOB-H2O=R_H2O/R_HSCOB HSCOB-->H2O
ahscobf=1.0 # Best fit value in Wegener et al.
ahscobb=ahscobf*ahscobeq
RHSCOB=RH2O/ahscobeq # D/H ratio in HS-CoB, assuming equilibrium with water
rev_hscob=0.99
# # Also consider isotopic exchange between CH3-SCoM and water, KFF and EFF from Wegener et al.
# ach3_h2oeq=0.8748 # CH3SCoM --> H2O equlibrium
# ach3_h2off=0.888 # CH3SCoM --> H2O forward
# ach3_h2ofb=ach3_h2oeq*ach3_h2off # backward

# Calculate the fractionation factors of the back reactions for mcr
a1cfb=a1cff*a1cfeq
a1dfbp=a1dffp*a1dfeqp # primary isotope effect backwards
a1dfbs=a1dffs*a1dfeqs # secondary isotope effect backwards
a1cdfbp=a1cdffp*a1cdfeqp
a1cdfbs=a1cdffs*a1cdfeqs
a1ddfbp=a1ddffp*a1ddfeqp
a1ddfbs=a1ddffs*a1ddfeqs

# The kinetic isotope effect of the second step (CH3-SCoM --> CHO-MFR), from the best-fit values in Wegener et al., 2021, Table S6
# Gamma values are set at 1
gamma2cd=1.0
gamma2dd=1.0
a2cff=0.979
a2dff=1.0
a2cdff=gamma2cd*a2cff*a2dff
a2ddff=gamma2dd*a2dff**2

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

R=np.zeros(12) # plus D and H pools
dRdt=np.zeros(12)
for i in range(5):    
    R[i]=abundance[i]/sum(abundance)
# Normalize the abundance

# Abundance of all relevant CH3-SCoM isotopologues 12CH3, 13CH3, 12CH2D, 13CH2D, 12CHD2
# Assume the same abundance as CH4
R[5:10]=R[0:5]

# D and H are in equilibrium with water
R[10]=1.0/(1.0+RHSCOB) # H
R[11]=(1-R[10]) # D
R0=R

# Construct ode
def dfdt(t,R,J):
    J1f,J1b,J2f=J     # unpack fluxes
    Jf = (J1f*R[0] + J1f*R[1]*a1cff
          + (1/4*J1f*R[2]*a1dffp + 3/4*J1f*R[2]*a1dffs)
          + (1/4*J1f*R[3]*a1cdffp + 3/4*J1f*R[3]*a1cdffs)
          + (1/2*J1f*R[4]*a1ddffp + 1/2*J1f*R[4]*a1ddffs))

    Jb = (J1b*R[5]*R[10] + J1b*R[6]*a1cfb*R[10]
          + (J1b*R[5]*R[11]*a1dfbp + J1b*R[7]*R[10]*a1dfbs)
          + (J1b*R[6]*R[11]*a1cdfbp + J1b*R[8]*R[10]*a1cdfbs)
          + (J1b*R[7]*R[11]*a1ddfbp + J1b*R[9]*R[10]*a1ddfbs))
    fhscobf=R[10]+R[11]*ahscobf
    fhscobb=(1-FH2O)+FH2O*ahscobb
    Jfhscob=(Jf-Jb)/(1-rev_hscob)
    Jbhscob=(Jf-Jb)*rev_hscob/(1-rev_hscob)
    dRdt[0]=-J1f*R[0]+J1b*R[5]*R[10] # 12CH4
    dRdt[1]=-J1f*R[1]*a1cff+J1b*R[6]*a1cfb*R[10] # 13CH4
    dRdt[2]=-1/4*J1f*R[2]*a1dffp-3/4*J1f*R[2]*a1dffs+J1b*R[5]*R[11]*a1dfbp+J1b*R[7]*R[10]*a1dfbs # 12CH3D
    dRdt[3]=-1/4*J1f*R[3]*a1cdffp-3/4*J1f*R[3]*a1cdffs+J1b*R[6]*R[11]*a1cdfbp+J1b*R[8]*R[10]*a1cdfbs # 13CH3D
    dRdt[4]=-1/2*J1f*R[4]*a1ddffp-1/2*J1f*R[4]*a1ddffs+J1b*R[7]*R[11]*a1ddfbp+J1b*R[9]*R[10]*a1ddfbs #12CH2D2
    dRdt[5]=J1f*R[0]+1/4*J1f*R[2]*a1dffp-J1b*R[5]*R[11]*a1dfbp-J1b*R[5]*R[10]-J2f*R[5] # 12CH3
    dRdt[6]=J1f*R[1]*a1cff+1/4*J1f*R[3]*a1cdffp-J1b*R[6]*R[11]*a1cdfbp-J1b*R[6]*R[10]*a1cfb-J2f*R[6]*a2cff # 13CH3
    dRdt[7]=3/4*J1f*R[2]*a1dffs+1/2*J1f*R[4]*a1ddffp-J1b*R[7]*R[10]*a1dfbs-J1b*R[7]*R[11]*a1ddfbp-J2f*R[7]*a2dff # 12CH2D
    dRdt[8]=3/4*J1f*R[3]*a1cdffs-J1b*R[8]*R[10]*a1cdfbs-J2f*R[8]*a2cdff # 13CH2D
    dRdt[9]=1/2*J1f*R[4]*a1ddffs-J1b*R[9]*R[10]*a1ddfbs-J2f*R[9]*a2ddff # 12CHD2
    dRdt[10]=(J1f*R[0]+J1f*R[1]*a1cff
    +3/4*J1f*R[2]*a1dffs+3/4*J1f*R[3]*a1cdffs+1/2*J1f*R[4]*a1ddffs
    -J1b*R[5]*R[10]-J1b*R[6]*R[10]*a1cfb
    -J1b*R[7]*R[10]*a1dfbs-J1b*R[8]*R[10]*a1cdfbs-J1b*R[9]*R[10]*a1ddfbs
    -Jfhscob*R[10]/fhscobf+Jbhscob*(1-FH2O)/fhscobb) # -1/(1-rev_hscob)*R[10]+rev_hscob/(1-rev_hscob)*(1-FH2O) # HS-CoB, including the sink to water, assuming reversibility is very close to 1
    dRdt[11]=(1/4*J1f*R[2]*a1dffp+1/4*J1f*R[3]*a1cdffp+1/2*J1f*R[4]*a1ddffp
    -J1b*R[5]*R[11]*a1dfbp-J1b*R[6]*R[11]*a1cdfbp-J1b*R[7]*R[11]*a1ddfbp
    -Jfhscob*R[11]*ahscobf/fhscobf+Jbhscob*FH2O*ahscobb/fhscobb) # -1/(1-rev_hscob)*R[11]*ahscobf+rev_hscob/(1-rev_hscob)*FH2O*ahscobb # DS-CoB, including the reaction from DS-CoB to water, here the reversibility is 0.99
    return dRdt

# Define a separate function to calculate the evolution of 
def jb_jf_ratio(t, R, J):
    J1f, J1b, J2f = J

    Jf = (J1f*R[0] + J1f*R[1]*a1cff
          + (1/4*J1f*R[2]*a1dffp + 3/4*J1f*R[2]*a1dffs)
          + (1/4*J1f*R[3]*a1cdffp + 3/4*J1f*R[3]*a1cdffs)
          + (1/2*J1f*R[4]*a1ddffp + 1/2*J1f*R[4]*a1ddffs))

    Jb = (J1b*R[5]*R[10] + J1b*R[6]*a1cfb*R[10]
          + (J1b*R[5]*R[11]*a1dfbp + J1b*R[7]*R[10]*a1dfbs)
          + (J1b*R[6]*R[11]*a1cdfbp + J1b*R[8]*R[10]*a1cdfbs)
          + (J1b*R[7]*R[11]*a1ddfbp + J1b*R[9]*R[10]*a1ddfbs))

    return Jb / Jf

# Process the data
def process(sol):
    tot=sum(sol.y[0:5]) # Total isotopologue abundance
    tot0=sum(R0[0:5]) # initial isotopologue abundance
    fCH4=tot/tot0 # Fraction of residual methane
    d13C_t=(sol.y[1]/sol.y[0]/RVPDB-1)*1000
    dD_t=(sol.y[2]/(4*sol.y[0])/RVSMOW-1)*1000
    # Calculate the stochastic distributions
    X13C=(sol.y[1]+sol.y[3])/tot # X13C 
    XD=(sol.y[2]+sol.y[3]+2*sol.y[4])/(4*tot) # XD
    X12C=1-X13C
    XH=1-XD
    sto_13CD=4*X13C*XD*XH**3/(X12C*XH**4)
    sto_DD=6*X12C*XD**2*XH**2/(X12C*XH**4)
    D13CH3D_t=(sol.y[3]/sol.y[0]/sto_13CD-1)*1000
    D12CH2D2_t=(sol.y[4]/sol.y[0]/sto_DD-1)*1000
    return tot,fCH4,d13C_t,dD_t,D13CH3D_t,D12CH2D2_t

soln=solve_ivp(dfdt,(t_lower,t_upper),R0,args=(J_n,),t_eval=tint_n,atol=1.0e-11,rtol=1.0e-9)
tot_n,fCH4_n,d13C_n,dD_n,D13CH3D_n,D12CH2D2_n=process(soln)
rev_n = np.array([jb_jf_ratio(t, R, J_n) for t, R in zip(soln.t, soln.y.T)])

soln_s=solve_ivp(dfdt,(t_lower,t_upper),R0,args=(J_s,),t_eval=tint_s,atol=1.0e-11,rtol=1.0e-9)
tot_s,fCH4_s,d13C_s,dD_s,D13CH3D_s,D12CH2D2_s=process(soln_s)
rev_s = np.array([jb_jf_ratio(t, R, J_s) for t, R in zip(soln_s.t, soln_s.y.T)])

# Plotting
# Import data
data = pd.read_csv('isotope_data.csv')
equib = pd.read_csv('equib.csv')
ANME2d=data[data['label']=='3'] # ANME-2d
# Previous AOM data
AOM_P=data[data['label']=='AOM_P']
AOM_P_LS=data[data['label']=='AOM_P_LS']
AOM_ono=data[data['label']=='AOM_ono']
AOM_wegener=data[data['label']=='AOM_Wegener']

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
ax_bulk.plot(d13C_s,dD_s, linewidth=2.5, color="blue")
ax_bulk.plot(d13C_n,dD_n, linewidth=2.5, color="orange")
ax_bulk.errorbar(ANME2d["d13C"],ANME2d["dD"],xerr=ANME2d["cse"],yerr=ANME2d["dse"], markersize=18,label=r'Nitrate-dependent AOM (this study)', fmt='o', 
        markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_bulk.errorbar(AOM_P["d13C"],AOM_P["dD"],xerr=AOM_P["cse"],yerr=AOM_P["dse"], markersize=14,label=r'Sulfate-dependent AOM (previous study)', fmt='s', 
        markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_bulk.errorbar(AOM_wegener["d13C"],AOM_wegener["dD"],xerr=AOM_wegener["cse"],yerr=AOM_wegener["dse"], markersize=14, fmt='o', 
        markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
set_axis(ax_bulk,'$\delta^{13}$C (\u2030)','$\delta$D (\u2030)')

fig_clump,ax_clump=plt.subplots(figsize=(12,12))
ax_clump.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax_clump.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)


ax_clump.plot(D13CH3D_n,D12CH2D2_n,linewidth=2.5,color="orange",label="Model, R=0")
ax_clump.plot(D13CH3D_s,D12CH2D2_s,linewidth=2.5,color="blue", label="Model, R=0.50-0.92")
ax_clump.errorbar(ANME2d["D13CH3D"],ANME2d["D12CH2D2"],xerr=ANME2d["cdse"],yerr=ANME2d["ddse"], markersize=18,label=r'Nitrate-dependent AOM (this study)', fmt='o', 
        markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_clump.errorbar(AOM_P["D13CH3D"],AOM_P["D12CH2D2"],xerr=AOM_P["cdse"],yerr=AOM_P["ddse"], markersize=14,label=r'Sulfate-dependent AOM (previous study)', fmt='s', 
        markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
set_axis(ax_clump,'$\Delta^{13}$CH$_3$D (\u2030)','$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
ax_clump.legend(fontsize=20,bbox_to_anchor=(1.05,1.0))

fig_f1,ax_f1=plt.subplots(figsize=(12,6))
fig_f2,ax_f2=plt.subplots(figsize=(12,6))
fig_f3,ax_f3=plt.subplots(figsize=(12,6))
fig_f4,ax_f4=plt.subplots(figsize=(12,6))
def plotf(x1,x2,n,ax):
    ax.plot(fCH4_n,x1,linewidth=2.5,color="orange")
    ax.plot(fCH4_s,x2,linewidth=2.5,color="blue")
    ax.scatter(ANME2d["f"],ANME2d[n])
    ax.scatter(AOM_P["f"],AOM_P[n])
    ax.scatter(AOM_wegener["f"],AOM_wegener[n])

plotf(d13C_n,d13C_s,"d13C",ax_f1)
plotf(dD_n,dD_s,"dD",ax_f2)
plotf(D13CH3D_n,D13CH3D_s,"D13CH3D",ax_f3)
plotf(D12CH2D2_n,D12CH2D2_s,"D12CH2D2",ax_f4)

