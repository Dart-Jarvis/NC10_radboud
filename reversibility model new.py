# This is a model to model the reversibility of AOM
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

output=True

dDH2O = -50.0 # permil dD_H2O
RVPDB = 0.0112372 # Standard carbon isotope ratio (VPDB)
RVSMOW = 1.5576e-4 # Standard hydrogen isotope ratio (VSMOW)
RH2O=RVSMOW*(dDH2O/1000+1) # D/H ratio in water
FH2O=RH2O/(1+RH2O)

# Define KFF and EFFs from Gropp et al., 2021; Wegner et al., 2022; and this study
# The forward KFF of the first step is defined by the experimental data (this might change after we get the ab initio result)
# a1cff= 0.9745 # Carbon isotope fractionation, use the net isotope fractionation of carbon in ANME experiment， 0.9745
# a1dffp= 0.4098 # primary hydrogen isotope fractionation, net is 0.7819, this value is adopted from Scheller et al., 2013
# a1dffs= (0.7819-a1dffp/4)*(4/3) # secondary hydrogen isotope fractionation
# gammaCD=0.72 # Needs to be much smaller than 1 in order to fit the data
# a1cdffp=a1dffp*a1cff*gammaCD # primary clumped isotopologue 13CD fractionation factor, net is 0.7559
# a1cdffs= (0.7559-a1cdffp/4)*(4/3) # secondary clumped isotopologue 13CD fractionation factor
# gammaDD=0.62 # Needs to be much smaller than 1 in order to fit the data
# a1ddffp=a1dffp*a1dffs*gammaDD # primary DD fractionation factor, net is 0.5925
# a1ddffs= (0.5925-a1ddffp/2)*2 # secondary DD fractionation factor
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# The experimental results from Scheller 2013, methane activation by mcr
a1cff= 1/1.039 # Derived Carbon isotope effect on methane formation
a1dffp= 1/2.44 # primary hydrogen isotope fractionation, KIE 2.44+-0.22, this value is adopted from Scheller et al., 2013
a1dffs= 1/1.17 # secondary hydrogen isotope fractionation, KIE 1.17+-0.05
a1dfnet=1/4*a1dffp+3/4*a1dffs
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
gamma2cd=1.000
gamma2dd=1.000
a2cff=0.979
a2dff=1.00
a2cdff=gamma2cd*a2cff*a2dff
a2ddff=gamma2dd*a2dff**2

# reversibility of cross-membrane transport
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

R=np.zeros(17) # plus D and H pools
dRdt=np.zeros(17)
for i in range(5):    
    R[i]=abundance[i]/sum(abundance)
# Normalize the abundance

# Abundance of all relevant CH3-SCoM isotopologues 12CH3, 13CH3, 12CH2D, 13CH2D, 12CHD2
# Assume the same abundance as CH4
R[5:10]=R[0:5]
R[12:17]=R[0:5]*10 # Methane isotopologue abundances outside, assuming the reservoir is 10 times larger
# D and H are in equilibrium with water
R[10]=1.0/(1.0+RHSCOB) # H
R[11]=(1-R[10]) # D
R0=R

# Construct ode
def dfdt(t,R,J):
    J1f,J1b=J     # unpack fluxes
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
    J2f=Jf-Jb # Steady state for CH3 species
    fch3f=R[5]+R[6]*a2cff+R[7]*a2dff+R[8]*a2cdff+R[9]*a2ddff # Partition of the fluxes
    Jin=(Jf-Jb)/(1-rev_tr)
    Jout=(Jf-Jb)*rev_tr/(1-rev_tr)
    fch4in=sum(R[12:17]) # Total methane abundance outside
    fch4out=sum(R[0:5]) # Total methane abundance inside
    dRdt[0]=Jin*R[12]/fch4in-Jout*R[0]/fch4out-J1f*R[0]+J1b*R[5]*R[10] # 12CH4
    dRdt[1]=Jin*R[13]/fch4in-Jout*R[1]/fch4out-J1f*R[1]*a1cff+J1b*R[6]*a1cfb*R[10] # 13CH4
    dRdt[2]=Jin*R[14]/fch4in-Jout*R[2]/fch4out-1/4*J1f*R[2]*a1dffp-3/4*J1f*R[2]*a1dffs+J1b*R[5]*R[11]*a1dfbp+J1b*R[7]*R[10]*a1dfbs # 12CH3D
    dRdt[3]=Jin*R[15]/fch4in-Jout*R[3]/fch4out-1/4*J1f*R[3]*a1cdffp-3/4*J1f*R[3]*a1cdffs+J1b*R[6]*R[11]*a1cdfbp+J1b*R[8]*R[10]*a1cdfbs # 13CH3D
    dRdt[4]=Jin*R[16]/fch4in-Jout*R[4]/fch4out-1/2*J1f*R[4]*a1ddffp-1/2*J1f*R[4]*a1ddffs+J1b*R[7]*R[11]*a1ddfbp+J1b*R[9]*R[10]*a1ddfbs #12CH2D2
    dRdt[5]=J1f*R[0]+1/4*J1f*R[2]*a1dffp-J1b*R[5]*R[11]*a1dfbp-J1b*R[5]*R[10]-J2f*R[5]/fch3f # 12CH3
    dRdt[6]=J1f*R[1]*a1cff+1/4*J1f*R[3]*a1cdffp-J1b*R[6]*R[11]*a1cdfbp-J1b*R[6]*R[10]*a1cfb-J2f*R[6]*a2cff/fch3f # 13CH3
    dRdt[7]=3/4*J1f*R[2]*a1dffs+1/2*J1f*R[4]*a1ddffp-J1b*R[7]*R[10]*a1dfbs-J1b*R[7]*R[11]*a1ddfbp-J2f*R[7]*a2dff/fch3f # 12CH2D
    dRdt[8]=3/4*J1f*R[3]*a1cdffs-J1b*R[8]*R[10]*a1cdfbs-J2f*R[8]*a2cdff/fch3f # 13CH2D
    dRdt[9]=1/2*J1f*R[4]*a1ddffs-J1b*R[9]*R[10]*a1ddfbs-J2f*R[9]*a2ddff/fch3f # 12CHD2
    dRdt[10]=(J1f*R[0]+J1f*R[1]*a1cff
    +3/4*J1f*R[2]*a1dffs+3/4*J1f*R[3]*a1cdffs+1/2*J1f*R[4]*a1ddffs
    -J1b*R[5]*R[10]-J1b*R[6]*R[10]*a1cfb
    -J1b*R[7]*R[10]*a1dfbs-J1b*R[8]*R[10]*a1cdfbs-J1b*R[9]*R[10]*a1ddfbs
    -Jfhscob*R[10]/fhscobf+Jbhscob*(1-FH2O)/fhscobb) # HS-CoB sink to H2O
    dRdt[11]=(1/4*J1f*R[2]*a1dffp+1/4*J1f*R[3]*a1cdffp+1/2*J1f*R[4]*a1ddffp
    -J1b*R[5]*R[11]*a1dfbp-J1b*R[6]*R[11]*a1cdfbp-J1b*R[7]*R[11]*a1ddfbp
    -Jfhscob*R[11]*ahscobf/fhscobf+Jbhscob*FH2O*ahscobb/fhscobb)
    dRdt[12]=-Jin*R[12]/fch4in+Jout*R[0]/fch4out
    dRdt[13]=-Jin*R[13]/fch4in+Jout*R[1]/fch4out
    dRdt[14]=-Jin*R[14]/fch4in+Jout*R[2]/fch4out
    dRdt[15]=-Jin*R[15]/fch4in+Jout*R[3]/fch4out
    dRdt[16]=-Jin*R[16]/fch4in+Jout*R[4]/fch4out
    return dRdt

# Define a separate function to calculate the evolution of 
def jb_jf_ratio(t, R, J):
    J1f, J1b = J

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
def model_rev(rev, tmax, num):
    J1f_input=1.0
    J1b_input=J1f_input*rev
    J=[J1f_input,J1b_input]
    tint=np.linspace(t_lower,tmax,num) 
    solution=solve_ivp(dfdt,(t_lower,tmax),R0,args=(J,),t_eval=tint,atol=1.0e-12,rtol=1.0e-9)
    tot_sol,fCH4_sol,d13C_sol,dD_sol,D13CH3D_sol,D12CH2D2_sol=process(solution)
    rev_sol = np.array([jb_jf_ratio(t, R, J) for t, R in zip(solution.t, solution.y.T)])
    return tot_sol,fCH4_sol,rev_sol,d13C_sol,dD_sol,D13CH3D_sol,D12CH2D2_sol

rev_list=[0.0,0.2,0.3,0.45,0.86]
t_lower=0.000 # minimum time for time interval
time_list=[8.5,11.0,13.0,16.0,68.0]
num=100000
tot=np.zeros([len(rev_list),num])
fCH4=np.zeros([len(rev_list),num])
rev=np.zeros([len(rev_list),num])
d13C=np.zeros([len(rev_list),num])
dD=np.zeros([len(rev_list),num])
D13CH3D=np.zeros([len(rev_list),num])
D12CH2D2=np.zeros([len(rev_list),num])
for i in range (len(rev_list)):
    tot[i,:],fCH4[i,:],rev[i,:],d13C[i,:],dD[i,:],D13CH3D[i,:],D12CH2D2[i,:]=model_rev(rev_list[i],time_list[i],num)

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
ax_bulk.plot(d13C[0,:],dD[0,:], linewidth=2.5, color="black", linestyle="-.", alpha=1.0, label="Model, R="+str(rev_list[0]),zorder=-2)
ax_bulk.plot(d13C[1,:],dD[1,:], linewidth=2.5, linestyle=":", color="purple", alpha=1.0,label="Model, R="+str(rev_list[1]),zorder=-2)
ax_bulk.plot(d13C[2,:],dD[2,:], linewidth=2.5, linestyle="--", color="blue", alpha=1.0,label="Model, R="+str(rev_list[2]),zorder=-2)
ax_bulk.plot(d13C[3,:],dD[3,:], linewidth=2.5, linestyle=(5,(10,3)), color="orange", alpha=1.0, label="Model, R="+str(rev_list[3]), zorder=-2)
ax_bulk.plot(d13C[4,:],dD[4,:], linewidth=2.5, color="red", alpha=1.0, label="Model, R="+str(rev_list[4]), zorder=-2)
ax_bulk.errorbar(ANME2d["d13C"],ANME2d["dD"],xerr=ANME2d["cse"],yerr=ANME2d["dse"], markersize=18,label=r'N-AOM (this study)', fmt='o', 
        markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_bulk.errorbar(AOM_P["d13C"],AOM_P["dD"],xerr=AOM_P["cse"],yerr=AOM_P["dse"], markersize=14,label=r'S-AOM (Liu et al.)', fmt='s', 
        markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_bulk.errorbar(AOM_wegener["d13C"],AOM_wegener["dD"],xerr=AOM_wegener["cse"],yerr=AOM_wegener["dse"], markersize=14, label=r"S-AOM (Wegener et al.)", 
        fmt='o', markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_bulk.errorbar(AOM_ono["d13C"],AOM_ono["dD"],xerr=AOM_ono["cse"],yerr=AOM_ono["dse"], markersize=14, label=r"S-AOM (Ono et al.)",
                 fmt='^', markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
set_axis(ax_bulk,'$\delta^{13}$C (\u2030)','$\delta$D (\u2030)')
ax_bulk.xaxis.set_minor_locator(MultipleLocator(2))
ax_bulk.yaxis.set_minor_locator(MultipleLocator(20))
ax_bulk.legend(fontsize=20)

fig_clump,ax_clump=plt.subplots(figsize=(12,12))
ax_clump.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax_clump.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)


ax_clump.plot(D13CH3D[0,:],D12CH2D2[0,:], linewidth=2.5, color="black",linestyle="-.",alpha=1.0,label="Model, R="+str(rev_list[0]),zorder=-2)
ax_clump.plot(D13CH3D[1,:],D12CH2D2[1,:], linewidth=2.5, linestyle=":", color="purple", alpha=1.0,label="Model, R="+str(rev_list[1]),zorder=-2)
ax_clump.plot(D13CH3D[2,:],D12CH2D2[2,:], linewidth=2.5, linestyle="--", color="blue", alpha=1.0,label="Model, R="+str(rev_list[2]),zorder=-2)
ax_clump.plot(D13CH3D[3,:],D12CH2D2[3,:], linewidth=2.5, linestyle=(5,(10,3)),color="orange", alpha=1.0, label="Model, R="+str(rev_list[3]),zorder=-2)
ax_clump.plot(D13CH3D[4,:],D12CH2D2[4,:], linewidth=2.5, color="red", alpha=1.0, label="Model, R="+str(rev_list[4]),zorder=-2)
ax_clump.errorbar(ANME2d["D13CH3D"],ANME2d["D12CH2D2"],xerr=ANME2d["cdse"],yerr=ANME2d["ddse"], markersize=18,label=r'Nitrate-dependent AOM (this study)', fmt='o', 
        markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
ax_clump.errorbar(AOM_P["D13CH3D"],AOM_P["D12CH2D2"],xerr=AOM_P["cdse"],yerr=AOM_P["ddse"], markersize=14,label=r'Sulfate-dependent AOM (Liu et al.)', fmt='s', 
        markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
set_axis(ax_clump,'$\Delta^{13}$CH$_3$D (\u2030)','$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
ax_clump.xaxis.set_minor_locator(MultipleLocator(2))
ax_clump.yaxis.set_minor_locator(MultipleLocator(4))


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

plotf(d13C,"d13C","cse",ax_f1)
set_axis(ax_f1,r"$f$",'$\delta^{13}$C (\u2030)')
plotf(dD,"dD","dse",ax_f2)
set_axis(ax_f2,r"$f$",'$\delta$D (\u2030)')
ax_f2.set_ylim([-220,350])
plotf(D13CH3D,"D13CH3D","cdse",ax_f3)
set_axis(ax_f3,r"$f$",'$\Delta^{13}$CH$_3$D (\u2030)')
plotf(D12CH2D2,"D12CH2D2","ddse",ax_f4)
set_axis(ax_f4,r"$f$",'$\Delta^{12}$CH$_2$D$_2$ (\u2030)')

if output==True:
    fig_bulk.savefig("bulk_model.pdf",bbox_inches="tight")
    fig_clump.savefig("clump_model.pdf",bbox_inches="tight")
