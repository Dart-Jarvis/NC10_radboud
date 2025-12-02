import matplotlib.pyplot as plt
import math as math
from math import sin as sin
from math import cos as cos
from math import asin as asin
from math import acos as acos
from math import sqrt as sqrt
import numpy as np
from numpy import log as ln
from numpy import log10 as log
from numpy import exp as exp
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import time

print(" ")
print(" ")
print("   ----------------------------------------------------------")
print("          UCLA AOM ISOTOPOLOGUE KINETIC MODEL - SIMPLE         ")
print("   ----------------------------------------------------------")
print(" ")
#------------------------------------------------------------------------------------------------
#INPUTS
nmolec=1.0e10
num=5000  # number of time steps

reversible=True # True
factor=6e-10 # e.g., 4.8e-6, k_reverse rate constant relative to oxidation rate constant
t_lower=0.0001 # minimum time for time interval
t_upper=1 # maximum time for time interval, dimensionless

Tkinetics=293.15 # 20 C, T for AOM kinetics and re-equilibration in K
T=200.0 # Equilibration temperature of initial methane in C, OVERRIDEN BY INPUT FILE
TK=T+273.15 # Equilibration temperature of initial methane in K

print('')
print('Number of time steps =',num)
print('k_rev/k = ',factor)
print('T AOM = %.3f K' %Tkinetics)

#INITIALIZE ARRAYS
dD_CH4_t=np.zeros(num)
d13C_CH4_t=np.zeros(num)
d13CH3D_t=np.zeros(num)
D13CH3D_t=np.zeros(num)
d12CH2D2_t=np.zeros(num)
D12CH2D2_t=np.zeros(num)
D_H_CH4=np.zeros(num)
C13_C12_CH4=np.zeros(num)
xH_CH4=np.zeros(num)
xD_CH4=np.zeros(num)
x12C_CH4=np.zeros(num)
x13C_CH4=np.zeros(num)

d13C_CH3_t=np.zeros(num)
dD_CH3_t=np.zeros(num)
xCH3_t=np.zeros(num)
xCH2D_t=np.zeros(num)
xCHD2_t=np.zeros(num)

# fractions of isotopologues, stochastic and equilibrium, for a single CH4 composition.
nCH4=10
xst_CH4=np.zeros(nCH4)
xeq_CH4=np.zeros(nCH4)
nCH3=6
xst_CH3=np.zeros(nCH3)
xeq_CH3=np.zeros(nCH3)

# Standard reference values
VSMOW=0.00015576
VPDB=0.0112372

# Isotopologue mole fraction names for all species
xnames=[
"12CH4",
"13CH4",
"12CH3D",
"13CH3D",
"12CH2D2",
"13CH2D2",
"12CHD3",
"13CHD3",
"12CD4",
"13CD4",
"12CH3",
"13CH3",
"12CH2D",
"13CH2D",
"12CHD2",
"13CHD2",
"H",
"D"]

# DEFINE FUNCTIONS FOR CALCULATING ISOTOPOLOUGE ABUNDANCES FOR CH4 AND CH3
#------------------------------------------------------------------------------------------------
def CH4_isotopologues(dD,d13C,TK):
# Calculates mole fractions of isotopologues of CH4 from input bulk isotope ratios
# for stochastic distribution and for a specified temperature TK in Kelvin.
# The order of isotopologue fractions, vector Y, is set here as well.
    VSMOW=0.00015576
    VPDB=0.0112372
    D_H=VSMOW*(dD/1000.0+1.0)
    C13_C12=VPDB*(d13C/1000.0+1.0)
    xH=1.0/(1.0+D_H)
    xD=D_H/(1.0+D_H)
    x12C=1.0/(1.0+C13_C12)      #x12C
    x13C=C13_C12/(1.0+C13_C12)  #x13C
    xst_CH4[0]=(xH**4)*x12C                 #x12CH4 =0
    xst_CH4[1]=(xH**4)*x13C                 #x13CH4 =1
    xst_CH4[2]=4.0*(xH**3)*xD*x12C          #x12CH3D =2
    xst_CH4[3]=4.0*(xH**3)*xD*x13C          #x13CH3D =3
    xst_CH4[4]=6.0*(xH**2)*(xD**2.0)*x12C   #x12CH2D2 =4
    xst_CH4[5]=6.0*(xH**2)*(xD**2.0)*x13C   #x13CH2D2 =5
    xst_CH4[6]=4.0*xH*(xD**3)*x12C          #x12CHD3 =6
    xst_CH4[7]=4.0*xH*(xD**3)*x13C          #x13CHD3 =7
    xst_CH4[8]=(xD**4)*x12C                 #x12CD4 =8
    xst_CH4[9]=(xD**4)*x13C                 #x13CD4 =9
    G5=1/TK
    alpha13D=1+0.03555020*G5-433.038*G5**2+1.27021e6*G5**3-5.94804e8*G5**4+1.19663e11*G5**5-9.0723e12*G5**6
    alphaD2=1+0.183798*G5-785.483*G5**2+1.056280e6*G5**3+9.37307e7*G5**4-8.91948e10*G5**5+9.90173e12*G5**6
    i=0
    while i < len(xst_CH4):
        xeq_CH4[i]=xst_CH4[i]
        i=i+1
    xeq_CH4[3]=xst_CH4[3]*alpha13D
    xeq_CH4[4]=xst_CH4[4]*alphaD2
    return xH,xD,x12C,x13C,xst_CH4,xeq_CH4,alpha13D,alphaD2
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
def CH3_isotopologues(dD,d13C):
# Calculates mole fractions of isotopologues of CH3 from input bulk isotope ratios
# for stochastic distribution.
    VSMOW=0.00015576
    VPDB=0.0112372
    D_H=VSMOW*(dD/1000.0+1.0)
    C13_C12=VPDB*(d13C/1000.0+1.0)
    xH=1.0/(1.0+D_H)
    xD=D_H/(1.0+D_H)
    x12C=1.0/(1.0+C13_C12)      #x12C
    x13C=C13_C12/(1.0+C13_C12)  #x13C
    xst_CH3[0]=(xH**3)*x12C                 #x12CH3 =0
    xst_CH3[1]=(xH**3)*x13C                 #x13CH3 =1
    xst_CH3[2]=3.0*(xH**2)*xD*x12C          #x12CH2D =2
    xst_CH3[3]=3.0*(xH**2)*xD*x13C          #x13CH2D =3
    xst_CH3[4]=3.0*(xH)*(xD**2.0)*x12C      #x12CHD2 =4
    xst_CH3[5]=3.0*(xH)*(xD**2.0)*x13C      #x13CHD2 =5
    i=0
    while i < len(xst_CH3):
        xeq_CH3[i]=xst_CH3[i]
        i=i+1
    return xst_CH3,xeq_CH3
#------------------------------------------------------------------------------------------------
# INITIALIZE COMPOSITION OF METHANE

#READ composition of initial CH4 from file
print('')
print('Opening AOM_kinetics_input.txt...')
name="AOM_kinetics_input.txt"
count = len(open(name).readlines())
print('  Input file length = ', count)
values=np.genfromtxt(name,'float')  # Read from file into values
ny=12
Y0=np.zeros(ny) # Assign zero to all initial abundances, then
for i in range(0,5):
    Y0[i]=nmolec*values[i]  # Assign first 5 CH4 relative abundances to Y0, put arbitrary multiplier here
# Calculate compositional parameters for input CH4
d13C_CH4=1000.0*((Y0[1]/Y0[0])/VPDB -1.0)
dD_CH4=1000.0*(((Y0[2]/Y0[0])/4.0)/VSMOW -1.0)
ratio_i_CH2D2_CH4=Y0[4]/Y0[0]
ratio_i_13CH3D_CH4=Y0[3]/Y0[0]
xH,xD,x12C,x13C,xst_CH4,xeq_CH4,alpha13D,alphaD2 = CH4_isotopologues(dD_CH4, d13C_CH4, TK)
D12CH2D2_CH4=1000.0*(ratio_i_CH2D2_CH4/(xst_CH4[4]/xst_CH4[0])-1.0)
D13CH3D_CH4=1000.0*(ratio_i_13CH3D_CH4/(xst_CH4[3]/xst_CH4[0])-1.0)
# Replace equilibrium composition of initial CH4 with the actual values input from file
xeq_CH4=values
ynames=[
"12CH4  ",
"13CH4  ",
"12CH3D ",
"13CH3D ",
"12CH2D2",
"12CH3  ",
"13CH3  ",
"12CH2D ",
"13CH2D ",
"12CHD2 ",
"H      ",
"D      "]
print('')
print('INITIAL CONDITION VECTOR (moles)')
i=0
while i < len(Y0):
    print("%i. %s \t = %.6e" %(i, ynames[i],Y0[i]))
    i=i+1
print('')
print("CH4 INITIAL COMPOSITION:")
print("   d13C\t =",d13C_CH4)
print("   dD \t =",dD_CH4)
print("   D12CH2D2\t =",D12CH2D2_CH4)
print("   D13CH3D\t =",D13CH3D_CH4)
print("   xH\t =",xH)
print("   xD\t =",xD)
print("   x12C\t =",x12C)
print("   x13C\t =",x13C)
print('')
print("Isotopologue   \t Stochastic \t Actual")
i=0
while i < len(xst_CH4):
    print("   %s \t %.4e \t %.4e  " %(xnames[i],xst_CH4[i],xeq_CH4[i]))
    i=i+1
ratio_i_CH2D2_CH4=xeq_CH4[4]/xeq_CH4[0]
ratio_i_13CH3D_CH4=xeq_CH4[3]/xeq_CH4[0]

# SETUP 16 REACTIONS USED TO DESCRIBE THE KINETICS
rxn_names=[
"CH4 = CH3 + H  ",
"CH3D = CH3 + D ",
"CH3D = CH2D + H",
"CH2D2 = CH2D + D",
"CH2D2 = CHD2 + H",
"13CH4 = 13CH3 + H",
"13CH3D = 13CH3 +D",
"13CH3D = 13CH2D+H",
"CH3 + H = CH4  ",
"CH3 + D = CH3D ",
"CH2D + H = CH3D",
"CH2D + D = CH2D2",
"CHD2 + H = CH2D2",
"13CH3 + H = 13CH4",
"13CH3 + D = 13CH3D",
"13CH2D + H = 13CH3D"
]
#------------------------------------------------------------------------------------------------
# FRACTIONATON FACTORS
nrxns=16
K=np.zeros(nrxns)
Alpha=np.ones(nrxns)

#Effective alpha and gamma from the Svalbard incubations in this study
#Primary vs. secondary isotope effects
# pMMO inhibitor 0.9548 0.5977	0.5564	0.2468 0.9750 0.6908 
# NC10+ANME 0.9779	0.8023	0.7832	0.6238	0.9983	0.9691
Alpha_D_primary = 0.8177 #arbitrary value, closed to the one reported by Scheller et al. (2013)
Alpha_D_eff = 0.8177
Alpha_D2_eff = 0.667
Alpha_13_eff = 0.9762
Alpha_13D_eff = 0.7952
Gamma_13D_eff = 0.9962
Gamma_D2_eff = 0.97756

Alpha[0]=1.0
Alpha[1]=Alpha_D_primary                   #CH3D  --> D
Alpha[2]=(Alpha_D_eff*4-Alpha[1])/3        #CH3D  --> H
Alpha[3]=Alpha[1]*Alpha[2]*Gamma_D2_eff    #CH2D2 --> D
Alpha[4]=Alpha_D2_eff*2-Alpha[3]           #CH2D2 --> H
Alpha[5]=Alpha_13_eff                      #13CH4  --> H
Alpha[6]=Alpha[1]*Alpha[5]*Gamma_13D_eff   #13CH3D  --> D
Alpha[7]=(Alpha_13D_eff*4-Alpha[6])/3      #13CH3D  --> H

#Equilibrium isotope effects for clumped isotopes
G5=1/Tkinetics
alpha13Dkinetics=1.0+0.03555020*G5-433.038*G5**2+1.27021e6*G5**3-5.94804e8*G5**4+1.19663e11*G5**5-9.0723e12*G5**6
alphaD2kinetics=1.0+0.183798*G5-785.483*G5**2+1.056280e6*G5**3+9.37307e7*G5**4-8.91948e10*G5**5+9.90173e12*G5**6

#Equilibrium isotope effects for both bulk and clumped isotopes
alpha_13CH4_eq = 1.0/np.exp(2.1/1000)        #Data from Groop et al. (2021, GCA) @ 25 C
alpha_12CH3D_P_eq = 1.0/np.exp(195.3/1000)
alpha_12CH3D_S_eq = 1.0/np.exp(55.4/1000)
gamma_13CH3D_P_eq = alpha13Dkinetics
gamma_13CH3D_S_eq = 1.0/0.9998
gamma_12CH2D2_P_eq = alphaD2kinetics
gamma_12CH2D2_S_eq = 1.0/0.9972

if reversible:
    Alpha[8]=Alpha[0]                                                              #for reversibility
    Alpha[9]=Alpha[1]*alpha_12CH3D_P_eq                                            #for reversibility
    Alpha[10]=Alpha[2]*alpha_12CH3D_S_eq                                           #for reversibility
    Alpha[11]=Alpha[3]*(gamma_12CH2D2_P_eq*alpha_12CH3D_P_eq*alpha_12CH3D_S_eq)    #for reversibility
    Alpha[12]=Alpha[4]*(gamma_12CH2D2_S_eq*alpha_12CH3D_S_eq*alpha_12CH3D_S_eq)    #for reversibility
    Alpha[13]=Alpha[5]*alpha_13CH4_eq                                              #for reversibility
    Alpha[14]=Alpha[6]*(gamma_13CH3D_P_eq*alpha_13CH4_eq*alpha_12CH3D_P_eq)        #for reversibility
    Alpha[15]=Alpha[7]*(gamma_13CH3D_S_eq*alpha_13CH4_eq*alpha_12CH3D_S_eq)        #for reversibility

print('')
i=0
while i < len(Alpha):
    print("Alpha for reaction %d.\t %s\t= %.6f" %(i,rxn_names[i],Alpha[i]))
    i=i+1

#------------------------------------------------------------------------------------------------
# SETUP ODEs OF FORM dY_i/dt = sum(K[R1][R2]) where sum is over the 16 reactions in which species i
# participates.
ynames=[
"12CH4  ",
"13CH4  ",
"12CH3D ",
"13CH3D ",
"12CH2D2",
"12CH3  ",
"13CH3  ",
"12CH2D ",
"13CH2D ",
"12CHD2 ",
"H      ",
"D      "]
ny=12
Y=np.zeros(ny)
Y0=np.zeros(ny) # Assign zero to all initial abundances, then
Y0[0:5]=nmolec*xeq_CH4[0:nCH4-5]  # Assign first 5 CH4 relative abundances to Y0, could put arbitrary multiplier here
# print('')

dYdt=np.zeros(ny) # Vector of dY/dt values evaluated in function below.

# The FUNCTION for the right-hand-side of each ordinary differential equation is here.
# The values returned are the rates for each species Y_i.
#------------------------------------------------------------------------------------------------
def dfdt(t,Y):
    #factor=0.0001  #0.0075
    K_oxidize=1.0
    K_synth=factor*K_oxidize
    K[0]=K_oxidize*Alpha[0]
    K[1]=1/4*K_oxidize*Alpha[1]
    K[2]=3/4*K_oxidize*Alpha[2]
    K[3]=0.5*K_oxidize*Alpha[3]
    K[4]=0.5*K_oxidize*Alpha[4]
    K[5]=K_oxidize*Alpha[5]
    K[6]=1/4*K_oxidize*Alpha[6]
    K[7]=3/4*K_oxidize*Alpha[7]
    K[8]=0.5*K_synth*Alpha[8]
    K[9]=0.5*K_synth*Alpha[9]
    K[10]=0.5*K_synth*Alpha[10]
    K[11]=0.5*K_synth*Alpha[11]
    K[12]=1.0*K_synth*Alpha[12]
    K[13]=0.5*K_synth*Alpha[13]
    K[14]=0.5*K_synth*Alpha[14]
    K[15]=1.0*K_synth*Alpha[15]
    
    dYdt[0]=-K[0]*Y[0]+K[8]*Y[5]*Y[10]
    dYdt[1]=-K[5]*Y[1]+K[13]*Y[6]*Y[10]
    dYdt[2]=-K[2]*Y[2]-K[1]*Y[2]+K[10]*Y[7]*Y[10]+K[9]*Y[5]*Y[11]
    dYdt[3]=-K[6]*Y[3]-K[7]*Y[3]+K[14]*Y[6]*Y[11]+K[15]*Y[8]*Y[10]
    dYdt[4]=-K[3]*Y[4]-K[4]*Y[4]+K[11]*Y[7]*Y[11]+K[12]*Y[9]*Y[10]
    dYdt[5]=-K[8]*Y[5]*Y[10]-K[9]*Y[5]*Y[11]+K[0]*Y[0]+K[1]*Y[2]
    dYdt[6]=-K[13]*Y[6]*Y[10]-K[14]*Y[6]*Y[11]+K[5]*Y[1]+K[6]*Y[3]
    dYdt[7]=-K[10]*Y[7]*Y[10]-K[11]*Y[7]*Y[11]+K[2]*Y[2]+K[3]*Y[4]
    dYdt[8]=-K[15]*Y[8]*Y[10]+K[7]*Y[3]
    dYdt[9]=-K[12]*Y[9]*Y[10]+K[4]*Y[4]
    dYdt[10]=-K[8]*Y[10]*Y[5]-K[10]*Y[10]*Y[7]-K[12]*Y[10]*Y[9]-K[13]*Y[10]*Y[6]-K[15]*Y[10]*Y[8] \
        +K[0]*Y[0]+K[2]*Y[2]+K[4]*Y[4]+K[5]*Y[1]+K[7]*Y[3]
    dYdt[11]=-K[9]*Y[11]*Y[5]-K[11]*Y[11]*Y[7]-K[14]*Y[11]*Y[6]+K[1]*Y[2]+K[3]*Y[4]+K[6]*Y[3]
    return dYdt
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
# SOLVE ODEs. Returned array Y has each element being an array for variable i at each time
tint=np.linspace(t_lower,t_upper,num)  # num is number of time steps specified at top of program
soln=solve_ivp(dfdt,(t_lower,t_upper),Y0,t_eval=tint,atol=1.0e-11,rtol=1.0e-9)
#------------------------------------------------------------------------------------------------

# PROCESS RESULTS INTO IDENTIFIABLE ISOTOPOLOGUE PARAMETERS
d13C_CH4_t=1000.0*((soln.y[1]/soln.y[0])/VPDB -1.0)
dD_CH4_t=1000.0*(((soln.y[2]/soln.y[0])/4.0)/VSMOW -1.0)
delta_CH2D2_t=1000.0*((soln.y[4]/soln.y[0])/ratio_i_CH2D2_CH4-1.0)
delta_13CH3D_t=1000.0*((soln.y[3]/soln.y[0])/ratio_i_13CH3D_CH4-1.0)
ratio_CH2D2_t=soln.y[4]/soln.y[0]
ratio_13CH3D_t=soln.y[3]/soln.y[0]

# Create array of stochastic ratios for CH2D2 for each solution from time t.
# We have one stochastic ratio CH2D2/CH4 for each time.
RstochD2=np.ones(len(soln.t))
i=0
while i < len(soln.t):
    xH,xD,x12C,x13C,xst_CH4,xeq_CH4,alpha13D,alphaD2 = CH4_isotopologues(dD_CH4_t[i], d13C_CH4_t[i], TK)
    RstochD2[i]=xst_CH4[4]/xst_CH4[0]
    i=i+1
D12CH2D2_t=1000.0*((soln.y[4]/soln.y[0])/RstochD2-1.0)

# Create array of stochastic ratios for 13CH3D for each solution from time t.
# We have one stochastic ratio 13CH3D/CH4 for each time.
Rstoch13CD=np.ones(len(soln.t))
i=0
while i < len(soln.t):
    xH,xD,x12C,x13C,xst_CH4,xeq_CH4,alpha13D,alphaD2 = CH4_isotopologues(dD_CH4_t[i], d13C_CH4_t[i], TK)
    Rstoch13CD[i]=xst_CH4[3]/xst_CH4[0]
    i=i+1
D13CH3D_t=1000.0*((soln.y[3]/soln.y[0])/Rstoch13CD-1.0)

# F VALUES FOR RAYLEIGH PLOTS, fraction of CH4 remaining, at each time
F_CH4=soln.y[0]/Y0[0]
minus_lnF_CH4=-np.log(F_CH4)
print('')
#print('F values =',F_CH4)

# PROCESS RESULTS FOR CH3: access single scalar in Y using Y[element][timestep], as in Y[5][1]
# yields CH3 concentration in the 2nd time step.
soln.y[5][0]=1.0e-12
d13C_CH3_t=1000.0*((soln.y[6]/soln.y[5])/VPDB-1.0)
dD_CH3_t=1000.0*(((soln.y[7]/soln.y[5])/4.0)/VSMOW -1.0)
xCH2D_t=soln.y[7]/soln.y[5]
xCHD2_t=soln.y[9]/soln.y[5]
x13CH2D_t=soln.y[8]/soln.y[5]
print('')
#print("d13C CH3(t)=",d13C_CH3_t)
#print("dD CH3(t)=",dD_CH3_t)
#print("xCH2D(t)=",xCH2D_t)
#print("x13CH2D(t)=",x13CH2D_t)
#print("xCHD2(t)=",xCHD2_t)

# PROCESS RESULTS FOR D/H CH3:
soln.y[10][0]=1.0e-10
dD_H_t=1000.0*((soln.y[11]/soln.y[10])/VSMOW-1.0)
#print("dD H=",dD_H_t)

## MASS BALANCE CHECK
# totalH_t=4.0*soln.y[0]+4.0*soln.y[1]+3.0*(soln.y[2]+soln.y[3]+soln.y[5]+soln.y[6])+2.0*(soln.y[4]+soln.y[7]+soln.y[8])+soln.y[10]
# totalD_t=soln.y[2]+soln.y[3]+soln.y[7]+soln.y[8]+2.0*(soln.y[4]+soln.y[9])+soln.y[11]
# total12C_t=soln.y[0]+soln.y[2]+soln.y[4]+soln.y[5]+soln.y[7]+soln.y[9]
# print("total H(t)=",totalH_t)
# print('')
# print("total D(t)=",totalD_t)
# print('')
# print("total 12C(t)=",total12C_t)

# EQUILIBRIUM CURVE for plotting
Tref=np.linspace(290.0,1200.0,100)
G5=1/Tref
alpha13Dref=1+0.03555020*G5-433.038*G5**2+1.27021e6*G5**3-5.94804e8*G5**4+1.19663e11*G5**5-9.0723e12*G5**6
alphaD2ref=1+0.183798*G5-785.483*G5**2+1.056280e6*G5**3+9.37307e7*G5**4-8.91948e10*G5**5+9.90173e12*G5**6
D13CH3D_EQ=1000.0*(alpha13Dref-1.0)
D12CH2D2_EQ=1000.0*(alphaD2ref-1.0)

#------------------------------------------------------------------------------------------------
# MAKE PLOTS
#Molecule abundances vs time
plt.figure("N vs time",figsize=(7.0,5.0))
plt.plot(soln.t,soln.y[0],color='black',label='CH$_4$')
plt.plot(soln.t,soln.y[5],color='grey',label='CH$_3$')
plt.plot(soln.t,soln.y[2],color='blue',label='$^{12}$CH$_3$D')
plt.plot(soln.t,soln.y[3],color='green',label='$^{13}$CH$_3$D')
plt.plot(soln.t,soln.y[4],color='red',label='$^{12}$CH$_2$D$_2$')
plt.xlabel('Time',fontsize=14,labelpad=12)
plt.ylabel('Abundance',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('CH4_vs_time_smpl',bbox_inches='tight',dpi=1000)

#d13C vs time
plt.figure("d13C vs time",figsize=(7.0,5.0))
plt.plot(soln.t,d13C_CH4_t,color='black',label='d13C CH4')
plt.xlabel('Time',fontsize=14,labelpad=12)
plt.ylabel('$\delta^13$C',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('d13C_CH4_vs_time_smpl',bbox_inches='tight',dpi=1000)

#d13C vs F
#find line of best fit
lndelta=1000.0*np.log((d13C_CH4_t/1000.0+1.0)/(d13C_CH4/1000.0+1.0))
result=linregress(minus_lnF_CH4,lndelta)
print("\nd13C vs -lnF slope=",result.slope)
print("intercept=",result.intercept)
alpha_fit=1.0-result.slope/1000.0
print('d13C alpha =', alpha_fit)
plt.figure("d13C-d13C vs F",figsize=(7.0,5.0))
plt.plot(minus_lnF_CH4,d13C_CH4_t-d13C_CH4,color='black',label=r'best-fit alpha = %.6f' %alpha_fit)
plt.xlabel('-lnF',fontsize=14,labelpad=12)
plt.ylabel('$\delta^{13}$C $-\delta^{13}$C$_0$',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('d13C_CH4_vs_F_smpl',bbox_inches='tight',dpi=1000)

#dD vs time
plt.figure("dD vs time",figsize=(7.0,5.0))
plt.plot(soln.t,dD_CH4_t,color='black',label='dD CH4')
plt.xlabel('Time',fontsize=14,labelpad=12)
plt.ylabel('$\delta$D',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('dD_CH4_vs_time_smpl',bbox_inches='tight',dpi=1000)

#dD vs F
#find line of best fit
lndelta=1000.0*np.log((dD_CH4_t/1000.0+1.0)/(dD_CH4/1000.0+1.0))
result=linregress(minus_lnF_CH4,lndelta)
print("\ndD vs -lnF slope=",result.slope)
print("intercept=",result.intercept)
alpha_fit=1.0-result.slope/1000.0
print('dD alpha =', alpha_fit)
plt.figure("dD-dD vs F",figsize=(7.0,5.0))
plt.plot(minus_lnF_CH4,dD_CH4_t-dD_CH4,color='black',label=r'best-fit alpha = %.6f' %alpha_fit)
plt.xlabel('-lnF',fontsize=14,labelpad=12)
plt.ylabel('$\delta$D $-\delta$D$_0$',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('dD_CH4_vs_F_smpl',bbox_inches='tight',dpi=1000)

#D12CH2D2 vs time
plt.figure("D12CH2D2 vs time",figsize=(7.0,5.0))
plt.plot(soln.t,D12CH2D2_t,color='black',label='DCH2D2')
plt.xlabel('Time',fontsize=14,labelpad=12)
plt.ylabel('$\Delta$CH$_2$D$_2$',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('D12CH2D2_vs_time_smpl',bbox_inches='tight',dpi=1000)

#dCH2D2 vs F, note dCH2D2 is relative to initial, so dCH2D2-dCH2D2o = dCH2D2
#find line of best fit
lndelta=1000.0*np.log(ratio_CH2D2_t/ratio_i_CH2D2_CH4)
result=linregress(minus_lnF_CH4,lndelta)
print("\ndCH2D2 vs -lnF slope=",result.slope)
print("intercept=",result.intercept)
alpha_fit=1.0-result.slope/1000.0
print('dCH2D2 alpha =', alpha_fit)
plt.figure("dCH2D2 vs F",figsize=(7.0,5.0))
plt.plot(minus_lnF_CH4,delta_CH2D2_t,color='black',label=r'best-fit alpha = %.6f' %alpha_fit)
plt.xlabel('-lnF',fontsize=14,labelpad=12)
plt.ylabel('$\delta$CH$_2$D$_2$ relative to initial',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('dCH2D2_vs_F_smpl',bbox_inches='tight',dpi=1000)

#D13CH3D vs time
plt.figure("D13CH3D vs time",figsize=(7.0,5.0))
plt.plot(soln.t,D13CH3D_t,linestyle='-',color='black',label='13CH3D')
plt.xlabel('Time',fontsize=14,labelpad=12)
plt.ylabel('$\Delta^{13}$CH$_3$D',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('D13CH3D_vs_time_smpl',bbox_inches='tight',dpi=1000)

#d13CH3D vs F, note d13CH3D is relative to initial, so value equals difference
#find line of best fit
lndelta=1000.0*np.log(ratio_13CH3D_t/ratio_i_13CH3D_CH4)
result=linregress(minus_lnF_CH4,lndelta)
print("\nd13CH3D vs -lnF slope=",result.slope)
print("intercept=",result.intercept)
alpha_fit=1.0-result.slope/1000.0
print('d13CH3D alpha =', alpha_fit)
plt.figure("d13CH3D vs F",figsize=(7.0,5.0))
plt.plot(minus_lnF_CH4,delta_13CH3D_t,color='black',label=r'best-fit alpha = %.6f' %alpha_fit)
plt.xlabel('-lnF',fontsize=14,labelpad=12)
plt.ylabel('$\delta^13$CH$_3$D relative to initial',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('d13CH3D_vs_F_smpl',bbox_inches='tight',dpi=1000)

#dD CH4 vs d13C CH4
plt.figure("dD vs d13C",figsize=(7.0,5.0))
plt.plot(d13C_CH4_t[1:num-1],dD_CH4_t[1:num-1],linestyle='-',color='black',label='Model')
plt.ylabel('$\delta$D',fontsize=14,labelpad=12)
plt.xlabel('$\delta^{13}$C',fontsize=14,labelpad=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('dD_vs_d13C_CH4_smpl',bbox_inches='tight',dpi=1000)

#D12CH2D2 vs D13CH3D
plt.figure("D12CH2D2 vs D13CH3D",figsize=(7.0,5.0))
plt.plot(D13CH3D_EQ,D12CH2D2_EQ,color='grey',label='Equilibrium')
plt.plot(D13CH3D_t,D12CH2D2_t,linestyle='-',color='black',label='Model')
plt.ylabel('$\Delta^{12}$CH$_2$D$_2$',fontsize=14,labelpad=12)
plt.xlabel('$\Delta^{13}$CH$_3$D',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
plt.savefig('D12CH2D2_vs_D13CH3D_smpl',bbox_inches='tight',dpi=1000)

plt.show()

a_file = open('F_CH4_output_smpl.txt', 'w')
for i in range(0,num):
    a_file.write("%10.5e\n " % F_CH4[i])
a_file.close()

a_file = open('CH4_moles_output_smpl.txt', 'w')
for i in range(0,num):
    a_file.write("%10.5e\n " % soln.y[0][i])
a_file.close()

a_file = open('D12CH2D2_output_smpl.txt', 'w')
for i in range(0,num):
    a_file.write("%10.5e\n " % D12CH2D2_t[i])
a_file.close()

a_file = open('D13CH3D_output_smpl.txt', 'w')
for i in range(0,num):
    a_file.write("%10.5e\n " % D13CH3D_t[i])
a_file.close()

a_file = open('dD_CH4_output_smpl.txt', 'w')
for i in range(0,num):
    a_file.write("%10.5e\n " % dD_CH4_t[i])
a_file.close()

a_file = open('d13C_CH4_output_smpl.txt', 'w')
for i in range(0,num):
    a_file.write("%10.5e\n " % d13C_CH4_t[i])
a_file.close()

print("")
print("done")
print("")