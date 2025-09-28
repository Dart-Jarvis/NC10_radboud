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
import scipy.special
import time

# Fractionation factors, in the order of 13C,D,13CD,DD
a_anme_hs=[0.9951, 0.851, 0.834, 0.660] # High sulfate AOM from Liu et al., 2023
a_nc10=[0.9778, 0.8019, 0.7843, 0.6296]

#INPUTS
nmolec=10000
num=100000  #number of time steps
#------------------------------------------------------------------------------------------------
t_lower=0.0001 # minimum time for time interval
t_upper= 1.0 # maximum time for time interval, dimensionless

Tkinetics=293.15 # 20 C, T for AeOM kinetics and re-equilibration in K
T=200.0 # Equilibration temperature of initial methane in C, OVERRIDEN BY INPUT FILE
TK=T+273.15 # Equilibration temperature of initial methane in K

output=False

r_aeom=0.8 # Relative rate of AeOM
r_aom=1.0-r_aeom
os=False # Open system or not
if os==True:
    t_upper=10000.0 # Change the t_upper for open system to make sure it reaches steady state
# Define a transport flux
def dfdt(t,Y):
    # total CH4 in and out
    tCH4=sum(Y[0:4])
    # Fraction of methane oxidized by AeOM
    K_aeom[0]=r_aeom*Alpha_aeom[0] # 12CH4
    K_aeom[1]=1/4*r_aeom*Alpha_aeom[1] # 12CH3D, primary
    K_aeom[2]=3/4*r_aeom*Alpha_aeom[2] # 12CH3D, secondary
    K_aeom[3]=0.5*r_aeom*Alpha_aeom[3] # 12CH2D2, primary
    K_aeom[4]=0.5*r_aeom*Alpha_aeom[4] # 12CH2D2, secondary
    K_aeom[5]=r_aeom*Alpha_aeom[5] # 13CH4
    K_aeom[6]=1/4*r_aeom*Alpha_aeom[6] # 13CH3D, primary
    K_aeom[7]=3/4*r_aeom*Alpha_aeom[7] # 13CH3D, secondary
    # Fraction of methane oxidized by AOM
    K_aom[0]=r_aom*Alpha_aom[0]
    K_aom[1]=1/4*r_aom*Alpha_aom[1] 
    K_aom[2]=3/4*r_aom*Alpha_aom[2] 
    K_aom[3]=0.5*r_aom*Alpha_aom[3]
    K_aom[4]=0.5*r_aom*Alpha_aom[4]
    K_aom[5]=r_aom*Alpha_aom[5]
    K_aom[6]=1/4*r_aom*Alpha_aom[6]
    K_aom[7]=3/4*r_aom*Alpha_aom[7]
    # Total methane oxidized, ignoring the less abundant isotopologues
    Kt= (K_aeom[0]+K_aom[0])*Y[0]+(K_aeom[5]+K_aom[5])*Y[1]+(K_aeom[1]+K_aom[1]+K_aeom[2]+K_aom[2])*Y[2]+(K_aeom[6]+K_aom[6]+K_aeom[7]+K_aom[7])*Y[3]+(K_aeom[3]+K_aom[3]+K_aeom[4]+K_aom[4])*Y[4] 
    print("Total methane oxidized:", Kt)
    print("Total CH4:", tCH4)
    phi=0.4
    Kin=Kt/phi
    Kout=Kin-Kt
    if os==False:
        Kin=0
        Kout=0
    # ODE for the temporal change of isotopologue abundance
    # A universal model for both closed and open systems
    dYdt[0]=-(K_aeom[0]+K_aom[0])*Y[0] + Kin*Y0[0]/tCH4 - Kout*Y[0]/tCH4 # 12CH4
    dYdt[1]=-(K_aeom[5]+K_aom[5])*Y[1] + Kin*Y0[1]/tCH4 - Kout*Y[1]/tCH4 # 13CH4
    dYdt[2]=-(K_aeom[2]+K_aom[2])*Y[2]-(K_aeom[1]+K_aom[1])*Y[2] + Kin*Y0[2]/tCH4 - Kout*Y[2]/tCH4 # 12CH3D
    dYdt[3]=-(K_aeom[6]+K_aom[6])*Y[3]-(K_aeom[7]+K_aom[7])*Y[3] + Kin*Y0[3]/tCH4 - Kout*Y[3]/tCH4 # 13CH3D
    dYdt[4]=-(K_aeom[3]+K_aom[3])*Y[4]-(K_aeom[4]+K_aom[4])*Y[4] + Kin*Y0[4]/tCH4 - Kout*Y[4]/tCH4 # 12CH2D2

    return dYdt

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

ynames=[
"12CH4  ",
"13CH4  ",
"12CH3D ",
"13CH3D ",
"12CH2D2",
"13CH2D2",
"12CHD3",
"13CHD3",
"12CD4 ",
"13CD4 ",
"H      ",
"D      "]

abundance=[9.8877187e-01,
1.0695670e-02,
5.2665729e-04,
5.7133199e-06,
1.0584170e-07,
1.1378978e-09,
9.3383922e-12,
1.0101457e-13,
3.1087380e-16,
3.3627612e-18]

ny=len(ynames)
Y0=np.zeros(ny)

for i in range(0,5):
    Y0[i]=nmolec*abundance[i]  # Assign first 5 CH4 relative abundances to Y0, put arbitrary multiplier here
# Calculate compositional parameters for input CH4
d13C_CH4=1000.0*((Y0[1]/Y0[0])/VPDB -1.0)
dD_CH4=1000.0*(((Y0[2]/Y0[0])/4.0)/VSMOW -1.0)
ratio_i_CH2D2_CH4=Y0[4]/Y0[0]
ratio_i_13CH3D_CH4=Y0[3]/Y0[0]
xH,xD,x12C,x13C,xst_CH4,xeq_CH4,alpha13D,alphaD2 = CH4_isotopologues(dD_CH4, d13C_CH4, TK)
D12CH2D2_CH4=1000.0*(ratio_i_CH2D2_CH4/(xst_CH4[4]/xst_CH4[0])-1.0)
D13CH3D_CH4=1000.0*(ratio_i_13CH3D_CH4/(xst_CH4[3]/xst_CH4[0])-1.0)

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

# For the CH4 destruction, we use the sqrt of reduced masses for the bond being ruptured.
# Rate constants are stored in array K. The reduced mass ratios are stored as fractionation
# factors that operate on the base rate constant.
G5=1/Tkinetics
alpha13Dkinetics=1.0+0.03555020*G5-433.038*G5**2+1.27021e6*G5**3-5.94804e8*G5**4+1.19663e11*G5**5-9.0723e12*G5**6
alphaD2kinetics=1.0+0.183798*G5-785.483*G5**2+1.056280e6*G5**3+9.37307e7*G5**4-8.91948e10*G5**5+9.90173e12*G5**6

#-------------------
nrxns=8
rxns=[
"12CH4 -> 12CH3 + H",
"12CH3D -> 12CH3 + D",
"12CH3D -> 12CH2D + H",
"12CH2D2 -> 12CH2D + D",
"12CH2D2 -> 12CHD2 + H",
"13CH4 -> 13CH3 + H",
"13CH3D -> 13CH3 + D",
"13CH3D -> 13CH2D + H"
]

Alpha_aeom=np.ones(nrxns)
Alpha_aom=np.ones(nrxns)
K_aeom=np.zeros(nrxns)
K_aom=np.zeros(nrxns)

Alpha_aeom[0]=1.0 # Set to 1 as default
Alpha_aeom[1]=a_nc10[1]
Alpha_aeom[2]=a_nc10[1]
Alpha_aeom[3]=a_nc10[3]
Alpha_aeom[4]=a_nc10[3]
Alpha_aeom[5]=a_nc10[0]
Alpha_aeom[6]=a_nc10[2]
Alpha_aeom[7]=a_nc10[2]

Alpha_aom[0]=1.0 # Set to 1 as default
Alpha_aom[1]=a_anme_hs[1]
Alpha_aom[2]=a_anme_hs[1]
Alpha_aom[3]=a_anme_hs[3]
Alpha_aom[4]=a_anme_hs[3]
Alpha_aom[5]=a_anme_hs[0]
Alpha_aom[6]=a_anme_hs[2]
Alpha_aom[7]=a_anme_hs[2]
    
print('')
i=0
while i < len(Alpha_aeom):
    print("Alpha AeOM for reaction %d.\t %s\t= %.6f" %(i,rxns[i],Alpha_aeom[i]))
    print("Alpha AOM for reaction %d.\t %s\t= %.6f" %(i,rxns[i],Alpha_aom[i]))
    i=i+1

#------------------------------------------------------------------------------------------------
# SETUP ODEs OF FORM dY_i/dt = sum(K[R1][R2]) where sum is over the 16 reactions in which species i
# participates.

print('')
print('INITIAL CONDITION VECTOR (moles)')
i=0
while i < ny:
    # Yout[i]=Y0[i]  # Initialize Y for calculation of rates
    print("%i. %s \t = %.6e" %(i, ynames[i],Y0[i]))
    i=i+1
totalH=4.0*Y0[0]+4.0*Y0[1]+3.0*(Y0[2]+Y0[3]+Y0[5]+Y0[6])+2.0*(Y0[4]+Y0[7]+Y0[8])+Y0[10]
totalD=Y0[2]+Y0[3]+Y0[7]+Y0[8]+2.0*(Y0[4]+Y0[9])
print("total H initial =",totalH)
print("total D initial=",totalD)


dYdt=np.zeros(ny) # Vector of dY/dt values evaluated in function below.

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
##plt.xlim([0,4000.0])
plt.xlabel('Time',fontsize=14,labelpad=12)
plt.ylabel('Abundance',labelpad=11,fontsize=14)
#plt.yscale("log")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
#plt.ylim([0.0,ylimit])
#plt.xlim([350.0,2600])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
plt.savefig('CH4_vs_time_smpl',bbox_inches='tight',dpi=1000)

#d13C vs time
plt.figure("d13C vs time",figsize=(7.0,5.0))
plt.plot(soln.t,d13C_CH4_t,color='black',label='d13C CH4')
##plt.xlim([0,4000.0])
plt.xlabel('Time',fontsize=14,labelpad=12)
plt.ylabel('$\delta^13$C',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
#plt.ylim([0.0,ylimit])
#plt.xlim([350.0,2600])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
plt.savefig('d13C_CH4_vs_time_smpl',bbox_inches='tight',dpi=1000)
#
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
##plt.xlim([0,4000.0])
plt.xlabel('-lnF',fontsize=14,labelpad=12)
plt.ylabel('$\delta^{13}$C $-\delta^{13}$C$_0$',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
#plt.ylim([0.0,ylimit])
#plt.xlim([350.0,2600])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
plt.savefig('d13C_CH4_vs_F_smpl',bbox_inches='tight',dpi=1000)

#dD vs time
plt.figure("dD vs time",figsize=(7.0,5.0))
plt.plot(soln.t,dD_CH4_t,color='black',label='dD CH4')
##plt.xlim([0,4000.0])
plt.xlabel('Time',fontsize=14,labelpad=12)
plt.ylabel('$\delta$D',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
#plt.ylim([0.0,ylimit])
#plt.xlim([350.0,2600])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
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
##plt.xlim([0,4000.0])
plt.xlabel('-lnF',fontsize=14,labelpad=12)
plt.ylabel('$\delta$D $-\delta$D$_0$',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
#plt.ylim([0.0,ylimit])
#plt.xlim([350.0,2600])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
plt.savefig('dD_CH4_vs_F_smpl',bbox_inches='tight',dpi=1000)
plt.plot(minus_lnF_CH4,dD_CH4_t-dD_CH4,color='black',label=r'best-fit alpha = %.6f' %alpha_fit)

#Bulk isotope plot
splD=[-13.248,-59.417, -73.200, 56.858, 87.230, -133.900, -92.041, 39.009]
splC=[-23.215, -28.143, -29.418, -16.442, -13.548, -37.023, -31.781, -18.377]
plt.figure("dD vs d13C",figsize=(7.0,5.0))
plt.plot(d13C_CH4_t,dD_CH4_t)
plt.scatter(splC,splD)

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
#plt.ylim([0.0,ylimit])
#plt.xlim([350.0,2600])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
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
##plt.xlim([0,4000.0])
plt.xlabel('-lnF',fontsize=14,labelpad=12)
plt.ylabel('$\delta$CH$_2$D$_2$ relative to initial',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
#plt.ylim([0.0,ylimit])
#plt.xlim([350.0,2600])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
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
#plt.ylim([0.0,ylimit])
#plt.xlim([350.0,2600])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
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
##plt.xlim([0,4000.0])
plt.xlabel('-lnF',fontsize=14,labelpad=12)
plt.ylabel('$\delta^13$CH$_3$D relative to initial',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
#plt.ylim([0.0,ylimit])
#plt.xlim([350.0,2600])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
plt.savefig('d13CH3D_vs_F_smpl',bbox_inches='tight',dpi=1000)

#dD CH4 vs d13C CH4
plt.figure("dD vs d13C",figsize=(14.0,10.0))
plt.plot(d13C_CH4_t[1:num-1],dD_CH4_t[1:num-1],linestyle='-',color='black',label='Model')
plt.ylabel('$\delta$D',fontsize=14,labelpad=12)
plt.xlabel('$\delta^{13}$C',fontsize=14,labelpad=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
#plt.ylim([-40.0,25.0])
#plt.xlim([-5,8])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
plt.savefig('dD_vs_d13C_CH4_smpl',bbox_inches='tight',dpi=1000)

#D12CH2D2 vs D13CH3D
spl13CD=[-0.945,0.141,0.414,-0.537,-0.702,2.336,0.877,-0.964]
splDD=[-3.029,-4.994,-4.307,12.27,18.526,2.858,-2.734,12.692]
plt.figure("D12CH2D2 vs D13CH3D",figsize=(14.0,10.0))
plt.plot(D13CH3D_EQ,D12CH2D2_EQ,color='grey',label='Equilibrium')
#plt.plot(D13CH3D_t,D12CH2D2_t,marker='o',markerfacecolor='white',linestyle='-',color='blue',label='Model')
plt.scatter(spl13CD,splDD,label="samples")
plt.plot(D13CH3D_t,D12CH2D2_t,linestyle='-',color='black',label='Model')
plt.scatter(D13CH3D_t[[0,2499,4999,7499,9999]],D12CH2D2_t[[0,2499,4999,7499,9999]],color='red')
plt.ylabel('$\Delta^{12}$CH$_2$D$_2$',fontsize=14,labelpad=12)
plt.xlabel('$\Delta^{13}$CH$_3$D',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
#plt.ylim([-40.0,25.0])
#plt.xlim([-5,8])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
plt.savefig('D12CH2D2_vs_D13CH3D_smpl',bbox_inches='tight',dpi=1000)
plt.show()

if output==True:
    a_file = open(s + '_F_CH4_output.txt', 'w')
    for i in range(0,num):
        a_file.write("%.5f\n" % F_CH4[i])
    a_file.close()

    a_file = open(s + '_CH4_moles_output.txt', 'w')
    for i in range(0,num):
        a_file.write("%.5f\n" % soln.y[0][i])
    a_file.close()

    a_file = open(s + '_D12CH2D2_output.txt', 'w')
    for i in range(0,num):
        a_file.write("%.5f\n" % D12CH2D2_t[i])
    a_file.close()

    a_file = open(s + '_D13CH3D_output.txt', 'w')
    for i in range(0,num):
        a_file.write("%.5f\n" % D13CH3D_t[i])
    a_file.close()

    a_file = open(s + '_dD_CH4_output.txt', 'w')
    for i in range(0,num):
        a_file.write("%.5f\n" % dD_CH4_t[i])
    a_file.close()

    a_file = open(s + '_d13C_CH4_output.txt', 'w')
    for i in range(0,num):
        a_file.write("%.5f\n" % d13C_CH4_t[i])
    a_file.close()
    
print("")
print("done")
print("")