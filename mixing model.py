import math
import numpy as np
# Modeling for mixing between mcr and pMMO
# Initial methane amount
n=440
tstep=10000

# Define alphas
aC_mcr_hs=0.9951
aD_mcr_hs=0.851
aCD_mcr_hs=0.834
aDD_mcr_hs=0.660

aC_pmo=0.9778	
aD_pmo=0.8019	
aCD_pmo=0.7842724	
aDD_pmo=0.629614

aC_smo_model=0.98247	
aD_smo_model=0.70350
aCD_smo_model=0.69051	
aDD_smo_model=0.47326

# Define relative rates rpmmo/rmcr
r=[0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Calculates mole fractions of isotopologues of CH4 from input bulk isotope ratios
# for stochastic distribution and for a specified temperature TK in Kelvin.
# The order of isotopologue fractions, vector Y, is set here as well.
def CH4_isotopologues(dD,d13C,TK):
    xst_CH4=np.zeros(10)
    xeq_CH4=np.zeros(10)
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


# Construct ode functions