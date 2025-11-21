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
print("           UCLA AOM ISOTOPOLOGUE KINETIC MODEL v2          ")
print("   ----------------------------------------------------------")
print(" ")
#------------------------------------------------------------------------------------------------
#INPUTS
nmolec=1.0e10
num=2000  #number of time steps

reversible=True # True or False, refers to kinetics apart from clumping alphas
minF=7.0e-8 #0.000005 # minimum F (fraction remaining CH4) to plot
factor=1.0e-6  #e.g., 1.0e-14, k_reverse rate constant relative to oxidation rate constant
t_lower=0.0001 # minimum time for time interval
t_upper= 100.0 #75.0 # 15.0 maximum time for time interval, dimensionless

Tkinetics=333.15 # T for AOM kinetics and re-equilibration in K
T=200.0 # Equilibration temperature of initial methane in C, OVERRIDEN BY INPUT FILE
TK=T+273.15 # Equilibration temperature of initial methane in K

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

# Isotopologue names
ynames_logs=[
"12CH4",
"13CH4",
"12CH3D",
"13CH3D",
"12CH2D2",
"13CH2D2",
"12CHD3 ",
"13CHD3 ",
"12CD4  ",
"13CD4  ",
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
    print("   %s \t %.4e \t %.4e  " %(ynames[i],xst_CH4[i],xeq_CH4[i]))
    i=i+1
ratio_i_CH2D2_CH4=xeq_CH4[4]/xeq_CH4[0]
ratio_i_13CH3D_CH4=xeq_CH4[3]/xeq_CH4[0]
    
#------------------------------------------------------------------------------------------------
# SETUP SPECIES ARRAYS
ny=16 # Number of isotopologue species, count from 0 to ny-1
nydot=51 # Number of isotopomers species, must use 1:nydot-1 to match func code
YDOT=np.zeros(nydot) # vector of first derivatives with time
Y=np.zeros(nydot) # vector of isotopomer abundances
Y_logs=np.zeros(ny) # logs refers to isotopoLOGS rather than isotopic isomers
Y0_logs=np.zeros(ny) # vector of initial isotopomer abundances
Y0=np.zeros(nydot) # Assign zero to all initial abundances
Y_masses=np.zeros(nydot) # Masses of each isotopomer
Qt=np.zeros(nydot) # translational partition function, but not used here

Ynames=[
"     ",
"H    ",
"D    ",
"CHHH ",
"QHHH ",
"CHHD ",
"CDHH ",
"CHDH ",
"CHHHH",
"QDHH ",
"QHHD ",
"QHDH ",
"QHHHH",
"CDHD ",
"CDDH ",
"CHDD ",
"CHDHH",
"CHHDH",
"CDHHH",
"CHHHD",
"QDHD ",
"QHDD ",
"QDDH ",
"QDHHH",
"QHHHD",
"QHDHH",
"QHHDH",
"CDDD ",
"CDDHH",
"CDHHD",
"CHDDH",
"CHDHD",
"CDHDH",
"CHHDD",
"QDDD ",
"QDHHD",
"QDHDH",
"QHHDD",
"QHDHD",
"QDDHH",
"QHDDH",
"CDHDD",
"CDDHD",
"CDDDH",
"CHDDD",
"QDHDD",
"QDDDH",
"QHDDD",
"QDDHD",
"CDDDD",
"QDDDD"
]

# Assign masses for each species
Y_masses=[
0.0,
1.007825,
2.014000,
15.023475,
16.026830,
16.029650,
16.029650,
16.029650,
16.031300,
17.033005,
17.033005,
17.033005,
17.034655,
17.035825,
17.035825,
17.035825,
17.037475,
17.037475,
17.037475,
17.037475,
18.039180,
18.039180,
18.039180,
18.040830,
18.040830,
18.040830,
18.040830,
18.042000,
18.043650,
18.043650,
18.043650,
18.043650,
18.043650,
18.043650,
19.045355,
19.047005,
19.047005,
19.047005,
19.047005,
19.047005,
19.047005,
19.049825,
19.049825,
19.049825,
19.049825,
20.053180,
20.053180,
20.053180,
20.053180,
20.056000,
21.059355
]

# Calculate translational partition functions for each species
h=6.62607015e-34 # m^2
kb=1.380649e-23
pi=3.14159265
dim=2.0 # 3 for 3D, 2 for 2D
i=1
while i < nydot-1:
    Qt[i]=((2.0*pi*Y_masses[i]*kb*Tkinetics)/h**2.0)**(dim/2.0)
    Qt[i]=Qt[i]*(1.0e-8)**(dim/2.0) #provices estimate of linear dimension of motion
    i=i+1

#------------------------------------------------------------------------------------------------
# FRACTIONATION FACTORS
nrxns=281
K=np.zeros(nrxns) # Rate constants evaluated in dy/dt function
Alpha=np.ones(nrxns) # isotopologue fractionaton factors evaluated here

mass_12C = 12.00
mass_13C = 13.003355
mass_H  = 1.00784
mass_D  = 2.01410178

redmass_12C_H=1.0/(1.0/mass_12C+1.0/mass_H)
redmass_13C_H=1.0/(1.0/mass_13C+1.0/mass_H)
redmass_12C_D=1.0/(1.0/mass_12C+1.0/mass_D)
redmass_13C_D=1.0/(1.0/mass_13C+1.0/mass_D)

print('')
print('')
print('KINETICS')
print("reduced mass 12C-H =\t %.8f" %redmass_12C_H )
print("reduced mass 13C-H =\t %.8f" %redmass_13C_H )
print("reduced mass 12C-D =\t %.8f" %redmass_12C_D )
print("reduced mass 13C-D =\t %.8f" %redmass_13C_D )

# For the CH4 destruction, we use the sqrt of reduced masses for the bond being ruptured.
# First assign denominators for these alphas
Alpha[1]=redmass_12C_H
Alpha[2]=redmass_12C_D
Alpha[3]=redmass_12C_H
Alpha[4]=redmass_12C_H
Alpha[5]=redmass_12C_H
Alpha[6]=redmass_12C_D
Alpha[7]=redmass_12C_H
Alpha[8]=redmass_12C_H
Alpha[9]=redmass_12C_H
Alpha[10]=redmass_12C_D
Alpha[11]=redmass_12C_H
Alpha[12]=redmass_12C_H
Alpha[13]=redmass_12C_H
Alpha[14]=redmass_12C_D
Alpha[15]=redmass_12C_H
Alpha[16]=redmass_12C_H
Alpha[17]=redmass_12C_H
Alpha[18]=redmass_12C_D
Alpha[19]=redmass_12C_D
Alpha[20]=redmass_12C_D
Alpha[21]=redmass_12C_H
Alpha[22]=redmass_12C_H
Alpha[23]=redmass_12C_H
Alpha[24]=redmass_12C_D
Alpha[25]=redmass_12C_D
Alpha[26]=redmass_12C_D
Alpha[27]=redmass_12C_H
Alpha[28]=redmass_12C_H
Alpha[29]=redmass_12C_H
Alpha[30]=redmass_12C_D
Alpha[31]=redmass_12C_D
Alpha[32]=redmass_12C_D
Alpha[33]=redmass_12C_H
Alpha[34]=redmass_12C_H
Alpha[35]=redmass_12C_H
Alpha[36]=redmass_12C_D
Alpha[37]=redmass_12C_D
Alpha[38]=redmass_12C_D
Alpha[39]=redmass_12C_H
Alpha[40]=redmass_12C_H
Alpha[41]=redmass_12C_H
Alpha[42]=redmass_12C_D
Alpha[43]=redmass_12C_D
Alpha[44]=redmass_12C_D
Alpha[45]=redmass_12C_H
Alpha[46]=redmass_12C_H
Alpha[47]=redmass_12C_H
Alpha[48]=redmass_12C_D
Alpha[49]=redmass_12C_D
Alpha[50]=redmass_12C_D
Alpha[51]=redmass_12C_H
Alpha[52]=redmass_12C_H
Alpha[53]=redmass_12C_H
Alpha[54]=redmass_12C_D
Alpha[55]=redmass_12C_D
Alpha[56]=redmass_12C_D
Alpha[57]=redmass_12C_H
Alpha[58]=redmass_12C_D
Alpha[59]=redmass_12C_D
Alpha[60]=redmass_12C_D
Alpha[61]=redmass_12C_H
Alpha[62]=redmass_12C_D
Alpha[63]=redmass_12C_D
Alpha[64]=redmass_12C_D
Alpha[65]=redmass_12C_H
Alpha[66]=redmass_12C_D
Alpha[67]=redmass_12C_D
Alpha[68]=redmass_12C_D
Alpha[69]=redmass_12C_H
Alpha[70]=redmass_12C_D
#13CH4 destruction
Alpha[1+140]=redmass_13C_H
Alpha[2+140]=redmass_13C_D
Alpha[3+140]=redmass_13C_H
Alpha[4+140]=redmass_13C_H
Alpha[5+140]=redmass_13C_H
Alpha[6+140]=redmass_13C_D
Alpha[7+140]=redmass_13C_H
Alpha[8+140]=redmass_13C_H
Alpha[9+140]=redmass_13C_H
Alpha[10+140]=redmass_13C_D
Alpha[11+140]=redmass_13C_H
Alpha[12+140]=redmass_13C_H
Alpha[13+140]=redmass_13C_H
Alpha[14+140]=redmass_13C_D
Alpha[15+140]=redmass_13C_H
Alpha[16+140]=redmass_13C_H
Alpha[17+140]=redmass_13C_H
Alpha[18+140]=redmass_13C_D
Alpha[19+140]=redmass_13C_D
Alpha[20+140]=redmass_13C_D
Alpha[21+140]=redmass_13C_H
Alpha[22+140]=redmass_13C_H
Alpha[23+140]=redmass_13C_H
Alpha[24+140]=redmass_13C_D
Alpha[25+140]=redmass_13C_D
Alpha[26+140]=redmass_13C_D
Alpha[27+140]=redmass_13C_H
Alpha[28+140]=redmass_13C_H
Alpha[29+140]=redmass_13C_H
Alpha[30+140]=redmass_13C_D
Alpha[31+140]=redmass_13C_D
Alpha[32+140]=redmass_13C_D
Alpha[33+140]=redmass_13C_H
Alpha[34+140]=redmass_13C_H
Alpha[35+140]=redmass_13C_H
Alpha[36+140]=redmass_13C_D
Alpha[37+140]=redmass_13C_D
Alpha[38+140]=redmass_13C_D
Alpha[39+140]=redmass_13C_H
Alpha[40+140]=redmass_13C_H
Alpha[41+140]=redmass_13C_H
Alpha[42+140]=redmass_13C_D
Alpha[43+140]=redmass_13C_D
Alpha[44+140]=redmass_13C_D
Alpha[45+140]=redmass_13C_H
Alpha[46+140]=redmass_13C_H
Alpha[47+140]=redmass_13C_H
Alpha[48+140]=redmass_13C_D
Alpha[49+140]=redmass_13C_D
Alpha[50+140]=redmass_13C_D
Alpha[51+140]=redmass_13C_H
Alpha[52+140]=redmass_13C_H
Alpha[53+140]=redmass_13C_H
Alpha[54+140]=redmass_13C_D
Alpha[55+140]=redmass_13C_D
Alpha[56+140]=redmass_13C_D
Alpha[57+140]=redmass_13C_H
Alpha[58+140]=redmass_13C_D
Alpha[59+140]=redmass_13C_D
Alpha[60+140]=redmass_13C_D
Alpha[61+140]=redmass_13C_H
Alpha[62+140]=redmass_13C_D
Alpha[63+140]=redmass_13C_D
Alpha[64+140]=redmass_13C_D
Alpha[65+140]=redmass_13C_H
Alpha[66+140]=redmass_13C_D
Alpha[67+140]=redmass_13C_D
Alpha[68+140]=redmass_13C_D
Alpha[69+140]=redmass_13C_H
Alpha[70+140]=redmass_13C_D
# Create actual alphas from inputs above
i=1
while i < 71:
    Alpha[i]=np.sqrt(redmass_12C_H/Alpha[i])
    Alpha[i+140]=np.sqrt(redmass_12C_H/Alpha[i+140])
    i=i+1

# CLUMPING ALPHAS, modify alphas for secondary isotope effects of clumping
alphaDD=  1.0 #0.867
alpha13CD= 1.0 #0.9303
i=18
while i <54:
    Alpha[i]=Alpha[i]*alphaDD
    i=i+1
i=142
while i < 211:  # 13CD
    Alpha[i]=Alpha[i]*alpha13CD
    i=i+1
i=158
while i < 211:
    Alpha[i]=Alpha[i]*alphaDD
    i=i+1

# EQUILIBRIUM ALPHAS for methane mass-18 rare isotopologues
G5=1/Tkinetics
alpha13Dkinetics=1.0+0.03555020*G5-433.038*G5**2+1.27021e6*G5**3-5.94804e8*G5**4+1.19663e11*G5**5-9.0723e12*G5**6
alphaD2kinetics=1.0+0.183798*G5-785.483*G5**2+1.056280e6*G5**3+9.37307e7*G5**4-8.91948e10*G5**5+9.90173e12*G5**6

#TESTING the effect of no thermodynamic advantage for clumping
#alpha13Dkinetics=1.0
#alphaD2kinetics=1.0

# REVERSIBILITY: complete reversibility of fractionaton between CH4 and CH3+H
if reversible:
    Alpha[71]=np.sqrt(redmass_12C_H/redmass_12C_H)
    Alpha[72]=np.sqrt(redmass_12C_H/redmass_12C_D)
    Alpha[73]=Alpha[72]
    Alpha[74]=Alpha[72]
    Alpha[75]=Alpha[72]
    i=76
    while i < 88:
        Alpha[i]=np.sqrt(redmass_12C_H/redmass_12C_H)
        i=i+1
    i=88
    while i < 106:
        Alpha[i]=np.sqrt(redmass_12C_H/redmass_12C_D)
        i=i+1
    i=106
    while i < 124:
        Alpha[i]=np.sqrt(redmass_12C_H/redmass_12C_H)
        i=i+1
    i=124
    while i < 136:
        Alpha[i]=np.sqrt(redmass_12C_H/redmass_12C_D)
        i=i+1
    Alpha[136]=np.sqrt(redmass_12C_H/redmass_12C_H)
    Alpha[137]=Alpha[136]
    Alpha[138]=Alpha[136]
    Alpha[139]=Alpha[136]
    Alpha[140]=np.sqrt(redmass_12C_H/redmass_12C_D)

    # 13C equivalents of above
    Alpha[71+140]=np.sqrt(redmass_12C_H/redmass_13C_H)
    Alpha[72+140]=np.sqrt(redmass_12C_H/redmass_13C_D)
    Alpha[73+140]=Alpha[72+140]
    Alpha[74+140]=Alpha[72+140]
    Alpha[75+140]=Alpha[72+140]
    i=76+140
    while i < 88 + 140:
        Alpha[i]=np.sqrt(redmass_12C_H/redmass_13C_H)
        i=i+1
    i=88+140
    while i < 106 + 140:
        Alpha[i]=np.sqrt(redmass_12C_H/redmass_13C_D)
        i=i+1
    i=106 +140
    while i < 124 +140:
        Alpha[i]=np.sqrt(redmass_12C_H/redmass_13C_H)
        i=i+1
    i=124 + 140
    while i < 136 + 140:
        Alpha[i]=np.sqrt(redmass_12C_H/redmass_13C_D)
        i=i+1
    Alpha[136+140]=np.sqrt(redmass_12C_H/redmass_13C_H)
    Alpha[137+140]=Alpha[136+140]
    Alpha[138+140]=Alpha[136+140]
    Alpha[139+140]=Alpha[136+140]
    Alpha[140+140]=np.sqrt(redmass_12C_H/redmass_12C_D)


# RETURN EQUILIBRIUM OPTION: These lines drive towards equilibrium for mass-18
i=71
while i < 141:
    if i > 87 and i < 124:
        Alpha[i]=Alpha[i]*alphaD2kinetics
        Alpha[i+140]=Alpha[i+140]*alphaD2kinetics*alpha13Dkinetics #means > 227 to < 264 for 13CD2
    i=i+1
i=212
while i < 281:
    if i > 211 and i < 228: #212 - 227, inclusive
        Alpha[i]=Alpha[i]*alpha13Dkinetics
    i=i+1

# Print alphas to screen
print('')
i=1
while i < nrxns:
    print("Alpha for reaction %d.\t= %.6f" %(i,Alpha[i]))
    i=i+1

#------------------------------------------------------------------------------------------------
# SETUP ODEs OF FORM dY_i/dt = sum(K[R1][R2]) where sum is over the reactions in which species i
# participates.

# INITIAL ISOTOPOLOGE ABUNDANCES
Y0_logs[0:9]=nmolec*xeq_CH4[0:9]
print('')
print('INITIAL CONDITION ISOTOPOLOGUES (moles)')
i=0
while i < len(Y0_logs):
    Y_logs[i]=Y0_logs[i]  # Initialize Y for calculation of rates
    print("%i. %s \t = %.6e" %(i, ynames_logs[i],Y0_logs[i]))
    i=i+1

# INITIAL CH4 ISOTOPOMER ABUNDANCES OBTAINED FROM INITIAL ISOTOPOLOGUE ABUNDANCES
Y0[8]=nmolec*xeq_CH4[0] #CH4
Y0[12]=nmolec*xeq_CH4[1] #QH4
Y0[16:19+1]=nmolec*xeq_CH4[2]/4.0 #CH3D
Y0[23:26+1]=nmolec*xeq_CH4[3]/4.0 #QH3D
Y0[28:33+1]=nmolec*xeq_CH4[4]/6.0 #12CH2D2
Y0[35:40+1]=nmolec*xeq_CH4[5]/6.0 #13CH2D2
Y0[41:44+1]=nmolec*xeq_CH4[6]/4.0 #12CHD3
Y0[45:48+1]=nmolec*xeq_CH4[7]/4.0 #13CHD3
Y0[49]=nmolec*xeq_CH4[8] #12CD4
Y0[50]=nmolec*xeq_CH4[9] #13CD4
print('')
print('INITIAL CONDITION ISOTOPOMERS')
i=1
while i< (len(Y0)):
    Y[i]=Y0[i]
    print("%i. %s \t = %.6e" %(i, Ynames[i],Y0[i]))
    i=i+1
    
#totalH=4.0*Y0[0]+4.0*Y0[1]+3.0*(Y0[2]+Y0[3]+Y0[5]+Y0[6])+2.0*(Y0[4]+Y0[7]+Y0[8])+Y0[10]
#totalD=Y0[2]+Y0[3]+Y0[7]+Y0[8]+2.0*(Y0[4]+Y0[9])
#print("total H initial =",totalH)
#print("total D initial=",totalD)

# The FUNCTION for the right-hand-side of each ordinary differential equation is here.
# The values returned are the rates for each species Y_i.
#------------------------------------------------------------------------------------------------
def dfdt(t,Y):
    # Assign rate constants
    #factor=0.0001  #0.0075
    K_oxidize=1.0
    K_synth=factor*K_oxidize
    # Highly symmetric reactants have no branching
    K[1]=K_oxidize*Alpha[1] # CH4 --> CH3 + H
    K[70]=K_oxidize*Alpha[70] # CD4 --> CD3 + D
    K[141]=K_oxidize*Alpha[141] # QH4 --> QH3 + H
    K[210]=K_oxidize*Alpha[210] # QD4 --> QD3 + D
    K[71]=K_synth*Alpha[71] # CH3 + H --> CH4
    K[140]=K_synth*Alpha[140] #CD4 + D --> CD4
    K[211]=K_synth*Alpha[211] # QHHH + H --> QH4
    K[280]=K_synth*Alpha[280] # QDDD + D --> QD4
    # Adjust rxn rate constants for for isomer branching
    i=1
    while i < 281:
        if i > 1 and i < 18:
            K[i]=K_oxidize*Alpha[i]/(4.0) # CH3D, each with 4 branches
        if i > 17 and i < 54:
            K[i]=K_oxidize*Alpha[i]/(6.0) # CH2D2, each with 6 branches
        if i > 53 and i < 70:
            K[i]=K_oxidize*Alpha[i]/(4.0) # CHD3, each with 4 branches
        if i > 141 and i < 158:
            K[i]=K_oxidize*Alpha[i]/(4.0) # QH3D, each with 4 branches
        if i > 157 and i < 194:
            K[i]=K_oxidize*Alpha[i]/(6.0) # QH2D2, each with 6 branches
        if i > 193 and i < 210:
            K[i]=K_oxidize*Alpha[i]/(4.0) # QHD3, each with 4 branches
        if i > 71 and i < 88:
            K[i]=K_synth*Alpha[i]/4.0 # CH3 + H or D --> CH3D, each with 4 branches
        if i > 87 and i < 124:
            K[i]=K_synth*Alpha[i]/6.0 # CH3 + H or D --> CH2D2, each with 6 branches
        if i > 123 and i < 140:
            K[i]=K_synth*Alpha[i]/4.0 # CH3 + H or D --> CHD3, each with 4 branches
        if i > 211 and i < 228:
            K[i]=K_synth*Alpha[i]/4.0 # QH3 + H or D --> QH3D, each with 4 branches
        if i > 227 and i < 264:
            K[i]=K_synth*Alpha[i]/6.0 # QH3 + H or D --> QH2D2, each with 6 branches
        if i > 263 and i < 280:
            K[i]=K_synth*Alpha[i]/4.0 # QH3 + H or D --> QHD3, each with 4 branches
        i=i+1
    #dy/dt = YDOT
    # H
    F=(0.
    +K[1]*Y[8]+K[3]*Y[19]+K[4]*Y[19]+K[5]*Y[19]+K[7]*Y[17]+K[8]*Y[17]
    +K[9]*Y[17]+K[11]*Y[16]+K[12]*Y[16]+K[13]*Y[16]+K[15]*Y[18]+K[16]
    *Y[18]+K[17]*Y[18]+K[21]*Y[33]+K[22]*Y[33]+K[23]*Y[33]+K[27]*Y[31]
    +K[28]*Y[31]+K[29]*Y[31]+K[33]*Y[29]+K[34]*Y[29]+K[35]*Y[29]+K[39]*Y[32]
    +K[40]*Y[32]+K[41]*Y[32]+K[45]*Y[28]+K[46]*Y[28]+K[47]*Y[28]+K[51]*Y[30]
    +K[52]*Y[30]+K[53]*Y[30]+K[57]*Y[44]+K[61]*Y[41]+K[65]*Y[42]+K[69]*Y[43]
    +K[141]*Y[12]+K[143]*Y[24]+K[144]*Y[24]+K[145]*Y[24]+K[147]*Y[26]
    +K[148]*Y[26]+K[149]*Y[26]+K[151]*Y[25]+K[152]*Y[25]+K[153]*Y[25]
    +K[155]*Y[23]+K[156]*Y[23]+K[157]*Y[23]+K[161]*Y[37]+K[162]*Y[37]
    +K[163]*Y[37]+K[167]*Y[38]+K[168]*Y[38]+K[169]*Y[38]+K[173]*Y[35]
    +K[174]*Y[35]+K[175]*Y[35]+K[179]*Y[36]+K[180]*Y[36]+K[181]*Y[36]
    +K[185]*Y[39]+K[186]*Y[39]+K[187]*Y[39]+K[191]*Y[40]+K[192]*Y[40]
    +K[193]*Y[40]+K[197]*Y[47]+K[201]*Y[45]+K[205]*Y[48]+K[209]*Y[46])
    D=(0.
    +K[71]*Y[3]+K[76]*Y[5]+K[77]*Y[5]+K[78]*Y[5]+K[79]*Y[5]+K[80]*Y[7]
    +K[81]*Y[7]+K[82]*Y[7]+K[83]*Y[7]+K[84]*Y[6]+K[85]*Y[6]+K[86]*Y[6]
    +K[87]*Y[6]+K[106]*Y[15]+K[107]*Y[15]+K[108]*Y[15]+K[109]*Y[15])
    D=D+K[110]*Y[15]+K[111]*Y[15]+K[112]*Y[13]+K[113]*Y[13]+K[114]*Y[13]
    D=D+K[115]*Y[13]+K[116]*Y[13]+K[117]*Y[13]+K[118]*Y[14]+K[119]*Y[14]
    D=D+K[120]*Y[14]+K[121]*Y[14]+K[122]*Y[14]+K[123]*Y[14]+K[136]*Y[27]
    D=(D+K[137]*Y[27]+K[138]*Y[27]+K[139]*Y[27]+K[211]*Y[4]+K[216]*Y[10]+
    K[217]*Y[10]+K[218]*Y[10]+K[219]*Y[10]+K[220]*Y[11]+K[221]*Y[11]+
    K[222]*Y[11]+K[223]*Y[11]+K[224]*Y[9]+K[225]*Y[9]+K[226]*Y[9]+K[227]*Y[9]
    +K[246]*Y[21]+K[247]*Y[21]+K[248]*Y[21]+K[249]*Y[21]+K[250]*Y[21]
    +K[251]*Y[21]+K[252]*Y[20]+K[253]*Y[20]+K[254]*Y[20]+K[255]*Y[20]
    +K[256]*Y[20]+K[257]*Y[20]+K[258]*Y[22]+K[259]*Y[22]+K[260]*Y[22]
    +K[261]*Y[22]+K[262]*Y[22]+K[263]*Y[22]+K[276]*Y[34]+K[277]*Y[34]
    +K[278]*Y[34]+K[279]*Y[34])
    YDOT[1]=F-(D*Y[1])
    # D
    F=(0.
    +K[2]*Y[19]+K[6]*Y[17]+K[10]*Y[16]+K[14]*Y[18]+K[18]*Y[33]+K[19]*
    Y[33]+K[20]*Y[33]+K[24]*Y[31]+K[25]*Y[31]+K[26]*Y[31]+K[30]*Y[29])
    F=(F+K[31]*Y[29]+K[32]*Y[29]+K[36]*Y[32]+K[37]*Y[32]+K[38]*Y[32]
    +K[42]*Y[28]+K[43]*Y[28]+K[44]*Y[28]+K[48]*Y[30]+K[49]*Y[30]+K[50]*Y[30]
    +K[54]*Y[44]+K[55]*Y[44]+K[56]*Y[44]+K[58]*Y[41]+K[59]*Y[41]+K[60]*Y[41]
    +K[62]*Y[42]+K[63]*Y[42]+K[64]*Y[42]+K[66]*Y[43]+K[67]*Y[43]+K[68]*Y[43]
    +K[70]*Y[49]+K[142]*Y[24]+K[146]*Y[26]+K[150]*Y[25]+K[154]*Y[23]+K[158]*Y[37]
    +K[159]*Y[37]+K[160]*Y[37]+K[164]*Y[38]+K[165]*Y[38]+K[166]*Y[38]
    +K[170]*Y[35]+K[171]*Y[35]+K[172]*Y[35]+K[176]*Y[36]+K[177]*Y[36]
    +K[178]*Y[36]+K[182]*Y[39]+K[183]*Y[39]+K[184]*Y[39]+K[188]*Y[40]
    +K[189]*Y[40]+K[190]*Y[40]+K[194]*Y[47]+K[195]*Y[47]+K[196]*Y[47]
    +K[198]*Y[45]+K[199]*Y[45]+K[200]*Y[45]+K[202]*Y[48]+K[203]*Y[48]
    +K[204]*Y[48]+K[206]*Y[46]+K[207]*Y[46]+K[208]*Y[46]+K[210]*Y[50])
    D=(0.
    +K[72]*Y[3]+K[73]*Y[3]+K[74]*Y[3]+K[75]*Y[3]+K[88]*Y[5]+K[89]*Y[5]
    +K[90]*Y[5]+K[91]*Y[5]+K[92]*Y[5]+K[93]*Y[5]+K[94]*Y[7]+K[95]*Y[7]
    +K[96]*Y[7]+K[97]*Y[7]+K[98]*Y[7]+K[99]*Y[7]+K[100]*Y[6]+K[101]
    *Y[6]+K[102]*Y[6]+K[103]*Y[6]+K[104]*Y[6]+K[105]*Y[6]+K[124]*Y[15]
    +K[125]*Y[15]+K[126]*Y[15]+K[127]*Y[15]+K[128]*Y[13]+K[129]*Y[13]
    +K[130]*Y[13]+K[131]*Y[13]+K[132]*Y[14]+K[133]*Y[14]+K[134]*Y[14]
    +K[135]*Y[14]+K[140]*Y[27]+K[212]*Y[4]+K[213]*Y[4]+K[214]*Y[4]+K[215]*Y[4]
    +K[228]*Y[10]+K[229]*Y[10]+K[230]*Y[10]+K[231]*Y[10]+K[232]*Y[10]
    +K[233]*Y[10]+K[234]*Y[11]+K[235]*Y[11]+K[236]*Y[11]+K[237]*Y[11]
    +K[238]*Y[11]+K[239]*Y[11]+K[240]*Y[9]+K[241]*Y[9]+K[242]*Y[9]+K[243]*Y[9]
    +K[244]*Y[9]+K[245]*Y[9]+K[264]*Y[21]+K[265]*Y[21]+K[266]*Y[21]
    +K[267]*Y[21]+K[268]*Y[20]+K[269]*Y[20]+K[270]*Y[20]+K[271]*Y[20]
    +K[272]*Y[22]+K[273]*Y[22]+K[274]*Y[22]+K[275]*Y[22]+K[280]*Y[34])
    YDOT[2]=F-(D*Y[2])
    # CHHH
    F=0.0+K[1]*Y[8]+K[2]*Y[19]+K[6]*Y[17]+K[10]*Y[16]+K[14]*Y[18]
    D=0.0+K[71]*Y[1]+K[72]*Y[2]+K[73]*Y[2]+K[74]*Y[2]+K[75]*Y[2]
    YDOT[3]=F-(D*Y[3])
    # QHHH
    F=0.0+K[141]*Y[12]+K[142]*Y[24]+K[146]*Y[26]+K[150]*Y[25]+K[154]*Y[23]
    D=0.0+K[211]*Y[1]+K[212]*Y[2]+K[213]*Y[2]+K[214]*Y[2]+K[215]*Y[2]
    YDOT[4]=F-(D*Y[4])
    # CHHD
    F=(0.0+K[3]*Y[19]+K[7]*Y[17]+K[11]*Y[16]+K[15]*Y[18]+K[18]*Y[33]+K[24]*
    Y[31]+K[30]*Y[29]+K[36]*Y[32]+K[42]*Y[28]+K[48]*Y[30])
    D=(0.0+K[76]*Y[1]+K[77]*Y[1]+K[78]*Y[1]+K[79]*Y[1]+K[88]*Y[2]+K[89]*Y[2
    ]+K[90]*Y[2]+K[91]*Y[2]+K[92]*Y[2]+K[93]*Y[2])
    YDOT[5]=F-(D*Y[5])
    # CDHH
    F=(0.0+K[5]*Y[19]+K[9]*Y[17]+K[13]*Y[16]+K[17]*Y[18]+K[20]*Y[33]+K[26]*
    Y[31]+K[32]*Y[29]+K[38]*Y[32]+K[44]*Y[28]+K[50]*Y[30])
    D=(0.0+K[84]*Y[1]+K[85]*Y[1]+K[86]*Y[1]+K[87]*Y[1]+K[100]*Y[2]+K[101]*Y
    [2]+K[102]*Y[2]+K[103]*Y[2]+K[104]*Y[2]+K[105]*Y[2])
    YDOT[6]=F-(D*Y[6])
    # CHDH
    F=(0.0+K[4]*Y[19]+K[8]*Y[17]+K[12]*Y[16]+K[16]*Y[18]+K[19]*Y[33]+K[25]*
    Y[31]+K[31]*Y[29]+K[37]*Y[32]+K[43]*Y[28]+K[49]*Y[30])
    D=(0.0+K[80]*Y[1]+K[81]*Y[1]+K[82]*Y[1]+K[83]*Y[1]+K[94]*Y[2]+K[95]*Y[2
    ]+K[96]*Y[2]+K[97]*Y[2]+K[98]*Y[2]+K[99]*Y[2])
    YDOT[7]=F-(D*Y[7])
    # CHHHH
    F=0.0+K[71]*Y[3]*Y[1]
    D=0.0+K[1]
    YDOT[8]=F-(D*Y[8])
    # QDHH
    F=(0.0+K[145]*Y[24]+K[149]*Y[26]+K[153]*Y[25]+K[157]*Y[23]+K[160]*Y[37]
    +K[166]*Y[38]+K[172]*Y[35]+K[178]*Y[36]+K[184]*Y[39]+K[190]*Y[40])
    D=(0.0+K[224]*Y[1]+K[225]*Y[1]+K[226]*Y[1]+K[227]*Y[1]+K[240]*Y[2]+K[241]*Y[2]
    +K[242]*Y[2]+K[243]*Y[2]+K[244]*Y[2]+K[245]*Y[2])
    YDOT[9]=F-(D*Y[9])
    # QHHD
    F=(0.0+K[143]*Y[24]+K[147]*Y[26]+K[151]*Y[25]+K[155]*Y[23]+K[158]*Y[37]
    +K[164]*Y[38]+K[170]*Y[35]+K[176]*Y[36]+K[182]*Y[39]+K[188]*Y[40])
    D=(0.0+K[216]*Y[1]+K[217]*Y[1]+K[218]*Y[1]+K[219]*Y[1]+K[228]*Y[2]+K[229]*Y[2]
    +K[230]*Y[2]+K[231]*Y[2]+K[232]*Y[2]+K[233]*Y[2])
    YDOT[10]=F-(D*Y[10])
    # QHDH
    F=(0.0+K[144]*Y[24]+K[148]*Y[26]+K[152]*Y[25]+K[156]*Y[23]+K[159]*Y[37]
    +K[165]*Y[38]+K[171]*Y[35]+K[177]*Y[36]+K[183]*Y[39]+K[189]*Y[40])
    D=(0.0+K[220]*Y[1]+K[221]*Y[1]+K[222]*Y[1]+K[223]*Y[1]+K[234]*Y[2]+K[235]*Y[2]
    +K[236]*Y[2]+K[237]*Y[2]+K[238]*Y[2]+K[239]*Y[2])
    YDOT[11]=F-(D*Y[11])
    # QHHHH
    F=0.0+K[211]*Y[4]*Y[1]
    D=0.0+K[141]
    YDOT[12]=F-(D*Y[12])
    # CDHD
    F=(0.0+K[22]*Y[33]+K[28]*Y[31]+K[34]*Y[29]+K[40]*Y[32]+K[46]*Y[28]+K[52]*Y[30]
    +K[55]*Y[44]+K[59]*Y[41]+K[63]*Y[42]+K[67]*Y[43])
    D=(0.0+K[112]*Y[1]+K[113]*Y[1]+K[114]*Y[1]+K[115]*Y[1]+K[116]*Y[1]+K[117]*Y[1]
    +K[128]*Y[2]+K[129]*Y[2]+K[130]*Y[2]+K[131]*Y[2])
    YDOT[13]=F-(D*Y[13])
    # CDDH
    F=(0.0+K[23]*Y[33]+K[29]*Y[31]+K[35]*Y[29]+K[41]*Y[32]+K[47]*Y[28]+K[53]*Y[30]
    +K[56]*Y[44]+K[60]*Y[41]+K[64]*Y[42]+K[68]*Y[43])
    D=(0.0+K[118]*Y[1]+K[119]*Y[1]+K[120]*Y[1]+K[121]*Y[1]+K[122]*Y[1]+K[123]*Y[1]
    +K[132]*Y[2]+K[133]*Y[2]+K[134]*Y[2]+K[135]*Y[2])
    YDOT[14]=F-(D*Y[14])
    # CHDD
    F=(0.0+K[21]*Y[33]+K[27]*Y[31]+K[33]*Y[29]+K[39]*Y[32]+K[45]*Y[28]+K[51]*Y[30]
    +K[54]*Y[44]+K[58]*Y[41]+K[62]*Y[42]+K[66]*Y[43])
    D=(0.0+K[106]*Y[1]+K[107]*Y[1]+K[108]*Y[1]+K[109]*Y[1]+K[110]*Y[1]+K[111]*Y[1]
    +K[124]*Y[2]+K[125]*Y[2]+K[126]*Y[2]+K[127]*Y[2])
    YDOT[15]=F-(D*Y[15])
    # CHDHH
    F=0.0+K[74]*Y[3]*Y[2]+K[78]*Y[5]*Y[1]+K[82]*Y[7]*Y[1]+K[86]*Y[6]*Y[1]
    D=0.0+K[10]+K[11]+K[12]+K[13]
    YDOT[16]=F-(D*Y[16])
    # CHHDH
    F=0.0+K[73]*Y[3]*Y[2]+K[77]*Y[5]*Y[1]+K[81]*Y[7]*Y[1]+K[85]*Y[6]*Y[1]
    D=0.0+K[6]+K[7]+K[8]+K[9]
    YDOT[17]=F-(D*Y[17])
    # CDHHH
    F=0.0+K[75]*Y[3]*Y[2]+K[79]*Y[5]*Y[1]+K[83]*Y[7]*Y[1]+K[87]*Y[6]*Y[1]
    D=0.0+K[14]+K[15]+K[16]+K[17]
    YDOT[18]=F-(D*Y[18])
    # CHHHD
    F=0.0+K[72]*Y[3]*Y[2]+K[76]*Y[5]*Y[1]+K[80]*Y[7]*Y[1]+K[84]*Y[6]*Y[1]
    D=0.0+K[2]+K[3]+K[4]+K[5]
    YDOT[19]=F-(D*Y[19])
    # QDHD
    F=(0.0+K[162]*Y[37]+K[168]*Y[38]+K[174]*Y[35]+K[180]*Y[36]+K[186]*Y[39]
    +K[192]*Y[40]+K[195]*Y[47]+K[199]*Y[45]+K[203]*Y[48]+K[207]*Y[46])
    D=(0.
    +K[252]*Y[1]+K[253]*Y[1]+K[254]*Y[1]+K[255]*Y[1]+K[256]*Y[1]+K[257]*Y[1]
    +K[268]*Y[2]+K[269]*Y[2]+K[270]*Y[2]+K[271]*Y[2])
    YDOT[20]=F-(D*Y[20])
    # QHDD
    F=(0.
    +K[161]*Y[37]+K[167]*Y[38]+K[173]*Y[35]+K[179]*Y[36]+K[185]*Y[39]
    +K[191]*Y[40]+K[194]*Y[47]+K[198]*Y[45]+K[202]*Y[48]+K[206]*Y[46])
    D=(0.
    +K[246]*Y[1]+K[247]*Y[1]+K[248]*Y[1]+K[249]*Y[1]+K[250]*Y[1]+K[251]*Y[1]
    +K[264]*Y[2]+K[265]*Y[2]+K[266]*Y[2]+K[267]*Y[2])
    YDOT[21]=F-(D*Y[21])
    # QDDH
    F=(0.
    +K[163]*Y[37]+K[169]*Y[38]+K[175]*Y[35]+K[181]*Y[36]+K[187]*Y[39]
    +K[193]*Y[40]+K[196]*Y[47]+K[200]*Y[45]+K[204]*Y[48]+K[208]*Y[46])
    D=(0.
    +K[258]*Y[1]+K[259]*Y[1]+K[260]*Y[1]+K[261]*Y[1]+K[262]*Y[1]+K[263]*Y[1]
    +K[272]*Y[2]+K[273]*Y[2]+K[274]*Y[2]+K[275]*Y[2])
    YDOT[22]=F-(D*Y[22])
    # QDHHH
    F=0.0+K[215]*Y[4]*Y[2]+K[219]*Y[10]*Y[1]+K[223]*Y[11]*Y[1]+K[227]*Y[9]*Y[1]
    D=0.0+K[154]+K[155]+K[156]+K[157]
    YDOT[23]=F-(D*Y[23])
    # QHHHD
    F=0.0+K[212]*Y[4]*Y[2]+K[216]*Y[10]*Y[1]+K[220]*Y[11]*Y[1]+K[224]*Y[9]*Y[1]
    D=0.0+K[142]+K[143]+K[144]+K[145]
    YDOT[24]=F-(D*Y[24])
    # QHDHH
    F=0.0+K[214]*Y[4]*Y[2]+K[218]*Y[10]*Y[1]+K[222]*Y[11]*Y[1]+K[226]*Y[9]*Y[1]
    D=0.0+K[150]+K[151]+K[152]+K[153]
    YDOT[25]=F-(D*Y[25])
    # QHHDH
    F=0.0+K[213]*Y[4]*Y[2]+K[217]*Y[10]*Y[1]+K[221]*Y[11]*Y[1]+K[225]*Y[9]*Y[1]
    D=0.0+K[146]+K[147]+K[148]+K[149]
    YDOT[26]=F-(D*Y[26])
    # CDDD
    F=0.0+K[57]*Y[44]+K[61]*Y[41]+K[65]*Y[42]+K[69]*Y[43]+K[70]*Y[49]
    D=0.0+K[136]*Y[1]+K[137]*Y[1]+K[138]*Y[1]+K[139]*Y[1]+K[140]*Y[2]
    YDOT[27]=F-(D*Y[27])
    # CDDHH
    F=(0.0+K[92]*Y[5]*Y[2]+K[98]*Y[7]*Y[2]+K[104]*Y[6]*Y[2]+K[110]*Y[15]*Y[1]
    +K[116]*Y[13]*Y[1]+K[122]*Y[14]*Y[1])
    D=0.0+K[42]+K[43]+K[44]+K[45]+K[46]+K[47]
    YDOT[28]=F-(D*Y[28])
    # CDHHD
    F=(0.0+K[90]*Y[5]*Y[2]+K[96]*Y[7]*Y[2]+K[102]*Y[6]*Y[2]+K[108]*Y[15]*Y[1]
    +K[114]*Y[13]*Y[1]+K[120]*Y[14]*Y[1])
    D=0.0+K[30]+K[31]+K[32]+K[33]+K[34]+K[35]
    YDOT[29]=F-(D*Y[29])
    # CHDDH
    F=(0.0+K[93]*Y[5]*Y[2]+K[99]*Y[7]*Y[2]+K[105]*Y[6]*Y[2]+K[111]*Y[15]*Y[1]
    +K[117]*Y[13]*Y[1]+K[123]*Y[14]*Y[1])
    D=0.0+K[48]+K[49]+K[50]+K[51]+K[52]+K[53]
    YDOT[30]=F-(D*Y[30])
    # CHDHD
    F=(0.0+K[89]*Y[5]*Y[2]+K[95]*Y[7]*Y[2]+K[101]*Y[6]*Y[2]+K[107]*Y[15]*Y[1]
    +K[113]*Y[13]*Y[1]+K[119]*Y[14]*Y[1])
    D=0.0+K[24]+K[25]+K[26]+K[27]+K[28]+K[29]
    YDOT[31]=F-(D*Y[31])
    # CDHDH
    F=(0.0+K[91]*Y[5]*Y[2]+K[97]*Y[7]*Y[2]+K[103]*Y[6]*Y[2]+K[109]*Y[15]*Y[1]
    +K[115]*Y[13]*Y[1]+K[121]*Y[14]*Y[1])
    D=0.0+K[36]+K[37]+K[38]+K[39]+K[40]+K[41]
    YDOT[32]=F-(D*Y[32])
    # CHHDD
    F=(0.0+K[88]*Y[5]*Y[2]+K[94]*Y[7]*Y[2]+K[100]*Y[6]*Y[2]+K[106]*Y[15]*Y[1]
    +K[112]*Y[13]*Y[1]+K[118]*Y[14]*Y[1])
    D=0.0+K[18]+K[19]+K[20]+K[21]+K[22]+K[23]
    YDOT[33]=F-(D*Y[33])
    # QDDD
    F=0.0+K[197]*Y[47]+K[201]*Y[45]+K[205]*Y[48]+K[209]*Y[46]+K[210]*Y[50]
    D=0.0+K[276]*Y[1]+K[277]*Y[1]+K[278]*Y[1]+K[279]*Y[1]+K[280]*Y[2]
    YDOT[34]=F-(D*Y[34])
    # QDHHD
    F=(0.0+K[230]*Y[10]*Y[2]+K[236]*Y[11]*Y[2]+K[242]*Y[9]*Y[2]+K[248]*Y[21]*Y[1]
    +K[254]*Y[20]*Y[1]+K[260]*Y[22]*Y[1])
    D=0.0+K[170]+K[171]+K[172]+K[173]+K[174]+K[175]
    YDOT[35]=F-(D*Y[35])
    # QDHDH
    F=(0.0+K[231]*Y[10]*Y[2]+K[237]*Y[11]*Y[2]+K[243]*Y[9]*Y[2]+K[249]*Y[21]*Y[1]
    +K[255]*Y[20]*Y[1]+K[261]*Y[22]*Y[1])
    D=0.0+K[176]+K[177]+K[178]+K[179]+K[180]+K[181]
    YDOT[36]=F-(D*Y[36])
    # QHHDD
    F=(0.0+K[228]*Y[10]*Y[2]+K[234]*Y[11]*Y[2]+K[240]*Y[9]*Y[2]+K[246]*Y[21]*Y[1]
    +K[252]*Y[20]*Y[1]+K[258]*Y[22]*Y[1])
    D=0.0+K[158]+K[159]+K[160]+K[161]+K[162]+K[163]
    YDOT[37]=F-(D*Y[37])
    # QHDHD
    F=(0.0+K[229]*Y[10]*Y[2]+K[235]*Y[11]*Y[2]+K[241]*Y[9]*Y[2]+K[247]*Y[21]*Y[1]
    +K[253]*Y[20]*Y[1]+K[259]*Y[22]*Y[1])
    D=0.0+K[164]+K[165]+K[166]+K[167]+K[168]+K[169]
    YDOT[38]=F-(D*Y[38])
    # QDDHH
    F=(0.0+K[232]*Y[10]*Y[2]+K[238]*Y[11]*Y[2]+K[244]*Y[9]*Y[2]+K[250]*Y[21]*Y[1]
    +K[256]*Y[20]*Y[1]+K[262]*Y[22]*Y[1])
    D=0.0+K[182]+K[183]+K[184]+K[185]+K[186]+K[187]
    YDOT[39]=F-(D*Y[39])
    # QHDDH
    F=(0.0+K[233]*Y[10]*Y[2]+K[239]*Y[11]*Y[2]+K[245]*Y[9]*Y[2]+K[251]*Y[21]*Y[1]
    +K[257]*Y[20]*Y[1]+K[263]*Y[22]*Y[1])
    D=0.0+K[188]+K[189]+K[190]+K[191]+K[192]+K[193]
    YDOT[40]=F-(D*Y[40])
    # CDHDD
    F=0.0+K[126]*Y[15]*Y[2]+K[130]*Y[13]*Y[2]+K[134]*Y[14]*Y[2]+K[138]*Y[27]*Y[1]
    D=0.0+K[58]+K[59]+K[60]+K[61]
    YDOT[41]=F-(D*Y[41])
    # CDDHD
    F=0.0+K[125]*Y[15]*Y[2]+K[129]*Y[13]*Y[2]+K[133]*Y[14]*Y[2]+K[137]*Y[27]*Y[1]
    D=0.0+K[62]+K[63]+K[64]+K[65]
    YDOT[42]=F-(D*Y[42])
    # CDDDH
    F=0.0+K[124]*Y[15]*Y[2]+K[128]*Y[13]*Y[2]+K[132]*Y[14]*Y[2]+K[136]*Y[27]*Y[1]
    D=0.0+K[66]+K[67]+K[68]+K[69]
    YDOT[43]=F-(D*Y[43])
    # CHDDD
    F=0.0+K[127]*Y[15]*Y[2]+K[131]*Y[13]*Y[2]+K[135]*Y[14]*Y[2]+K[139]*Y[27]*Y[1]
    D=0.0+K[54]+K[55]+K[56]+K[57]
    YDOT[44]=F-(D*Y[44])
    # QDHDD
    F=0.0+K[266]*Y[21]*Y[2]+K[270]*Y[20]*Y[2]+K[274]*Y[22]*Y[2]+K[278]*Y[34]*Y[1]
    D=0.0+K[198]+K[199]+K[200]+K[201]
    YDOT[45]=F-(D*Y[45])
    # QDDDH
    F=0.0+K[264]*Y[21]*Y[2]+K[268]*Y[20]*Y[2]+K[272]*Y[22]*Y[2]+K[276]*Y[34]*Y[1]
    D=0.0+K[206]+K[207]+K[208]+K[209]
    YDOT[46]=F-(D*Y[46])
    # QHDDD
    F=0.0+K[267]*Y[21]*Y[2]+K[271]*Y[20]*Y[2]+K[275]*Y[22]*Y[2]+K[279]*Y[34]*Y[1]
    D=0.0+K[194]+K[195]+K[196]+K[197]
    YDOT[47]=F-(D*Y[47])
    # QDDHD
    F=0.0+K[265]*Y[21]*Y[2]+K[269]*Y[20]*Y[2]+K[273]*Y[22]*Y[2]+K[277]*Y[34]*Y[1]
    D=0.0+K[202]+K[203]+K[204]+K[205]
    YDOT[48]=F-(D*Y[48])
    # CDDDD
    F=0.0+K[140]*Y[27]*Y[2]
    D=0.0+K[70]
    YDOT[49]=F-(D*Y[49])
    # QDDDD
    F=0.0+K[280]*Y[34]*Y[2]
    D=0.0+K[210]
    YDOT[50]=F-(D*Y[50])
    return YDOT
#------------------------------------------------------------------------------------------------
# t=1 # For testing only
# deriv=dfdt(t,Y)
# print('derivatives=',deriv)


#------------------------------------------------------------------------------------------------
# SOLVE ODEs. Returned array Y has each element being an array for variable i at each time
#t_lower=0.0001
#t_upper=6.0
tint=np.linspace(t_lower,t_upper,num)  # num is number of time steps specified at top of program
soln=solve_ivp(dfdt,(t_lower,t_upper),Y0,t_eval=tint,atol=1.0e-11,rtol=1.0e-9)
#print(soln.t)
#print(soln.y)
#------------------------------------------------------------------------------------------------

print('')
print('FINAL ISOTOPOMERS')
i=1
while i< (len(Y0)):
    print("%i. %s \t = %.6e" %(i, Ynames[i],soln.y[i][num-1]))
    i=i+1
    
print('')
print('FINAL YDOT=',YDOT)

print('')
#print('K=',K)

# PROCESS RESULTS INTO IDENTIFIABLE ISOTOPOLOGUE PARAMETERS
d13C_CH4_t=1000.0*((soln.y[12]/soln.y[8])/VPDB -1.0)
dD_CH4_t=1000.0*((((soln.y[16]+soln.y[17]+soln.y[18]+soln.y[19])/4.0)/soln.y[8])/VSMOW -1.0)


# Create array of stochastic ratios for ISOTOPOLOGUE CH2D2 for each solution from time t.
# We have one stochastic ratio CH2D2/CH4 for each time.
RstochD2=np.ones(len(soln.t))
i=0
while i < len(soln.t):
    xH,xD,x12C,x13C,xst_CH4,xeq_CH4,alpha13D,alphaD2 = CH4_isotopologues(dD_CH4_t[i], d13C_CH4_t[i], TK)
    RstochD2[i]=xst_CH4[4]/xst_CH4[0]
    i=i+1
D12CH2D2_t=1000.0*(((soln.y[28]+soln.y[29]+soln.y[30]+soln.y[31]+soln.y[32]+soln.y[33])/soln.y[8])/RstochD2-1.0)
ratio_CH2D2_t=((soln.y[28]+soln.y[29]+soln.y[30]+soln.y[31]+soln.y[32]+soln.y[33])/soln.y[8])
delta_CH2D2_t=1000.0*(ratio_CH2D2_t/ratio_i_CH2D2_CH4 -1.0)  # little delta useful for Rayleigh plots

# Create array of stochastic ratios for 13CH3D for each solution from time t.
# We have one stochastic ratio 13CH3D/CH4 for each time.
Rstoch13CD=np.ones(len(soln.t))
i=0
while i < len(soln.t):
    xH,xD,x12C,x13C,xst_CH4,xeq_CH4,alpha13D,alphaD2 = CH4_isotopologues(dD_CH4_t[i], d13C_CH4_t[i], TK)
    Rstoch13CD[i]=xst_CH4[3]/xst_CH4[0]
    i=i+1
D13CH3D_t=1000.0*(((soln.y[23]+soln.y[24]+soln.y[25]+soln.y[26])/soln.y[8])/Rstoch13CD-1.0)
ratio_13CH3D_t=((soln.y[23]+soln.y[24]+soln.y[25]+soln.y[26])/soln.y[8])
delta_13CH3D_t=1000.0*(ratio_13CH3D_t/ratio_i_13CH3D_CH4 -1.0)

# F VALUES FOR RAYLEIGH PLOTS, fraction of CH4 remaining, at each time
F_CH4=soln.y[8]/Y0_logs[0]
minus_lnF_CH4=-np.log(F_CH4)
print('F=',F_CH4)

# USE F VALUES TO CUT THE TIME SEQUENCE OF VALUES TO THOSE GREATER THAN A REASONABLE VALUE
nseg_min_indices=np.where(F_CH4 >= minF)
nseg_min=np.amax(nseg_min_indices)
print('index for F > %.3e = %d '%(minF, nseg_min))

# PROCESS RESULTS FOR CH3: access single scalar in Y using Y[element][timestep], as in Y[5][1]
# yields CH3 concentration in the 2nd time step.
soln.y[3][0]=1.0e-12
d13C_CH3_t=1000.0*((soln.y[4]/soln.y[3])/VPDB-1.0)
dD_CH3_t=1000.0*((((soln.y[5]+soln.y[6]+soln.y[7])/soln.y[5])/3.0)/VSMOW -1.0)
print('')
#print("d13C CH3(t)=",d13C_CH3_t)
#print("dD CH3(t)=",dD_CH3_t)

# PROCESS RESULTS FOR D/H:
soln.y[1][0]=1.0e-10
dD_H_t=1000.0*((soln.y[2]/soln.y[1])/VSMOW-1.0)
#print("dD H=",dD_H_t)

# EQUILIBRIUM CURVE for plotting
Tref=np.linspace(273.0,1600.0,100)
G5=1/Tref
alpha13Dref=1+0.03555020*G5-433.038*G5**2+1.27021e6*G5**3-5.94804e8*G5**4+1.19663e11*G5**5-9.0723e12*G5**6
alphaD2ref=1+0.183798*G5-785.483*G5**2+1.056280e6*G5**3+9.37307e7*G5**4-8.91948e10*G5**5+9.90173e12*G5**6
D13CH3D_EQ=1000.0*(alpha13Dref-1.0)
D12CH2D2_EQ=1000.0*(alphaD2ref-1.0)

#------------------------------------------------------------------------------------------------
# MAKE PLOTS
#Molecule abundances vs time
plt.figure("N vs time",figsize=(7.0,5.0))
plt.plot(soln.t[0:nseg_min],soln.y[8,0:nseg_min],color='black',label='CH$_4$')
plt.plot(soln.t[0:nseg_min],soln.y[3,0:nseg_min],color='grey',label='CH$_3$')
#plt.xlim([0,4000.0])
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
plt.savefig('CH4_vs_time_v2',bbox_inches='tight',dpi=1000)

#d13C vs time
plt.figure("d13C vs time",figsize=(7.0,5.0))
plt.plot(soln.t[0:nseg_min],d13C_CH4_t[0:nseg_min],color='black',label='d13C CH4')
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
plt.savefig('d13C_CH4_vs_time_v2',bbox_inches='tight',dpi=1000)

#d13C vs F
#find line of best fit
lndelta=1000.0*np.log((d13C_CH4_t[0:nseg_min]/1000.0+1.0)/(d13C_CH4/1000.0+1.0))
result=linregress(minus_lnF_CH4[0:nseg_min],lndelta)
print("d13C vs -lnF slope=",result.slope)
print("intercept=",result.intercept)
alpha_fit=1.0-result.slope/1000.0
print('d13C alpha =', alpha_fit)
plt.figure("d13C-d13C vs F",figsize=(7.0,5.0))
plt.plot(minus_lnF_CH4[0:nseg_min],d13C_CH4_t[0:nseg_min]-d13C_CH4,color='black',label=r'best-fit alpha = %.6f' %alpha_fit)
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
plt.savefig('d13C_CH4_vs_F_v2',bbox_inches='tight',dpi=1000)

#dD vs time
plt.figure("dD vs time",figsize=(7.0,5.0))
plt.plot(soln.t[0:nseg_min],dD_CH4_t[0:nseg_min],color='black',label='dD CH4')
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
plt.savefig('dD_CH4_vs_time_v2',bbox_inches='tight',dpi=1000)


#dD vs F
#find line of best fit
lndelta=1000.0*np.log((dD_CH4_t[0:nseg_min]/1000.0+1.0)/(dD_CH4/1000.0+1.0))
result=linregress(minus_lnF_CH4[0:nseg_min],lndelta)
print("\ndD vs -lnF slope=",result.slope)
print("intercept=",result.intercept)
alpha_fit=1.0-result.slope/1000.0
print('dD alpha =', alpha_fit)
plt.figure("dD-dD vs F",figsize=(7.0,5.0))
plt.plot(minus_lnF_CH4[0:nseg_min],dD_CH4_t[0:nseg_min]-dD_CH4,color='black',label=r'best-fit alpha = %.6f' %alpha_fit)
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
plt.savefig('dD_CH4_vs_F_v2',bbox_inches='tight',dpi=1000)

#D12CH2D2 vs time
plt.figure("D12CH2D2 vs time",figsize=(7.0,5.0))
plt.plot(soln.t[0:nseg_min],D12CH2D2_t[0:nseg_min],color='black',label='DCH2D2')
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
plt.savefig('D12CH2D2_vs_time_v2',bbox_inches='tight',dpi=1000)

#dCH2D2 vs F, note dCH2D2 is relative to initial, so dCH2D2-dCH2D2o = dCH2D2
#find line of best fit
lndelta=1000.0*np.log(ratio_CH2D2_t[0:nseg_min]/ratio_i_CH2D2_CH4)
result=linregress(minus_lnF_CH4[0:nseg_min],lndelta)
print("\ndCH2D2 vs -lnF slope=",result.slope)
print("intercept=",result.intercept)
alpha_fit=1.0-result.slope/1000.0
print('dCH2D2 alpha =', alpha_fit)
plt.figure("dCH2D2 vs F",figsize=(7.0,5.0))
plt.plot(minus_lnF_CH4[0:nseg_min],delta_CH2D2_t[0:nseg_min],color='black',label=r'best-fit alpha = %.6f' %alpha_fit)
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
plt.savefig('dCH2D2_vs_F_v2',bbox_inches='tight',dpi=1000)

#D13CH3D vs time
plt.figure("D13CH3D vs time",figsize=(7.0,5.0))
#plt.plot(soln.t[0:nseg_min],D13CH3D_t,marker='o',markerfacecolor='white',linestyle='-',color='blue',label='13CH3D')
plt.plot(soln.t[0:nseg_min],D13CH3D_t[0:nseg_min],color='black',label='13CH3D')
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
plt.savefig('D13CH3D_vs_time_v2',bbox_inches='tight',dpi=1000)


#d13CH3D vs F, note d13CH3D is relative to initial, so value equals difference
#find line of best fit
lndelta=1000.0*np.log(ratio_13CH3D_t[0:nseg_min]/ratio_i_13CH3D_CH4)
result=linregress(minus_lnF_CH4[0:nseg_min],lndelta)
print("\nd13CH3D vs -lnF slope=",result.slope)
print("intercept=",result.intercept)
alpha_fit=1.0-result.slope/1000.0
print('d13CH3D alpha =', alpha_fit)
plt.figure("d13CH3D vs F",figsize=(7.0,5.0))
plt.plot(minus_lnF_CH4[0:nseg_min],delta_13CH3D_t[0:nseg_min],color='black',label=r'best-fit alpha = %.6f' %alpha_fit)
##plt.xlim([0,4000.0])
plt.xlabel('-lnF',fontsize=14,labelpad=12)
plt.ylabel('$\delta^13$CH$_3$D$_2$ relative to initial',labelpad=11,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yticks(fontsize=13)
plt.tick_params(which='minor',width=0.75)
plt.legend(loc='best', frameon=False)
#plt.ylim([0.0,ylimit])
#plt.xlim([350.0,2600])
#plt.minorticks_on()
#plt.tick_params(which='both',bottom=True,top=True,left=True,right=True)
plt.savefig('d13CH3D_vs_F_v2',bbox_inches='tight',dpi=1000)


#dD CH4 vs d13C CH4
plt.figure("dD vs d13C",figsize=(7.0,5.0))
#plt.plot(d13C_CH4_t[1:num-1],dD_CH4_t[1:num-1],marker='o',markerfacecolor='white',linestyle='-',color='blue',label='Model')
plt.plot(d13C_CH4_t[0:nseg_min][1:num-1],dD_CH4_t[0:nseg_min][1:num-1],color='black',label='Model')
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
plt.savefig('dD_vs_d13C_CH4_v2',bbox_inches='tight',dpi=1000)

#D12CH2D2 vs D13CH3D
plt.figure("D12CH2D2 vs D13CH3D",figsize=(7.0,5.0))
plt.plot(D13CH3D_EQ,D12CH2D2_EQ,color='grey',label='Equilibrium')
#plt.plot(D13CH3D_t,D12CH2D2_t,marker='o',markerfacecolor='white',linestyle='-',color='blue',label='Model')
plt.plot(D13CH3D_t[0:nseg_min],D12CH2D2_t[0:nseg_min],linestyle='-',color='black',label='Model')
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
plt.savefig('D12CH2D2_vs_D13CH3D_v2',bbox_inches='tight',dpi=1000)

plt.show()

a_file = open('F_CH4_output_v2.txt', 'w')
for i in range(0,nseg_min):
    a_file.write("%10.5e\n " % F_CH4[i])
a_file.close()

a_file = open('CH4_moles_output_vw.txt', 'w')
for i in range(0,nseg_min):
    a_file.write("%10.5e\n " % soln.y[0][i])
a_file.close()

a_file = open('D12CH2D2_output_v2.txt', 'w')
for i in range(0,nseg_min):
    a_file.write("%10.5e\n " % D12CH2D2_t[i])
a_file.close()

a_file = open('D13CH3D_output_v2.txt', 'w')
for i in range(0,nseg_min):
    a_file.write("%10.5e\n " % D13CH3D_t[i])
a_file.close()

a_file = open('dD_CH4_output_v2.txt', 'w')
for i in range(0,nseg_min):
    a_file.write("%10.5e\n " % dD_CH4_t[i])
a_file.close()

a_file = open('d13C_CH4_output_v2.txt', 'w')
for i in range(0,nseg_min):
    a_file.write("%10.5e\n " % d13C_CH4_t[i])
a_file.close()

print("")
print("done")
print("")


