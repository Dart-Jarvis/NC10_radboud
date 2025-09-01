# Code to convert the frequencies I got from the ab initio calculations
import pandas as pd
import numpy as np
import math

ts_freq_df=pd.read_csv("smmo_frequency_ts.csv")
ch4_freq_df=pd.read_csv("smmo_frequency_ch4.csv")

# Extract frequencies of isotopologues
iso=["CHHHH","CHHHD","CHHDH","CHDHH","CDHHH",
     "CHHDD","CHDHD","CDHHD","CDHDH","CDDHH",
     "CHDDH","CDDDH","CDDHD","CDHDD","CHDDD",
     "CDDDD","QHHHH","QHHHD","QHHDH","QHDHH",
     "QDHHH","QHHDD","QHDHD","QDHHD","QDHDH",
     "QDDHH","QHDDH","QDDDH","QDDHD","QDHDD",
     "QHDDD","QDDDD"]

iso_ch4=["CHHHH","CHHHD","CHHDD","CHDDD","CDDDD",
         "QHHHH","QHHHD","QHHDD","QHDDD","QDDDD"]

def read_freq(df,iso_name):
    l=0
    for i in iso_name:
        length=len(df[df['Isotopologue']==i])
        if length>=l:
            l=length

    freq_array=np.zeros((l,len(iso_name)))

    for j in range(len(iso_name)):
        temp_dt=df[df['Isotopologue']==iso_name[j]]
        freq_array[:,j]=temp_dt["Frequency"]
    return freq_array

ts_freq=read_freq(ts_freq_df,iso)
ch4_freq=read_freq(ch4_freq_df,iso_ch4)

T=273.15+25 # Temperature in K
# Calculate the partition function from Lasaga 1991, GRL
def calc_rpf(freq, T):
    rpf=np.zeros(freq.shape[1])
    c = 2.99792458e10   # speed of light in cm/s
    h=6.626*10**(-34) # Plank constant, J/s
    kb=1.380649*10**(-23) # Boltzmann constant, J/K
    for i in range(freq.shape[1]):
        rpf_temp=1.0
        for j in range(1,freq.shape[0]): 
            # Make sure to rule out the imaginary frequency (Mode 1) when calculating rpf
            nu = freq[j, i] * c  # convert to Hz
            rpf_temp=rpf_temp*h*nu/(kb*T)/(1-math.exp((-h*nu)/(kb*T)))
        rpf[i]=rpf_temp
    return rpf

# Calculate the KIE between two isotopologues, iso2 is normally CH4, iso2=0;
# iso3 and 4 are for CH4 isotopologues, iso4 is normally CH4, iso4=0
def calc_rpfr(iso1,iso2,iso3,iso4): 
    h=6.626*10**(-34) # Plank constant, J/s
    c= 2.99792458e10   # speed of light in cm/s
    kb=1.380649*10**(-23) # Boltzmann constant, J/K 
    # Identify the imaginary frequency ratio
    img_freq_ratio=ts_freq[0,iso1]/ts_freq[0,iso2]
    # Calculate the tunneling factor, set it at 1 now
    tun=1.0
    # Calculate delta E, make sure to convert the frequency to Hz first
    deltaE=0.5*h*c*sum(ts_freq[:,iso1])-0.5*h*c*sum(ts_freq[:,iso2])-(0.5*h*c*sum(ch4_freq[:,iso3])-0.5*h*c*sum(ch4_freq[:,iso4]))
    # Calculate RPFR
    rpfr=img_freq_ratio*tun*ts_rpf[iso1]/ts_rpf[iso2]*ch4_rpf[iso4]/ch4_rpf[iso3]*math.exp(-deltaE/(kb*T))
    return rpfr

ts_rpf=calc_rpf(ts_freq,T)
ch4_rpf=calc_rpf(ch4_freq,T)

# Calculate carbon isotope fractionation
a13=calc_rpfr(16,0,5,0) # 0.9713 at 27 oC, from Li et al., 2024, GCA
a2H=1/4*(calc_rpfr(1,0,1,0)+calc_rpfr(2,0,1,0)
         +calc_rpfr(3,0,1,0)+calc_rpfr(4,0,1,0))# Primary + secondary isotope effects
aQH3D=1/4*(calc_rpfr(17,0,6,0)+calc_rpfr(18,0,6,0)
           +calc_rpfr(19,0,6,0)+calc_rpfr(20,0,6,0))
aCH2D2=1/6*(calc_rpfr(5,0,2,0)+calc_rpfr(6,0,2,0)+calc_rpfr(7,0,2,0)
            +calc_rpfr(8,0,2,0)+calc_rpfr(9,0,2,0)+calc_rpfr(10,0,2,0))

print(a13,"\n",a2H, "\n", aQH3D, "\n", aCH2D2)