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

l=0
for i in iso:
    length=len(ts_freq_df[ts_freq_df['Isotopologue']==i])
    if length>=l:
        l=length

ts_freq=np.zeros((l,len(iso)))

for i in range(len(iso)):
    temp_dt=ts_freq_df[ts_freq_df['Isotopologue']==iso[i]]
    ts_freq[:,i]=temp_dt["Frequency"]

T=273.15+25 # Temperature in K
# Calculate the partition function from Lasaga 1991, GRL
def calc_rpf(freq, T):
    rpf=np.zeros(freq.shape[1])
    c = 2.99792458e10   # speed of light in cm/s
    h=6.626*10**(-34) # Plank constant, J/s
    kb=1.380649*10**(-23) # Boltzmann constant, J/K
    for i in range(freq.shape[1]):
        rpf_temp=1.0
        for j in range(freq.shape[0]):
            nu = freq[j, i] * c  # convert to Hz
            rpf_temp=rpf_temp*h*nu/(kb*T)/(1-math.exp((-h*nu)/(kb*T)))
        rpf[i]=rpf_temp
    return rpf

ts_rpf=calc_rpf(ts_freq,T)
print(ts_rpf[16]/ts_rpf[0])
