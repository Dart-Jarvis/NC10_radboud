# Code to convert the frequencies I got from the ab initio calculations
import pandas as pd
import numpy as np
import math

c=2.99792458e10   # speed of light in cm/s
h=6.62607015*10**(-34) # Plank constant, J/s
kb=1.380649*10**(-23) # Boltzmann constant, J/K
hbar=6.62607015*10**(-34)/(2*math.pi)
R=8.314

# Read data
ts_freq_df=pd.read_csv("smmo_frequency_ts.csv")
ts_freq_df2=pd.read_csv("CH4_OH_ts.csv")
ch4_freq_df=pd.read_csv("smmo_frequency_ch4.csv")
ch4_freq_df2=pd.read_csv("OH_model.csv") # Calculation results from Haghneghdar et al., 2017

def sort_frequencies(df: pd.DataFrame):
    """
    Given a frequency dataframe with columns ['Isotopologue', 'Frequency'],
    return a dictionary mapping each isotopologue to (imaginary, real) frequencies.
    
    - Imaginary: negative cm^-1 values
    - Real: positive cm^-1 values
    """
    # Make sure Frequency is numeric
    df = df.copy()
    df["freq_cm"] = pd.to_numeric(df["Frequency"], errors="coerce")
    df = df.dropna(subset=["freq_cm"])
    
    result = {}
    for iso, sub in df.groupby("Isotopologue"):
        freqs = sub["freq_cm"].to_numpy()
        imag = freqs[freqs < 0]
        real = freqs[freqs > 0]
        result[iso] = {
            "imaginary": imag.tolist(),
            "real": real.tolist()
        }
    return result

# sMMO ab initio
ts_freq=sort_frequencies(ts_freq_df)
ch4_freq=sort_frequencies(ch4_freq_df)

# CH4-OH from Mojhgan's model
ts_freq_OH=sort_frequencies(ts_freq_df2) 
ch4_freq_OH=sort_frequencies(ch4_freq_df2)

# Calculate the partition function from Lasaga 1991, GRL
def calc_rpf(freq, T):
    rpf={}
    for key in freq.keys():
        rpf_temp=1.0
        for j in freq[key]["real"]: 
            # Make sure to rule out the imaginary frequency (Mode 1) when calculating rpf
            nu = abs(j) * c  # convert to Hz
            rpf_temp=rpf_temp*h*nu/(kb*T)/(1-math.exp((-h*nu)/(kb*T)))
        rpf[key]=rpf_temp
    return rpf

# Calculate the tunneling factor kappa, based on Wigner correction
def calc_kappa(freq,T):
    kappa={}
    for key in freq.keys():
        kappa[key]=1+(hbar*2*math.pi*c*abs(freq[key]["imaginary"][0])/kb/T)**2/24
    return kappa

kappa_ts=calc_kappa(ts_freq,T)
kappa_ts_OH=calc_kappa(ts_freq_OH,T)

# Calculate the KIE between one isotopologue and ch4, iso1 is the name in ts
# iso2 is the name in ch4, 
# freq is a dictionary with all vibrational frequencies for ch4 in reactants
# freq_ts is a dictionary with all vibrational frequencies for ts
def calc_rpfr(freq_ts:dict, freq:dict, kappa:dict, iso1:str,iso2:str, tunneling:bool): 
    # Calculate rpf first
    ts_rpf=calc_rpf(freq_ts,T)
    ch4_rpf=calc_rpf(freq,T)    
    # Identify the imaginary frequency ratio
    img_freq_ratio=freq_ts[iso1]["imaginary"][0]/freq_ts["CHHHH"]["imaginary"][0]
    # Calculate the tunneling factor
    if tunneling==True:
        tun=kappa[iso1]/kappa["CHHHH"]
    else:
        tun=1 
    # Calculate delta E, make sure to convert the frequency to Hz first
    deltaE=0.5*h*c*sum(freq_ts[iso1]["real"])-0.5*h*c*sum(freq_ts["CHHHH"]["real"])-(0.5*h*c*sum(freq[iso2]["real"])-0.5*h*c*sum(freq["CHHHH"]["real"]))
    # Calculate RPFR
    rpfr=img_freq_ratio*tun*ts_rpf[iso1]/ts_rpf["CHHHH"]*ch4_rpf["CHHHH"]/ch4_rpf[iso2]*math.exp(-deltaE/(kb*T))
    return rpfr

# # For Mojhgan's model for CH4-OH oxidation
T=25+273.15
a13_OH=calc_rpfr(ts_freq_OH,ch4_freq_OH,kappa_ts_OH,"QHHHH","QHHHH",True)
aH_OH=1/4*(calc_rpfr(ts_freq_OH,ch4_freq_OH,kappa_ts_OH,"CHHHD","CHHHD", True)+
        calc_rpfr(ts_freq_OH,ch4_freq_OH,kappa_ts_OH,"CHHDH","CHHHD", True)+
        calc_rpfr(ts_freq_OH,ch4_freq_OH,kappa_ts_OH,"CHDHH","CHHHD", True)+   
        calc_rpfr(ts_freq_OH,ch4_freq_OH,kappa_ts_OH,"CDHHH","CHHHD", True))

print(a13_OH, aH_OH)

a13=calc_rpfr(ts_freq,ch4_freq,kappa_ts,"QHHHH","QHHHH",True)
aH=1/4*(calc_rpfr(ts_freq,ch4_freq,kappa_ts_OH,"CHHHD","CHHHD", True)+
        calc_rpfr(ts_freq,ch4_freq,kappa_ts_OH,"CHHDH","CHHHD", True)+
        calc_rpfr(ts_freq,ch4_freq,kappa_ts_OH,"CHDHH","CHHHD", True)+   
        calc_rpfr(ts_freq,ch4_freq,kappa_ts_OH,"CDHHH","CHHHD", True))
print("sMMO ab initio", "\n", a13, "\n", aH)