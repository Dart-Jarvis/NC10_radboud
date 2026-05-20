# Code to convert the frequencies I got from the ab initio calculations
import pandas as pd
import numpy as np
import math

c=2.99792458e10   # speed of light in cm/s
h=6.62607015*10**(-34) # Plank constant, J/s
kb=1.380649*10**(-23) # Boltzmann constant, J/K
hbar=6.62607015*10**(-34)/(2*math.pi)
R=8.314
T=30+273.15
# Read data
ts_freq_sMMO_df=pd.read_csv("smmo_frequency_ts.csv")
ts_freq_df2=pd.read_csv("CH4_OH_ts.csv")
ch4_freq_df=pd.read_csv("smmo_frequency_ch4.csv")
ts_freq_mcr_df=pd.read_csv("mcr_frequency_ts.csv")
ch4_freq_df2=pd.read_csv("OH_model.csv") # Calculation results from Haghneghdar et al., 2017
ts_mcr_full_df=pd.read_csv("mcr_ts_freq_filtered.csv")
mcr_int_df=pd.read_csv("mcr_int_freq.csv")

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
ts_freq_sMMO=sort_frequencies(ts_freq_sMMO_df)
ch4_freq_sMMO=sort_frequencies(ch4_freq_df)
# mcr ab initio
ts_freq_mcr=sort_frequencies(ts_freq_mcr_df)
ts_freq_mcr_full=sort_frequencies(ts_mcr_full_df)
# CH4-OH from Mojhgan's model
ts_freq_OH=sort_frequencies(ts_freq_df2) 
ch4_freq_OH=sort_frequencies(ch4_freq_df2)
# CH4 intermediate with mcr
mcr_int_freq=sort_frequencies(mcr_int_df)

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

def calc_eq_rpf(freq, T):
    rpf_eq={}
    for key in freq.keys():
        rpf_temp=1.0
        for j in freq[key]["real"]: 
            # Make sure to rule out the imaginary frequency (Mode 1) when calculating rpf
            ui = h*abs(j)*c/(kb*T)  # convert to Hz
            rpf_temp=rpf_temp*ui*math.exp(-ui/2)/(1-math.exp(-ui))
        rpf_eq[key]=rpf_temp
    return rpf_eq

# Calculate the tunneling factor kappa, based on Wigner correction
def calc_kappa(freq,T):
    kappa={}
    for key in freq.keys():
        kappa[key]=1+(hbar*2*math.pi*c*abs(freq[key]["imaginary"][0])/kb/T)**2/24
    return kappa

kappa_ts_sMMO=calc_kappa(ts_freq_sMMO,T)
kappa_ts_mcr=calc_kappa(ts_freq_mcr,T)
kappa_ts_mcr_full=calc_kappa(ts_freq_mcr_full,T)
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

def calc_rpfr_eq(freq_r,freq_p,iso1,iso2):
    r_rpf=calc_eq_rpf(freq_r,T)
    p_rpf=calc_eq_rpf(freq_p,T)
    rpfr_eq=p_rpf[iso1]/p_rpf["CHHHH"]*r_rpf["CHHHH"]/r_rpf[iso2]
    return rpfr_eq

def convert_alphaeq(freq_r,freq_p,name):
    a13=calc_rpfr_eq(freq_r,freq_p,"QHHHH","QHHHH")
    aH=1/4*(calc_rpfr_eq(freq_r,freq_p,"CHHHD","CHHHD")+
            calc_rpfr_eq(freq_r,freq_p,"CHHDH","CHHHD")+
            calc_rpfr_eq(freq_r,freq_p,"CHDHH","CHHHD")+
            calc_rpfr_eq(freq_r,freq_p,"CDHHH","CHHHD"))
    aCD=1/4*(calc_rpfr_eq(freq_r,freq_p,"QHHHD","QHHHD")+
             calc_rpfr_eq(freq_r,freq_p,"QHHDH","QHHHD")+
             calc_rpfr_eq(freq_r,freq_p,"QHDHH","QHHHD")+
             calc_rpfr_eq(freq_r,freq_p,"QDHHH","QHHHD"))
    aDD=1/6*(calc_rpfr_eq(freq_r,freq_p,"CHHDD","CHHDD")+
             calc_rpfr_eq(freq_r,freq_p,"CHDHD","CHHDD")+
             calc_rpfr_eq(freq_r,freq_p,"CDHHD","CHHDD")+
             calc_rpfr_eq(freq_r,freq_p,"CDHDH","CHHDD")+
             calc_rpfr_eq(freq_r,freq_p,"CDDHH","CHHDD")+
             calc_rpfr_eq(freq_r,freq_p,"CHDDH","CHHDD"))
    aHp=calc_rpfr_eq(freq_r,freq_p,"CHHHD","CHHHD")
    aHs=1/3*(calc_rpfr_eq(freq_r,freq_p,"CHHDH","CHHHD")+
             calc_rpfr_eq(freq_r,freq_p,"CHDHH","CHHHD")+
             calc_rpfr_eq(freq_r,freq_p,"CDHHH","CHHHD"))
    aCDp=calc_rpfr_eq(freq_r,freq_p,"QHHHD","QHHHD")
    aCDs=1/3*(calc_rpfr_eq(freq_r,freq_p,"QHHDH","QHHHD")+
             calc_rpfr_eq(freq_r,freq_p,"QHDHH","QHHHD")+
             calc_rpfr_eq(freq_r,freq_p,"QDHHH","QHHHD"))
    aDDp=1/3*(calc_rpfr_eq(freq_r,freq_p,"CHHDD","CHHDD")+
             calc_rpfr_eq(freq_r,freq_p,"CHDHD","CHHDD")+
             calc_rpfr_eq(freq_r,freq_p,"CDHHD","CHHDD"))
    aDDs=1/3*(calc_rpfr_eq(freq_r,freq_p,"CDHDH","CHHDD")+
             calc_rpfr_eq(freq_r,freq_p,"CDDHH","CHHDD")+
             calc_rpfr_eq(freq_r,freq_p,"CHDDH","CHHDD"))
    print(name,"\n","alpha13C: ", a13,"\n","alphaD: ", aH,"\n", "alpha13CD: ", aCD,"\n","alphaDD: ", aDD,"\n")
    return a13,aH,aCD,aDD,aHp,aHs,aCDp,aCDs,aDDp,aDDs

def convert_alpha(ts_freq,ch4_freq,kappa, name):
    a13=calc_rpfr(ts_freq,ch4_freq,kappa,"QHHHH","QHHHH",True)
    aH=1/4*(calc_rpfr(ts_freq,ch4_freq,kappa,"CHHHD","CHHHD", True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHHDH","CHHHD", True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHDHH","CHHHD", True)+   
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHHH","CHHHD", True))
    aCD=1/4*(calc_rpfr(ts_freq,ch4_freq,kappa,"QHHHD","QHHHD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"QHHDH","QHHHD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"QHDHH","QHHHD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"QDHHH","QHHHD",True))
    aDD=1/6*(calc_rpfr(ts_freq,ch4_freq,kappa,"CHHDD","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHDDH","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDDHH","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHDHD","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHHD","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHDH","CHHDD",True))
    aHp=calc_rpfr(ts_freq,ch4_freq,kappa,"CHHHD","CHHHD", True)
    aHs=1/3*(calc_rpfr(ts_freq,ch4_freq,kappa,"CHHDH","CHHHD", True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHDHH","CHHHD", True)+   
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHHH","CHHHD", True))
    aCDp=calc_rpfr(ts_freq,ch4_freq,kappa,"QHHHD","QHHHD",True)
    aCDs=1/3*(calc_rpfr(ts_freq,ch4_freq,kappa,"QHHDH","QHHHD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"QHDHH","QHHHD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"QDHHH","QHHHD",True))
    aDDp=1/3*(calc_rpfr(ts_freq,ch4_freq,kappa,"CHHDD","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHDHD","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHHD","CHHDD",True))
    aDDs=1/3*(calc_rpfr(ts_freq,ch4_freq,kappa,"CHDDH","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDDHH","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHDH","CHHDD",True))
    print(name,"\n","alpha13C: ", a13,"\n","alphaD: ", aH,"\n", "alpha13CD: ", aCD,"\n","alphaDD: ", aDD,"\n")
    print("aHp: ", aHp, "\n", "aHs: ", aHs, "\n", "aCDp: ", aCDp, "\n", "aCDs: ", aCDs, "\n", "aDDp: ", aDDp, "\n", "aDDs: ", aDDs)
    return a13,aH,aCD,aDD

def convert_alpha_int(ts_freq,ch4_freq,kappa, name):
    a13=calc_rpfr(ts_freq,ch4_freq,kappa,"QHHHH","QHHHH",True)
    aH=1/4*(calc_rpfr(ts_freq,ch4_freq,kappa,"CHHHD","CHHHD", True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHHDH","CHHDH", True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHDHH","CHDHH", True)+   
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHHH","CDHHH", True))
    aCD=1/4*(calc_rpfr(ts_freq,ch4_freq,kappa,"QHHHD","QHHHD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"QHHDH","QHHDH",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"QHDHH","QHDHH",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"QDHHH","QDHHH",True))
    aDD=1/6*(calc_rpfr(ts_freq,ch4_freq,kappa,"CHHDD","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHDDH","CHDDH",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDDHH","CDDHH",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHDHD","CHDHD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHHD","CDHHD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHDH","CDHDH",True))
    aHp=calc_rpfr(ts_freq,ch4_freq,kappa,"CHHHD","CHHHD", True)
    aHs=1/3*(calc_rpfr(ts_freq,ch4_freq,kappa,"CHHDH","CHHDH", True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHDHH","CHDHH", True)+   
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHHH","CDHHH", True))
    aCDp=calc_rpfr(ts_freq,ch4_freq,kappa,"QHHHD","QHHHD",True)
    aCDs=1/3*(calc_rpfr(ts_freq,ch4_freq,kappa,"QHHDH","QHHDH",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"QHDHH","QHDHH",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"QDHHH","QDHHH",True))
    aDDp=1/3*(calc_rpfr(ts_freq,ch4_freq,kappa,"CHHDD","CHHDD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CHDHD","CHDHD",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHHD","CDHHD",True))
    aDDs=1/3*(calc_rpfr(ts_freq,ch4_freq,kappa,"CHDDH","CHDDH",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDDHH","CDDHH",True)+
            calc_rpfr(ts_freq,ch4_freq,kappa,"CDHDH","CDHDH",True))
    print(name,"\n","alpha13C: ", a13,"\n","alphaD: ", aH,"\n", "alpha13CD: ", aCD,"\n","alphaDD: ", aDD,"\n")
    print("aHp: ", aHp, "\n", "aHs: ", aHs, "\n", "aCDp: ", aCDp, "\n", "aCDs: ", aCDs, "\n", "aDDp: ", aDDp, "\n", "aDDs: ", aDDs)
    return a13,aH,aCD,aDD

a13_OH,aH_OH,aCD_OH,aDD_OH=convert_alpha(ts_freq_OH,ch4_freq_OH,kappa_ts_OH, "OH")
print("\n")
a13_sMMO,aH_sMMO,aCD_sMMO,aDD_sMMO=convert_alpha(ts_freq_sMMO,ch4_freq_sMMO,kappa_ts_sMMO, "sMMO")
print("\n")
a13_mcr,aH_mcr,aCD_mcr,aDD_mcr=convert_alpha(ts_freq_mcr,ch4_freq_sMMO,kappa_ts_mcr, "mcr")
print("\n")
a13_mcr_full,aH_mcr_full,aCD_mcr_full,aDD_mcr_full=convert_alpha(ts_freq_mcr_full,mcr_int_freq,kappa_ts_mcr_full, "mcr-full")
print("\n")
a13_eq_int,aH_eq_int,aCD_eq_int,aDD_eq_int,aHp_eq,aHs_eq,aCDp_eq,aCDs_eq,aDDp_eq,aDDs_eq=convert_alphaeq(ch4_freq_sMMO,mcr_int_freq,"Equilibrium-ch4->int")

print("Data from Scheller:")
print("a13C:", 1/1.039)
print("aH:", 0.7435)
print("aHp:", 0.77398)
print("aHs:", 0.80864)
print("\n")

a13_mcr_tot=a13_mcr_full*a13_eq_int
aH_mcr_tot=aH_mcr_full*aH_eq_int
aCD_mcr_tot=aCD_mcr_full*aCD_eq_int
aDD_mcr_tot=aDD_mcr_full*aDD_eq_int
aHp_mcr_tot=aHp_eq*0.5202677142901856 
aHs_mcr_tot=aHs_eq*0.8340612886584756
aCDp_mcr_tot=aCDp_eq*0.4844256973210034
aCDs_mcr_tot=aCDs_eq*0.7814320692232366
aDDp_mcr_tot=aDDp_eq*0.4292911654029593
aDDs_mcr_tot=aDDs_eq*0.6971721934051316

print("Full mcr reaction")
print("a13C:", a13_mcr_tot)
print("aH:", aH_mcr_tot)
print("aCD:", aCD_mcr_tot)
print("aDD:", aDD_mcr_tot)
print("aHp:", aHp_mcr_tot)
print("aHs:", aHs_mcr_tot)
print("aCDp:", aCDp_mcr_tot)
print("aCDs:", aCDs_mcr_tot)
print("aDDp:", aDDp_mcr_tot)
print("aDDs:", aDDs_mcr_tot)
print("\n")


