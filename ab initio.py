# Code to convert the frequencies I got from the ab initio calculations
import pandas as pd
import numpy as np
import math

ts_freq_df=pd.read_csv("smmo_frequency_ts.csv")
ts_freq_df2=pd.read_csv("CH4_OH_ts.csv")
ch4_freq_df=pd.read_csv("smmo_frequency_ch4.csv")
ch4_freq_df2=pd.read_csv("OH_model.csv") # Calculation results from Haghneghdar et al., 2017


# Extract frequencies of isotopologues
# iso=["CHHHH","CHHHD","CHHDH","CHDHH","CDHHH",
#      "CHHDD","CHDHD","CDHHD","CDHDH","CDDHH",
#      "CHDDH","CDDDH","CDDHD","CDHDD","CHDDD",
#      "CDDDD","QHHHH","QHHHD","QHHDH","QHDHH",
#      "QDHHH","QHHDD","QHDHD","QDHHD","QDHDH",
#      "QDDHH","QHDDH","QDDDH","QDDHD","QDHDD",
#      "QHDDD","QDDDD"]

# iso_ch4=["CHHHH","CHHHD","CHHDD","CHDDD","CDDDD",
#          "QHHHH","QHHHD","QHHDD","QHDDD","QDDDD"]

# # Isotopologue names in Mojhgan's models for CH4-OH
# iso2=["CHHHH","CHHHD","CHHDH","CHDHH","CDHHH",
#      "CHHDD","CHDHD","CDHHD","CDHDH","CDDHH",
#      "CHDDH","QHHHH","QHHHD","QHHDH","QHDHH",
#      "QDHHH"]

# iso_ch4_2=["CHHHH","CHHHD","CHHDD",
#          "QHHHH","QHHHD"]

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
ts_freq2=sort_frequencies(ts_freq_df2) 
ch4_freq2=sort_frequencies(ch4_freq_df2)

# def read_freq(df,iso_name):
#     l=0
#     for i in iso_name:
#         length=len(df[df['Isotopologue']==i])
#         if length>=l:
#             l=length

#     freq_array=np.zeros((l,len(iso_name)))

#     for j in range(len(iso_name)):
#         temp_dt=df[df['Isotopologue']==iso_name[j]]
#         freq_array[:,j]=temp_dt["Frequency"]
#     return freq_array

# ts_freq=read_freq(ts_freq_df,iso)
# ts_freq2=read_freq(ts_freq_df2,iso2)
# ch4_freq=read_freq(ch4_freq_df,iso_ch4)
# ch4_freq2=read_freq(ch4_freq_df2,iso_ch4_2)


T=273.15+30 # Temperature in K
# Calculate the partition function from Lasaga 1991, GRL
def calc_rpf(freq, T, img):
    rpf=np.zeros(freq.shape[1])
    c = 2.99792458e10   # speed of light in cm/s
    h=6.62607015*10**(-34) # Plank constant, J/s
    kb=1.380649*10**(-23) # Boltzmann constant, J/K
    for i in range(freq.shape[1]):
        rpf_temp=1.0
        if img==1:
            for j in range(1,freq.shape[0]): 
                # Make sure to rule out the imaginary frequency (Mode 1) when calculating rpf
                nu = abs(freq[j, i]) * c  # convert to Hz
                rpf_temp=rpf_temp*h*nu/(kb*T)/(1-math.exp((-h*nu)/(kb*T)))
        elif img==0:
            for j in range(freq.shape[0]): 
                # No imaginary frequency for CH4 molecule
                nu = abs(freq[j, i]) * c  # convert to Hz
                rpf_temp=rpf_temp*h*nu/(kb*T)/(1-math.exp((-h*nu)/(kb*T)))

        rpf[i]=rpf_temp
    return rpf

# Calculate the tunneling factor kappa, based on Wigner correction
def calc_kappa(freq,T):
    kappa=np.zeros(freq.shape[1])
    hbar=6.62607015*10**(-34)/(2*math.pi)
    c= 2.99792458e10   # speed of light in cm/s
    kb=1.380649*10**(-23) # Boltzmann constant, J/K
    for i in range(len(kappa)):
        kappa[i]=1+(hbar*2*math.pi*c*freq[0,i]/kb/T)**2/24
    return kappa

kappa_ts=calc_kappa(ts_freq,T)

# Calculate the KIE between two isotopologues, iso2 is normally CH4, iso2=0;
# iso3 and 4 are for CH4 isotopologues, iso4 is normally CH4, iso4=0
def calc_rpfr(iso1,iso2,iso3,iso4,tunneling): 
    h=6.62607015*10**(-34) # Plank constant, J/s
    c= 2.99792458e10   # speed of light in cm/s
    kb=1.380649*10**(-23) # Boltzmann constant, J/K
    # R=8.314  # Ideal gas constant J/(mol*K)
    # Identify the imaginary frequency ratio
    img_freq_ratio=ts_freq[0,iso1]/ts_freq[0,iso2]
    # Calculate the tunneling factor
    if tunneling==1:
        tun=kappa_ts[iso1]/kappa_ts[iso2]
    else:
        tun=1
    
    # Calculate delta E, make sure to convert the frequency to Hz first
    deltaE=0.5*h*c*sum(ts_freq[1:,iso1])-0.5*h*c*sum(ts_freq[1:,iso2])-(0.5*h*c*sum(ch4_freq[:,iso3])-0.5*h*c*sum(ch4_freq[:,iso4]))
    # Calculate RPFR
    rpfr=img_freq_ratio*tun*ts_rpf[iso1]/ts_rpf[iso2]*ch4_rpf[iso4]/ch4_rpf[iso3]*math.exp(-deltaE/(kb*T))
    return rpfr

ts_rpf=calc_rpf(ts_freq,T,1) # Has imaginary frequency for transition state
ch4_rpf=calc_rpf(ch4_freq,T,0) # No imaginary frequency for ch4 molecule

# For Mojhgan's model for CH4-OH oxidation
T=25+273.15
ts_rpf2=calc_rpf(ts_freq2,T,1)
ch4_rpf2=calc_rpf(ch4_freq2,T,0)

# Calculate carbon isotope fractionation
a13=calc_rpfr(16,0,5,0,1) # 0.9713 at 27 oC, from Li et al., 2024, GCA
a2H=1/4*(calc_rpfr(1,0,1,0,1)+calc_rpfr(2,0,1,0,1)
         +calc_rpfr(3,0,1,0,1)+calc_rpfr(4,0,1,0,1))# Primary + secondary isotope effects
aQH3D=1/4*(calc_rpfr(17,0,6,0,1)+calc_rpfr(18,0,6,0,1)
           +calc_rpfr(19,0,6,0,1)+calc_rpfr(20,0,6,0,1))
aCH2D2=1/6*(calc_rpfr(5,0,2,0,1)+calc_rpfr(6,0,2,0,1)+calc_rpfr(7,0,2,0,1)
            +calc_rpfr(8,0,2,0,1)+calc_rpfr(9,0,2,0,1)+calc_rpfr(10,0,2,0,1))
aCD4=calc_rpfr(31,0,9,0,1)
# Calculate CH4-OH results for testing
ts_rpf=calc_rpf(ts_freq2,T,1) # Has imaginary frequency for transition state
ch4_rpf=calc_rpf(ch4_freq2,T,0) # No imaginary frequency for ch4 molecule
a13_OH=calc_rpfr(11,0,3,0,0)
a2H_OH=1/4*(calc_rpfr(1,0,1,0,0)+calc_rpfr(2,0,1,0,0)+calc_rpfr(3,0,1,0,0)+calc_rpfr(4,0,1,0,0))
aQH3D_OH=1/4*(calc_rpfr(12,0,4,0,0)+calc_rpfr(13,0,4,0,0)+calc_rpfr(14,0,4,0,0)+calc_rpfr(15,0,4,0,0))
aCH2D2_OH=1/6*(calc_rpfr(5,0,2,0,0)+calc_rpfr(6,0,2,0,0)+calc_rpfr(7,0,2,0,0)
               +calc_rpfr(8,0,2,0,0)+calc_rpfr(9,0,2,0,0)+calc_rpfr(10,0,2,0,0))

print("sMMO ab initio", "\n", a13,"\n",a2H, "\n", aQH3D, "\n", aCH2D2, "\n", aCD4, "\n")
print("CH4-OH ab initio", "\n", a13_OH,"\n",a2H_OH, "\n", aQH3D_OH, "\n", aCH2D2_OH, "\n")