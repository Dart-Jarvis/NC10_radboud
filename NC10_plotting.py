import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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

plt.rcParams['savefig.facecolor']='white'
data = pd.read_csv('isotope_data.csv')
equib = pd.read_csv('equib.csv')
alpha_raw = pd.read_csv('fractionation factors.csv')
isotopef= pd.read_csv('isotope_f.csv')

T0=data[data['label']=='0'] # T0, tank gas and controls
NC10=data[data['label']=='1'] # NC10
NC10_ANME=data[data['label']=='2'] # NC10+ANME
ANME2d=data[data['label']=='3'] # ANME-2d
Oct=data[data['label']=='4'] # pMMO inhibited experiments
BES=data[data['label']=='5'] # MCR inhibited experiments
AeOM_P=data[data['label']=='P'] # Previous data from Li et al., 2024
AeOM_P1=data[data['label']=='P1'] # Previous data from Krause et al., 2022
AeOM_P2=data[data['label']=='P2'] # wang et al., 2016
AOM_P=data[data['label']=='AOM_P'] # Previous data from Liu et al., 2023, Kinetic Sulfate AOM
AOM_P_LS=data[data['label']=='AOM_P_LS'] # AOM data from Liu et al., 2023, equilibrium S-AOM
AOM_ono=data[data['label']=='AOM_ono'] # AOM data from Ono et al., 2021 (high sulfate series)
AeOM_m=AeOM_P[AeOM_P['f']>=0.1]
AOM_wegener=data[data['label']=='AOM_Wegener']
AOM_wegener_LS=data[data['label']=='AOM_Wegener_LS']

# Create a mask to determine which series of data to lot
mask={
    "save_fig":1,
    "T0": 1,
    "NC10":1,
    "NC10+ANME":1,
    "ANME2d":1,
    "Oct":1,
    "BES":1,
    "previous data":0,
}

fig1,ax1=plt.subplots(figsize=(12,12))
fig2,ax2=plt.subplots(figsize=(12,12))
ax1.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax1.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)

def quick_plot(ax,x,y,xerr,yerr,msk):
    if msk["T0"]==1:
        ax.errorbar(T0[x],T0[y],xerr=T0[xerr],yerr=T0[yerr], markersize=32,label=r'Tank gas', fmt='*', 
                    markerfacecolor='None', markeredgecolor='purple',markeredgewidth=3.5, ecolor='purple', elinewidth=2.5, zorder=4)
    if msk["NC10"]==1:
        ax.errorbar(NC10[x],NC10[y],xerr=NC10[xerr],yerr=NC10[yerr], markersize=18,label=r'NC10', fmt='o', 
                    markerfacecolor='None', markeredgecolor='red',markeredgewidth=3.5, ecolor='red', elinewidth=2.5, zorder=3)
    if msk["NC10+ANME"]==1:
        ax.errorbar(NC10_ANME[x],NC10_ANME[y],xerr=NC10_ANME[xerr],yerr=NC10_ANME[yerr], markersize=18,label=r'NC10+ANME', fmt='^', 
                    markerfacecolor='None', markeredgecolor='blue',markeredgewidth=3.5, ecolor='blue', elinewidth=2.5, zorder=3)
    if msk["ANME2d"]==1:
        ax.errorbar(ANME2d[x],ANME2d[y],xerr=ANME2d[xerr],yerr=ANME2d[yerr], markersize=18,label=r'ANME', fmt='s', 
                    markerfacecolor='None', markeredgecolor='black',markeredgewidth=3.5, ecolor='black', elinewidth=2.5, zorder=3)
    if msk["Oct"]==1:
        ax.errorbar(Oct[x],Oct[y],xerr=Oct[xerr],yerr=Oct[yerr], markersize=18,label=r'NC10+ANME+Oct (pMMO inhibited)', fmt='D', 
                    markerfacecolor='None', markeredgecolor='black',markeredgewidth=3.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["BES"]==1:
        ax.errorbar(BES[x],BES[y],xerr=BES[xerr],yerr=BES[yerr], markersize=18,label=r'NC10+ANME+BES (mcr inhibited)', fmt='v', 
                    markerfacecolor='None', markeredgecolor='red',markeredgewidth=3.5, ecolor='red', elinewidth=2.5, zorder=2)
    if msk["previous data"]==1:
        ax.errorbar(AeOM_m[x],AeOM_m[y],xerr=AeOM_m[xerr],yerr=AeOM_m[yerr], markersize=14,label=r'AeOM', fmt='o', 
                    markerfacecolor='None', markeredgecolor='gray',markeredgewidth=3.5, ecolor='gray', elinewidth=2.5, zorder=-1)
        ax.errorbar(AeOM_P1[x],AeOM_P1[y],xerr=AeOM_P1[xerr],yerr=AeOM_P1[yerr], markersize=14, fmt='o', 
                    markerfacecolor='None', markeredgecolor='gray',markeredgewidth=3.5, ecolor='gray', elinewidth=2.5, zorder=-1)  
        ax.errorbar(AeOM_P2[x],AeOM_P2[y],xerr=AeOM_P2[xerr],yerr=AeOM_P2[yerr], markersize=14, fmt='o', 
                    markerfacecolor='None', markeredgecolor='gray',markeredgewidth=3.5, ecolor='gray', elinewidth=2.5, zorder=-1)          
        ax.errorbar(AOM_P[x],AOM_P[y],xerr=AOM_P[xerr],yerr=AOM_P[yerr], markersize=14,label=r'S-AOM (high sulfate)', fmt='s', 
                    markerfacecolor='None', markeredgecolor='gray',markeredgewidth=3.5, ecolor='gray', elinewidth=2.5, zorder=-1)
        ax.errorbar(AOM_P_LS[x],AOM_P_LS[y],xerr=AOM_P_LS[xerr],yerr=AOM_P_LS[yerr], markersize=14, fmt='^', label=r"S-AOM (low sulfate)",
                markerfacecolor='None', markeredgecolor='gray',markeredgewidth=3.5, ecolor='gray', elinewidth=2.5, zorder=-1)
        ax.errorbar(AOM_ono[x],AOM_ono[y],xerr=AOM_ono[xerr],yerr=AOM_ono[yerr], markersize=14, fmt='s', 
                markerfacecolor='None', markeredgecolor='gray',markeredgewidth=3.5, ecolor='gray', elinewidth=2.5, zorder=-1)
        ax.errorbar(AOM_wegener[x],AOM_wegener[y],xerr=AOM_wegener[xerr],yerr=AOM_wegener[yerr], markersize=14, fmt='s', 
                markerfacecolor='None', markeredgecolor='gray',markeredgewidth=3.5, ecolor='gray', elinewidth=2.5, zorder=-1)
        ax.errorbar(AOM_wegener_LS[x],AOM_wegener_LS[y],xerr=AOM_wegener_LS[xerr],yerr=AOM_wegener_LS[yerr], markersize=14, fmt='^', 
                markerfacecolor='None', markeredgecolor='gray',markeredgewidth=3.5, ecolor='gray', elinewidth=2.5, zorder=-1)

def set_axis(ax,xlabel,ylabel):
    ax.set_ylabel(ylabel, fontdict = font_labels)
    ax.set_xlabel(xlabel, fontdict = font_labels)
    ax.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
    ax.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)
      

quick_plot(ax1,"D13CH3D","D12CH2D2","cdse","ddse",mask)
quick_plot(ax2,"d13C","dD","cse","dse",mask)

# ax1.legend(fontsize=24,loc="upper left")
set_axis(ax1,'$\Delta^{13}$CH$_3$D (\u2030)','$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
# ax1.set_ylim([-23,50])
# ax1.set_xlim([-4,16])
ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.4))

set_axis(ax2,'$\delta^{13}$C (\u2030)','$\delta$D (\u2030)')
ax2.yaxis.set_minor_locator(MultipleLocator(4))
ax2.xaxis.set_minor_locator(MultipleLocator(1))

# Plot the data in this study together with data from previous studies
comp_b,ax_comp_b=plt.subplots(figsize=(12,12))
comp_c,ax_comp_c=plt.subplots(figsize=(12,12))
mask["previous data"]=1

ax_comp_c.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax_comp_c.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)
quick_plot(ax_comp_c,"D13CH3D","D12CH2D2","cdse","ddse",mask)
quick_plot(ax_comp_b,"d13C","dD","cse","dse",mask)

# ax1.legend(fontsize=24,loc="upper left")
set_axis(ax_comp_c,'$\Delta^{13}$CH$_3$D (\u2030)','$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
ax_comp_c.yaxis.set_minor_locator(MultipleLocator(4))
ax_comp_c.xaxis.set_minor_locator(MultipleLocator(2))

set_axis(ax_comp_b,'$\delta^{13}$C (\u2030)','$\delta$D (\u2030)')
ax_comp_b.yaxis.set_minor_locator(MultipleLocator(20))
ax_comp_b.xaxis.set_minor_locator(MultipleLocator(4))
ax_comp_b.legend(fontsize=24)

# The compilation of isotope fractionation factors
alphas={
    "NC10": alpha_raw[alpha_raw['Label']=="NC10"],
    "ANME2d": alpha_raw[alpha_raw['Label']=="ANME2d"],
    "NC10+ANME": alpha_raw[alpha_raw['Label']=="NC10+ANME"],
    "Oct": alpha_raw[alpha_raw['Label']=="pMMO_inhibit"],
    "BES": alpha_raw[alpha_raw['Label']=="mcr_inhibit"],
    "S-AOM": alpha_raw[alpha_raw['Label']=="S-AOM"],
    "AeOM": alpha_raw[alpha_raw['Label']=="pMMO_P"],
    "OH": alpha_raw[alpha_raw['Label']=="OH"],
    "Cl": alpha_raw[alpha_raw['Label']=="Cl"],
    "red_mass": alpha_raw[alpha_raw['Label']=="red_mass"],
    "Ab initio (sMMO)": alpha_raw[alpha_raw['Label']=="sMMO"],
    "Ab initio (MCR)":alpha_raw[alpha_raw['Label']=="MCR"],
    "Ab initio (pMMO)":alpha_raw[alpha_raw['Label']=="pMMO"]
}


f=np.arange(0.2,1.1,0.1)
def plt_model(T0,ff):
    m=np.zeros([len(f),9])
    C0=T0["d13C"]
    D0=T0["dD"]
    cd0=T0["D13CH3D"]
    dd0=T0["D12CH2D2"] # Zero point
    aC_model=ff["a13"]
    aD_model=ff["aD"]
    aCD_model=ff["a13CD"]
    aDD_model=ff["aDD"] # Fractionation factors
    for i in range(len(f)):
        m[i,0]=(C0+1000)*f[i]**(aC_model-1)-1000 # Carbon isotope
        m[i,1]=(D0+1000)*f[i]**(aD_model-1)-1000 # Hydrogen isotope
        m[i,2]=cd0+1000*(aCD_model-aC_model-aD_model+1)*np.log(f[i]) # 13CH3D
        m[i,3]=dd0+1000*(aDD_model-2*aD_model+1)*np.log(f[i]) # 12CH2D2
        m[i,4]=f[i]
        m[i,5]=np.log(f[i])*(aC_model-1)
        m[i,6]=np.log(f[i])*(aD_model-1)
        m[i,7]=np.log(f[i])*(aCD_model-1)
        m[i,8]=np.log(f[i])*(aDD_model-1)
    return m

NC10_f=plt_model(NC10.iloc[0],alphas["NC10"])
ANME_f=plt_model(ANME2d.iloc[0],alphas["ANME2d"])
NC10_ANME_f=plt_model(NC10_ANME.iloc[0],alphas["NC10+ANME"])
Oct_f=plt_model(Oct.iloc[0],alphas["Oct"])
BES_f=plt_model(BES.iloc[0],alphas["BES"])

# Regression lines
figcf,axcf=plt.subplots(figsize=(8,8))
figdf,axdf=plt.subplots(figsize=(8,8))
figcdf,axcdf=plt.subplots(figsize=(8,8))
figddf,axddf=plt.subplots(figsize=(8,8))

maskf={
    "NC10": 1,
    "NC10+ANME":0,
    "ANME2d":0,
    "Oct":0,
    "BES":1,
    "S-AOM":0,
    "AeOM":1,
    "OH":0,
    "Cl":0,
    "red_mass":0,
    "Ab initio (sMMO)": 1,
    "Ab initio (MCR)": 0,
    "Ab initio (pMMO)":1
}

def plot_isotopef(ax,x,y,xerr,yerr,msk):
    name_yerr=["lncse","lndse","lncdse","lnddse"]
    idx=name_yerr.index(yerr)
    if msk["NC10"]==1:
        ax.errorbar(NC10[x],NC10[y],xerr=NC10[xerr],yerr=NC10[yerr], markersize=18,label=r'NC10', fmt='o', 
                    markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
        ax.plot(-np.log(NC10_f[:,4]),NC10_f[:,idx+5], "--", c="black", linewidth=3.0, label="NC10")

    if msk["ANME2d"]==1:
        ax.errorbar(ANME2d[x],ANME2d[y],xerr=ANME2d[xerr],yerr=ANME2d[yerr], markersize=18,label=r'ANME', fmt='s', 
                    markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
        ax.plot(-np.log(ANME_f[:,4]),ANME_f[:,idx+5], "-", c="black", linewidth=3.0, label="ANME")
    if msk["NC10+ANME"]==1:
        ax.errorbar(NC10_ANME[x],NC10_ANME[y],xerr=NC10_ANME[xerr],yerr=NC10_ANME[yerr], markersize=18,label=r'NC10+ANME', fmt='^', 
                    markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
        ax.plot(-np.log(NC10_ANME_f[:,4]),NC10_ANME_f[:,idx+5], ":", c="black", linewidth=3.0, label="NC10+ANME")

    if msk["Oct"]==1:
        ax.errorbar(Oct[x],Oct[y],xerr=Oct[xerr],yerr=Oct[yerr], markersize=18,label=r'NC10+ANME+Oct (pMMO inhibited)', fmt='D', 
                    markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
        ax.plot(-np.log(Oct_f[:,4]),Oct_f[:,idx+5], "--", c="red", linewidth=3.0, label="NC10+ANME (pMMO inhibited)")
    if msk["BES"]==1:
        ax.errorbar(BES[x],BES[y],xerr=BES[xerr],yerr=BES[yerr], markersize=18,label=r'NC10+ANME+BES (mcr inhibited)', fmt='v', 
                    markerfacecolor='cyan', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
        ax.plot(-np.log(BES_f[:,4]),BES_f[:,idx+5], "--", c="blue", linewidth=3.0, label="NC10+ANME (mcr inhibited)")

plot_isotopef(axcf,"lnf","ln(c/c0)","lnfse","lncse",maskf)
set_axis(axcf,r"-ln$f$", r"ln$\frac{\delta^{13}C+1000}{\delta^{13}C_{ini}+1000}$")
axcf.set_ylim([-0.001,0.02])
axcf.set_xlim([-0.05,1])
axcf.yaxis.set_minor_locator(MultipleLocator(0.001))
axcf.xaxis.set_minor_locator(MultipleLocator(0.1))

plot_isotopef(axdf,"lnf","ln(d/d0)","lnfse","lndse",maskf)
set_axis(axdf,r"-ln$f$", r"ln$\frac{\delta D+1000}{\delta D_{ini}+1000}$")
axdf.set_xlim([-0.05,1])
axdf.xaxis.set_minor_locator(MultipleLocator(0.1))
axdf.set_ylim([-0.01,0.2])
axdf.yaxis.set_minor_locator(MultipleLocator(0.01))

plot_isotopef(axcdf,"lnf","ln(cd/cd0)","lnfse","lncdse",maskf)
set_axis(axcdf,r"-ln$f$", r"ln$\frac{\delta^{13}CH_3D+1000}{\delta^{13}CH_3D_{ini}+1000}$")
axcdf.set_xlim([-0.05,1])
axcdf.xaxis.set_minor_locator(MultipleLocator(0.1))
axcdf.set_ylim([-0.01,0.2])
axcdf.yaxis.set_minor_locator(MultipleLocator(0.01))

plot_isotopef(axddf,"lnf","ln(dd/dd0)","lnfse","lnddse",maskf)
set_axis(axddf,r"-ln$f$", r"ln$\frac{\delta^{12}CH_2D_2+1000}{\delta^{13}CH_2D_{2 ini}+1000}$")
axddf.set_xlim([-0.05,1])
axddf.xaxis.set_minor_locator(MultipleLocator(0.1))
axddf.set_ylim([-0.02,0.35])
axddf.yaxis.set_minor_locator(MultipleLocator(0.02))
axddf.legend(fontsize=15, bbox_to_anchor=(1.04,0.8))

# Plot the fractionation factors
series=[r"$^{\rm 13C}\alpha$", r"$^{\rm D}\alpha$", r"$^{\rm 13CD}\alpha$", r"$^{\rm D2}\alpha$", r"$^{\rm 13CD}\gamma$", r"$^{\rm D2}\gamma$"]
# series=[r"$^{\rm 13C}\alpha$", r"$^{\rm D}\alpha$", r"$^{\rm 13CD}\alpha$", r"$^{\rm DD}\alpha$"]
color_dict={
    "NC10": ["red","o",2],
    "NC10+ANME":["blue","^",2],
    "ANME2d": ["black","s",2],
    "Oct":["black","v",1],
    "BES":["red","v",1],
    "S-AOM":["gray","s",-1],
    "AeOM":["gray","o",-1],
    "OH":["gray","D",-2],
    "Cl":["gray","^",-2],
    "red_mass":["gray","v",-1],
    "Ab initio (sMMO)":["red","^",2],
    "Ab initio (MCR)":["black","o",2],
    "Ab initio (pMMO)":["gray","^",2]
}

fig3,ax3=plt.subplots(figsize=(8,8))
for key in alphas.keys():
    temp=alphas[key].to_numpy()
    if maskf[key]!=0:
        for i in range(len(temp)):
            if i==0:
                ax3.scatter(series,temp[i,1:13:2], c="None", marker=color_dict[key][1], s=120, edgecolors=color_dict[key][0],linewidths=2.5,label=key, zorder=color_dict[key][2])
            else:
                ax3.scatter(series,temp[i,1:13:2], c="None", marker=color_dict[key][1], edgecolors=color_dict[key][0], linewidths=2.5, s=120, zorder=color_dict[key][2])


ax3.legend(fontsize=18, bbox_to_anchor=(1.04,0.9))
ax3.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=24)
ax3.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=24)
ax3.yaxis.set_minor_locator(MultipleLocator(0.02))

maskf={
    "NC10": 0,
    "NC10+ANME":1,
    "ANME2d":1,
    "Oct":1,
    "BES":0,
    "S-AOM":1,
    "AeOM":0,
    "OH":0,
    "Cl":0,
    "red_mass":0,
    "Ab initio (sMMO)": 0,
    "Ab initio (MCR)": 1,
    "Ab initio (pMMO)":0
}
fig4,ax4=plt.subplots(figsize=(8,8))
for key in alphas.keys():
    temp=alphas[key].to_numpy()
    if maskf[key]!=0:
        for i in range(len(temp)):
            if i==0:
                ax4.scatter(series,temp[i,1:13:2], c="None", marker=color_dict[key][1], s=120, edgecolors=color_dict[key][0],linewidths=2.5,label=key, zorder=color_dict[key][2])
            else:
                ax4.scatter(series,temp[i,1:13:2], c="None", marker=color_dict[key][1], edgecolors=color_dict[key][0], linewidths=2.5, s=120, zorder=color_dict[key][2])
ax4.legend(fontsize=18, bbox_to_anchor=(1.04,0.9))
ax4.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=24)
ax4.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=24)
ax4.yaxis.set_minor_locator(MultipleLocator(0.02))

# With f
fig5,ax5=plt.subplots(figsize=(12,6))
fig6,ax6=plt.subplots(figsize=(12,6))
fig7,ax7=plt.subplots(figsize=(12,6))
fig8,ax8=plt.subplots(figsize=(12,6))

def quick_plot_f(ax,y,ye,msk):
    if msk["T0"]==1:
        ax.errorbar(T0['f'],T0[y], xerr=T0['fse'], yerr=T0[ye], markersize=18,label=r'T0', fmt='*', 
                    markerfacecolor='purple', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["NC10"]==1:
        ax.errorbar(NC10['f'],NC10[y], xerr=NC10['fse'], yerr=NC10[ye], markersize=18,label=r'NC10', fmt='o', 
                    markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
    if msk["NC10+ANME"]==1:
        ax.errorbar(NC10_ANME['f'],NC10_ANME[y], xerr=NC10_ANME['fse'], yerr=NC10_ANME[ye], markersize=18,label=r'NC10+ANME', fmt='^', 
                    markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
    if msk["ANME2d"]==1:
        ax.errorbar(ANME2d['f'],ANME2d[y], xerr=ANME2d['fse'], yerr=ANME2d[ye], markersize=18,label=r'ANME', fmt='s', 
                    markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
    if msk["Oct"]==1:
        ax.errorbar(Oct["f"],Oct[y],xerr=Oct["fse"],yerr=Oct[ye], markersize=18,label=r'NC10+ANME (pMMO inhibited)', fmt='D', 
                    markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["BES"]==1:
        ax.errorbar(BES["f"],BES[y],xerr=BES["fse"],yerr=BES[ye], markersize=18,label=r'NC10+ANME (mcr inhibited)', fmt='v', 
                    markerfacecolor='cyan', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["previous data"]==1:
        ax.errorbar(AeOM_m["f"],AeOM_m[y],xerr=AeOM_m["fse"],yerr=AeOM_m[ye], markersize=14,label=r'AeOM (previous study)', fmt='o', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
        ax.errorbar(AeOM_P1["f"],AeOM_P1[y],xerr=AeOM_P1["fse"],yerr=AeOM_P1[ye], markersize=14, fmt='o', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)  
        ax.errorbar(AeOM_P2["f"],AeOM_P2[y],xerr=AeOM_P2["fse"],yerr=AeOM_P2[ye], markersize=14, fmt='o', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)          
        ax.errorbar(AOM_P["f"],AOM_P[y],xerr=AOM_P["fse"],yerr=AOM_P[ye], markersize=14,label=r'Sulfate-dependent AOM (previous study)', fmt='s', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
        ax.errorbar(AOM_P_LS["f"],AOM_P_LS[y],xerr=AOM_P_LS["fse"],yerr=AOM_P_LS[ye], markersize=14, fmt='s', 
                markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
        ax.errorbar(AOM_ono["f"],AOM_ono[y],xerr=AOM_ono["fse"],yerr=AOM_ono[ye], markersize=14, fmt='s', 
                markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)

def plot_mixing(ax,var):
    for i in range(len(mixing_results["r"])):
        ax.plot(mixing_results["f"][i],mixing_results[var][i], color="black", linewidth=2.0, alpha=mixing_results["alpha"][i], label=r"$R_{\rm NC10}$ ="+str(mixing_results["r"][i]))
    ax.invert_xaxis()


mixing_raw=pd.read_csv("mixing.csv")
mixing_results={
    "alpha": [1.0,0.7,0.5,0.4],
    "r": [1, 0.66, 0.33, 0.0],
    "f": [mixing_raw["f_100"],mixing_raw["f_66"],mixing_raw["f_33"],mixing_raw["f_0"]],
    "d13C": [mixing_raw["d13C_100"],mixing_raw["d13C_66"],mixing_raw["d13C_33"],mixing_raw["d13C_0"]],
    "dD": [mixing_raw["dD_100"],mixing_raw["dD_66"],mixing_raw["dD_33"],mixing_raw["dD_0"]],
    "D13CH3D": [mixing_raw["D13CH3D_100"],mixing_raw["D13CH3D_66"],mixing_raw["D13CH3D_33"],mixing_raw["D13CH3D_0"]],
    "D12CH2D2": [mixing_raw["D12CH2D2_100"],mixing_raw["D12CH2D2_66"],mixing_raw["D12CH2D2_33"],mixing_raw["D12CH2D2_0"]]
}

mask_f2={
    "save_fig":1,
    "T0": 0,
    "NC10":0,
    "NC10+ANME":1,
    "ANME2d":0,
    "Oct":0,
    "BES":0,
    "previous data":0,
}

quick_plot_f(ax5,"d13C","cse",mask_f2)
plot_mixing(ax5,"d13C")
set_axis(ax5,r"$f$", r"$\delta^{13}$C" + " (\u2030)")
ax5.set_xlim([1.04,0.32])
ax5.xaxis.set_minor_locator(MultipleLocator(0.1))
ax5.yaxis.set_minor_locator(MultipleLocator(2))
ax5.legend(fontsize=20)
quick_plot_f(ax6,"dD","dse",mask_f2)
plot_mixing(ax6,"dD")
set_axis(ax6,r"$f$", r"$\delta$D" + " (\u2030)")
ax6.set_xlim([1.04,0.32])
ax6.xaxis.set_minor_locator(MultipleLocator(0.1))
ax6.yaxis.set_minor_locator(MultipleLocator(10))
quick_plot_f(ax7,"D13CH3D","cdse",mask_f2)
plot_mixing(ax7,"D13CH3D")
ax7.set_ylim([-2.3,4.2])
ax7.set_xlim([1.04,0.32])
ax7.xaxis.set_minor_locator(MultipleLocator(0.1))
ax7.yaxis.set_minor_locator(MultipleLocator(0.4))
set_axis(ax7,r"$f$", r"$\Delta^{13}$CH$_3$D" + " (\u2030)")
quick_plot_f(ax8,"D12CH2D2","ddse",mask_f2)
plot_mixing(ax8,"D12CH2D2")
set_axis(ax8,r"$f$", r"$\Delta^{12}$CH$_2$D$_2$" + " (\u2030)")
ax8.set_xlim([1.04,0.32])
ax8.xaxis.set_minor_locator(MultipleLocator(0.1))
ax8.yaxis.set_minor_locator(MultipleLocator(2))

# Plot the methane oxidation rates with different experimental conditions
raw_rate = pd.read_csv("methanotrophy_rate.csv")
rate_data = {
    "NC10": raw_rate["NC10"],
    "NC10+ANME": raw_rate["NC10+ANME"],
    "ANME": raw_rate["ANME"],
    "NC10+ANME+Oct": raw_rate["NC10+ANME+Oct"],
    "NC10+ANME+BES": raw_rate["NC10+ANME+BES"]
}

# Use the same hollow-marker edge colors and marker types as Figure 1
rate_style = {
    "NC10": {"color": "red", "marker": "o"},
    "NC10+ANME": {"color": "blue", "marker": "^"},
    "ANME": {"color": "black", "marker": "s"},
    "NC10+ANME+Oct": {"color": "black", "marker": "D"},
    "NC10+ANME+BES": {"color": "red", "marker": "v"}
}


def plot_rate_groups(groups_to_plot, outfile):
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, group in enumerate(groups_to_plot):
        # Convert to numeric and ignore empty/NaN cells in each group
        vals = pd.to_numeric(rate_data[group], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue

        color = rate_style[group]["color"]
        marker = rate_style[group]["marker"]

        # Plot all replicates at the same x-position for each experimental group
        xvals = np.full(len(vals), i, dtype=float)

        # Plot all individual rate measurements
        ax.scatter(
            xvals, vals,
            s=85,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=2.0,
            zorder=3
        )

        # Plot the group average as a gray star with a standard-deviation error bar
        mean_val = np.mean(vals)
        std_val = np.std(vals) if len(vals) > 1 else 0.0
        ax.errorbar(
            i, mean_val,
            yerr=std_val,
            fmt="*",
            markersize=20,
            markerfacecolor="gray",
            markeredgecolor="black",
            markeredgewidth=1.5,
            ecolor="black",
            elinewidth=2.5,
            capsize=6,
            capthick=2.0,
            zorder=4
        )

    ax.set_xticks(range(len(groups_to_plot)))
    ax.set_xticklabels(groups_to_plot, fontsize=15)
    ax.set_ylabel(r'Methanotrophy rate ($\mu$mol/day/g dw)', fontsize=16)
    ax.tick_params(axis='y', labelsize=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.5, len(groups_to_plot) - 0.5)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()


# All experimental groups
plot_rate_groups(
    ["NC10", "NC10+ANME", "ANME", "NC10+ANME+Oct", "NC10+ANME+BES"],
    "rate_all.pdf"
)

# Zoomed plot excluding NC10, as in the original script
plot_rate_groups(
    ["NC10+ANME", "ANME", "NC10+ANME+Oct", "NC10+ANME+BES"],
    "rate_zoom.pdf"
)

# Methane consumption curves
consumption=pd.read_csv("methane oxidation curve.csv",header=None)
NC10_con={
    "t": consumption.iloc[7,1:],
    "exp": consumption.iloc[8,1:],
    "exp_se": consumption.iloc[9,1:],
    "t_ctrl": consumption.iloc[10,1:],
    "ctrl": consumption.iloc[11,1:],
    "ctrl_se": consumption.iloc[12,1:]
}

ANME_con={
    "t": consumption.iloc[14,1:],
    "exp": consumption.iloc[15,1:],
    "exp_se": consumption.iloc[16,1:],
    "t_ctrl": consumption.iloc[17,1:],
    "ctrl": consumption.iloc[18,1:],
    "ctrl_se": consumption.iloc[19,1:]
}

NC10_ANME_con={
    "t": consumption.iloc[0,1:],
    "exp": consumption.iloc[1,1:],
    "exp_se": consumption.iloc[2,1:],
    "t_ctrl": consumption.iloc[3,1:],
    "ctrl": consumption.iloc[4,1:],
    "ctrl_se": consumption.iloc[5,1:]
}

Oct_con={
    "t": consumption.iloc[21,1:],
    "exp": consumption.iloc[22,1:],
    "exp_se": consumption.iloc[23,1:],
    "t_ctrl": consumption.iloc[24,1:],
    "ctrl": consumption.iloc[25,1:],
    "ctrl_se": consumption.iloc[26,1:]
}

BES_con={
    "t": consumption.iloc[28,1:],
    "exp": consumption.iloc[29,1:],
    "exp_se": consumption.iloc[30,1:],
    "t_ctrl": consumption.iloc[31,1:],
    "ctrl": consumption.iloc[32,1:],
    "ctrl_se": consumption.iloc[33,1:]
}

con_nc10,ax_nc10=plt.subplots(figsize=(12,12))
ax_nc10.errorbar(NC10_con["t"],NC10_con["exp"],xerr=None,yerr=NC10_con["exp_se"],fmt="o-",color="red", markersize=18, linewidth=2.0,
            markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='red', elinewidth=2.5, label="NC10 experiment")
ax_nc10.errorbar(NC10_con["t_ctrl"],NC10_con["ctrl"],xerr=None,yerr=NC10_con["ctrl_se"],fmt="o-",color="black", markersize=18, linewidth=2.0,
            markerfacecolor='black', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, label="NC10 control", zorder=-1)
ax_nc10.errorbar(NC10_ANME_con["t"],NC10_ANME_con["exp"],xerr=None,yerr=NC10_ANME_con["exp_se"],fmt="^--",color="red", markersize=18, linewidth=2.0,
            markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='red', elinewidth=2.5, label="NC10+ANME experiment")
ax_nc10.errorbar(NC10_ANME_con["t_ctrl"],NC10_ANME_con["ctrl"],xerr=None,yerr=NC10_ANME_con["ctrl_se"],fmt="^--",color="black", markersize=18, linewidth=2.0,
            markerfacecolor='black', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, label="NC10+ANME control", zorder=-1)
ax_nc10.legend(fontsize=18)
set_axis(ax_nc10,"Elapse time (days)", r"Headspace methane ($\mu$mol)")
ax_nc10.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_nc10.xaxis.set_minor_locator(AutoMinorLocator(5))

con_anme,ax_anme=plt.subplots(figsize=(12,12))
ax_anme.errorbar(ANME_con["t"],ANME_con["exp"],xerr=None,yerr=ANME_con["exp_se"],fmt="o-",color="red", markersize=18, linewidth=2.0,
            markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='red', elinewidth=2.5, label="ANME experiment")
ax_anme.errorbar(ANME_con["t_ctrl"],ANME_con["ctrl"],xerr=None,yerr=ANME_con["ctrl_se"],fmt="o-",color="black", markersize=18, linewidth=2.0,
            markerfacecolor='black', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, label="ANME control", zorder=-1)
ax_anme.errorbar(BES_con["t"],BES_con["exp"],xerr=None,yerr=BES_con["exp_se"],fmt="v--",color="red", markersize=18, linewidth=2.0,
            markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='red', elinewidth=2.5, label="NC10+ANME+BES experiment")
ax_anme.errorbar(BES_con["t_ctrl"],BES_con["ctrl"],xerr=None,yerr=BES_con["ctrl_se"],fmt="v--",color="black", markersize=18, linewidth=2.0,
            markerfacecolor='black', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, label="NC10+ANME+BES control", zorder=-1)
ax_anme.errorbar(Oct_con["t"],Oct_con["exp"],xerr=None,yerr=Oct_con["exp_se"],fmt="s-.",color="red", markersize=18, linewidth=2.0,
            markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='red', elinewidth=2.5, label="NC10+ANME+Oct experiment")
ax_anme.errorbar(Oct_con["t_ctrl"],Oct_con["ctrl"],xerr=None,yerr=Oct_con["ctrl_se"],fmt="s-.",color="black", markersize=18, linewidth=2.0,
            markerfacecolor='black', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, label="NC10+ANME+Oct control", zorder=-1)
ax_anme.legend(fontsize=18)
set_axis(ax_anme,"Elapse time (days)", r"Headspace methane ($\mu$mol)")
ax_anme.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_anme.xaxis.set_minor_locator(AutoMinorLocator(5))

# =============================================================================
# End incorporated fig0 uncertainty plotting block
# =============================================================================

if mask["save_fig"]==1:
    fig1.savefig('clumped_sum.pdf', bbox_inches='tight')
    fig2.savefig('bulk_sum.pdf', bbox_inches='tight')
    comp_b.savefig("comp_bulk.pdf", bbox_inches='tight')
    comp_c.savefig("comp_clumped.pdf", bbox_inches='tight')
    figcf.savefig('frac_c.pdf', bbox_inches='tight')
    figdf.savefig('frac_d.pdf', bbox_inches='tight')
    figcdf.savefig('frac_cd.pdf', bbox_inches='tight')
    figddf.savefig('frac_dd.pdf', bbox_inches='tight')
    fig3.savefig('ff_pMMO.pdf', bbox_inches='tight')
    fig4.savefig('ff_mcr.pdf', bbox_inches='tight')
    fig5.savefig("d13C_f.pdf", bbox_inches='tight')
    fig6.savefig("dD_f.pdf", bbox_inches='tight')
    fig7.savefig("D13CD_f.pdf", bbox_inches='tight')
    fig8.savefig("D2_f.pdf", bbox_inches='tight')
    con_nc10.savefig("con_nc10.pdf", bbox_inches='tight')
    con_anme.savefig("con_anme.pdf", bbox_inches='tight')

