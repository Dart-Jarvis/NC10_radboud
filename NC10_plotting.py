import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *

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
AeOM_m=AeOM_P[AeOM_P['f']>=0.4]

# Create a mask to determine which series of data to lot
mask={
    "save_fig":1,
    "T0": 1,
    "NC10":1,
    "NC10+ANME":1,
    "ANME2d":1,
    "Oct":1,
    "BES":1,
    "previous data":1,
}

fig1,ax1=plt.subplots(figsize=(12,12))
fig2,ax2=plt.subplots(figsize=(12,12))
ax1.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax1.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)

def quick_plot(ax,x,y,xerr,yerr,msk):
    if msk["T0"]==1:
        ax.errorbar(T0[x],T0[y],xerr=T0[xerr],yerr=T0[yerr], markersize=35,label=r'Tank gas', fmt='*', 
                    markerfacecolor='purple', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["NC10"]==1:
        ax.errorbar(NC10[x],NC10[y],xerr=NC10[xerr],yerr=NC10[yerr], markersize=18,label=r'NC10', fmt='o', 
                    markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
    if msk["NC10+ANME"]==1:
        ax.errorbar(NC10_ANME[x],NC10_ANME[y],xerr=NC10_ANME[xerr],yerr=NC10_ANME[yerr], markersize=18,label=r'NC10+ANME', fmt='^', 
                    markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
    if msk["ANME2d"]==1:
        ax.errorbar(ANME2d[x],ANME2d[y],xerr=ANME2d[xerr],yerr=ANME2d[yerr], markersize=18,label=r'ANME', fmt='s', 
                    markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
    if msk["Oct"]==1:
        ax.errorbar(Oct[x],Oct[y],xerr=Oct[xerr],yerr=Oct[yerr], markersize=18,label=r'NC10+ANME (pMMO inhibited)', fmt='D', 
                    markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["BES"]==1:
        ax.errorbar(BES[x],BES[y],xerr=BES[xerr],yerr=BES[yerr], markersize=18,label=r'NC10+ANME (mcr inhibited)', fmt='v', 
                    markerfacecolor='cyan', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["previous data"]==1:
        ax.errorbar(AeOM_m[x],AeOM_m[y],xerr=AeOM_m[xerr],yerr=AeOM_m[yerr], markersize=18,label=r'AeOM (Li et al., 2024)', fmt='o', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
        ax.errorbar(AeOM_P1[x],AeOM_P1[y],xerr=AeOM_P1[xerr],yerr=AeOM_P1[yerr], markersize=18,label=r'AeOM (Krause et al., 2022)', fmt='^', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)  
        ax.errorbar(AeOM_P2[x],AeOM_P2[y],xerr=AeOM_P2[xerr],yerr=AeOM_P2[yerr], markersize=18,label=r'AeOM (Wang et al., 2016)', fmt='D', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)          
        ax.errorbar(AOM_P[x],AOM_P[y],xerr=AOM_P[xerr],yerr=AOM_P[yerr], markersize=18,label=r'Kinetic S-AOM (Liu et al., 2023)', fmt='s', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
        ax.errorbar(AOM_P_LS[x],AOM_P_LS[y],xerr=AOM_P_LS[xerr],yerr=AOM_P_LS[yerr], markersize=18,label=r'Equilibrium S-AOM (Liu et al., 2023)', fmt='v', 
                markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
        ax.errorbar(AOM_ono[x],AOM_ono[y],xerr=AOM_ono[xerr],yerr=AOM_ono[yerr], markersize=18,label=r'S-AOM (Ono et al., 2021)', fmt='x', 
                markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)

def set_axis(ax,xlabel,ylabel):
    ax.set_ylabel(ylabel, fontdict = font_labels)
    ax.set_xlabel(xlabel, fontdict = font_labels)
    ax.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
    ax.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)
      

quick_plot(ax1,"D13CH3D","D12CH2D2","cdse","ddse",mask)
quick_plot(ax2,"d13C","dD","cse","dse",mask)

ax1.legend(fontsize=15)
set_axis(ax1,'$\Delta^{13}$CH$_3$D (\u2030)','$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
ax1.set_ylim([-23,50])
ax1.set_xlim([-4,16])
ax1.yaxis.set_minor_locator(MultipleLocator(2))
ax1.xaxis.set_minor_locator(MultipleLocator(1))

set_axis(ax2,'$\delta^{13}$C (\u2030)','$\delta$D (\u2030)')
ax2.yaxis.set_minor_locator(MultipleLocator(10))
ax2.xaxis.set_minor_locator(MultipleLocator(1))

# The compilation of isotope fractionation factors
alphas={
    "NC10": alpha_raw[alpha_raw['Label']=="NC10"],
    "ANME2d": alpha_raw[alpha_raw['Label']=="ANME2d"],
    "NC10+ANME": alpha_raw[alpha_raw['Label']=="NC10+ANME"],
    "Oct": alpha_raw[alpha_raw['Label']=="pMMO_inhibit"],
    "BES": alpha_raw[alpha_raw['Label']=="mcr_inhibit"],
    "S-AOM": alpha_raw[alpha_raw['Label']=="S-AOM"],
    "pMMO_P": alpha_raw[alpha_raw['Label']=="pMMO_P"],
    "OH": alpha_raw[alpha_raw['Label']=="OH"],
    "Cl": alpha_raw[alpha_raw['Label']=="Cl"],
    "red_mass": alpha_raw[alpha_raw['Label']=="red_mass"],
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
    "NC10+ANME":1,
    "ANME2d":1,
    "Oct":1,
    "BES":1,
    "S-AOM":1,
    "pMMO_P":1,
    "OH":1,
    "Cl":1,
    "red_mass":1
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
        ax.errorbar(Oct[x],Oct[y],xerr=Oct[xerr],yerr=Oct[yerr], markersize=18,label=r'NC10+ANME (pMMO inhibited)', fmt='D', 
                    markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
        ax.plot(-np.log(Oct_f[:,4]),Oct_f[:,idx+5], "--", c="red", linewidth=3.0, label="NC10+ANME (pMMO inhibited)")
    if msk["BES"]==1:
        ax.errorbar(BES[x],BES[y],xerr=BES[xerr],yerr=BES[yerr], markersize=18,label=r'NC10+ANME (mcr inhibited)', fmt='v', 
                    markerfacecolor='cyan', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
        ax.plot(-np.log(BES_f[:,4]),BES_f[:,idx+5], "--", c="blue", linewidth=3.0, label="NC10+ANME (mcr inhibited)")

plot_isotopef(axcf,"-lnf","ln(c/c0)","lnfse","lncse",maskf)
set_axis(axcf,r"-ln$f$", r"ln$\frac{\delta^{13}C+1000}{\delta^{13}C_{ini}+1000}$")
axcf.set_ylim([-0.001,0.02])
axcf.set_xlim([-0.05,1])
axcf.yaxis.set_minor_locator(MultipleLocator(0.001))
axcf.xaxis.set_minor_locator(MultipleLocator(0.1))

plot_isotopef(axdf,"-lnf","ln(d/d0)","lnfse","lndse",maskf)
set_axis(axdf,r"-ln$f$", r"ln$\frac{\delta D+1000}{\delta D_{ini}+1000}$")
axdf.set_xlim([-0.05,1])
axdf.xaxis.set_minor_locator(MultipleLocator(0.1))
axdf.set_ylim([-0.01,0.2])
axdf.yaxis.set_minor_locator(MultipleLocator(0.01))

plot_isotopef(axcdf,"-lnf","ln(cd/cd0)","lnfse","lncdse",maskf)
set_axis(axcdf,r"-ln$f$", r"ln$\frac{\delta^{13}CH_3D+1000}{\delta^{13}CH_3D_{ini}+1000}$")
axcdf.set_xlim([-0.05,1])
axcdf.xaxis.set_minor_locator(MultipleLocator(0.1))
axcdf.set_ylim([-0.01,0.2])
axcdf.yaxis.set_minor_locator(MultipleLocator(0.01))

plot_isotopef(axddf,"-lnf","ln(dd/dd0)","lnfse","lnddse",maskf)
set_axis(axddf,r"-ln$f$", r"ln$\frac{\delta^{12}CH_2D_2+1000}{\delta^{13}CH_2D_{2 ini}+1000}$")
axddf.set_xlim([-0.05,1])
axddf.xaxis.set_minor_locator(MultipleLocator(0.1))
axddf.set_ylim([-0.02,0.35])
axddf.yaxis.set_minor_locator(MultipleLocator(0.02))
axddf.legend(fontsize=15, bbox_to_anchor=(1.04,0.8))

# Plot the fractionation factors
series=[r"$^{\rm 13C}\alpha$", r"$^{\rm D}\alpha$", r"$^{\rm 13CD}\alpha$", r"$^{\rm D2}\alpha$", r"$^{\rm 13CD}\gamma$", r"$^{\rm D2}\gamma$"]
color_dict={
    "NC10": ["orange","o",2],
    "NC10+ANME":["blue","^",2],
    "ANME2d": ["yellow","s",2],
    "Oct":["red","D",1],
    "BES":["cyan","v",1],
    "S-AOM":["white","s",-1],
    "pMMO_P":["white","o",-1],
    "OH":["gray","D",-2],
    "Cl":["gray","^",-2],
    "red_mass":["gray","v",-1]
}
fig3,ax3=plt.subplots(figsize=(8,8))

for key in alphas.keys():
    temp=alphas[key].to_numpy()
    if maskf[key]!=0:
        for i in range(len(temp)):
            if i==0:
                ax3.scatter(series,temp[i,1:13:2], c=color_dict[key][0], marker=color_dict[key][1], s=60, edgecolors='black',label=key, zorder=color_dict[key][2])
            else:
                ax3.scatter(series,temp[i,1:13:2], c=color_dict[key][0], marker=color_dict[key][1], edgecolors='black', s=60, zorder=color_dict[key][2])


ax3.legend(fontsize=18, bbox_to_anchor=(1.04,0.9))
ax3.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=24)
ax3.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=24)
ax3.yaxis.set_minor_locator(MultipleLocator(0.02))

# With f
# fig5,ax5=plt.subplots(figsize=(12,6))
# fig6,ax6=plt.subplots(figsize=(12,6))
# fig7,ax7=plt.subplots(figsize=(12,6))
# fig8,ax8=plt.subplots(figsize=(12,6))

# def quick_plot_f(ax,y,ye,msk):
#     if msk["T0"]==1:
#         ax.errorbar(T0['f'],T0[y], xerr=T0['fse'], yerr=T0[ye], markersize=30,label=r'T0', fmt='*', 
#                     markerfacecolor='purple', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
#     if msk["NC10"]==1:
#         ax.errorbar(NC10['f'],NC10[y], xerr=NC10['fse'], yerr=NC10[ye], markersize=24,label=r'NC10', fmt='o', 
#                     markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
#     if msk["NC10+ANME"]==1:
#         ax.errorbar(NC10_ANME['f'],NC10_ANME[y], xerr=NC10_ANME['fse'], yerr=NC10_ANME[ye], markersize=24,label=r'NC10+ANME', fmt='^', 
#                     markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
#     if msk["ANME2d"]==1:
#         ax.errorbar(ANME2d['f'],ANME2d[y], xerr=ANME2d['fse'], yerr=ANME2d[ye], markersize=24,label=r'ANME-2d', fmt='s', 
#                     markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
#     if msk["Oct"]==1:
#         ax.errorbar(Oct["f"],Oct[y],xerr=Oct["fse"],yerr=Oct[ye], markersize=18,label=r'NC10+ANME (pMMO inhibited)', fmt='D', 
#                     markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
#     if msk["BES"]==1:
#         ax.errorbar(BES["f"],BES[y],xerr=BES["fse"],yerr=BES[ye], markersize=18,label=r'NC10+ANME (mcr inhibited)', fmt='v', 
#                     markerfacecolor='cyan', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
#     if msk["previous data"]==1:
#         ax.errorbar(AeOM_m['f'][AeOM_m['f']>=0.4],AeOM_m[y], xerr=AeOM_m['fse'], yerr=AeOM_m[ye], markersize=16,label=r'AeOM (Li et al., 2024)', fmt='o', 
#                     markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)
#         ax.errorbar(AeOM_P1['f'],AeOM_P1[y], xerr=AeOM_P1['fse'], yerr=AeOM_P1[ye], markersize=16,label=r'AeOM (Krause et al., 2022)', fmt='^', 
#                     markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)
#         ax.errorbar(AeOM_P2['f'],AeOM_P2[y], xerr=AeOM_P2['fse'], yerr=AeOM_P2[ye], markersize=16,label=r'AeOM (Wang et al., 2016)', fmt='D', 
#                     markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)    
#         ax.errorbar(AOM_P['f'],AOM_P[y], xerr=AOM_P['fse'], yerr=AOM_P[ye], markersize=16,label=r'AOM (Liu et al., 2023)', fmt='s', 
#                     markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1) 

# Plot epsilons
def plot_isotope_eps(ax,x,y,xerr,yerr,msk):
    if msk["NC10"]==1:
        ax.errorbar(NC10[x],NC10[y],xerr=NC10[xerr],yerr=NC10[yerr], markersize=18,label=r'NC10', fmt='o', 
                    markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
    if msk["ANME2d"]==1:
        ax.errorbar(ANME2d[x],ANME2d[y],xerr=ANME2d[xerr],yerr=ANME2d[yerr], markersize=18,label=r'ANME', fmt='s', 
                    markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
    if msk["NC10+ANME"]==1:
        ax.errorbar(NC10_ANME[x],NC10_ANME[y],xerr=NC10_ANME[xerr],yerr=NC10_ANME[yerr], markersize=18,label=r'NC10+ANME', fmt='^', 
                    markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
    if msk["Oct"]==1:
        ax.errorbar(Oct[x],Oct[y],xerr=Oct[xerr],yerr=Oct[yerr], markersize=18,label=r'NC10+ANME (pMMO inhibited)', fmt='D', 
                    markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["BES"]==1:
        ax.errorbar(BES[x],BES[y],xerr=BES[xerr],yerr=BES[yerr], markersize=18,label=r'NC10+ANME (mcr inhibited)', fmt='v', 
                    markerfacecolor='cyan', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)

fig_eps_c_d,ax_eps_c_d=plt.subplots(figsize=(8,8))
fig_eps_cd_dd,ax_eps_cd_dd=plt.subplots(figsize=(8,8))

# Calucate the epsilon ratios in the previous studies for plotting the curves
eps_data = pd.read_csv('epsilon.csv')
# The compilation of isotope fractionation factors
eps={
    "NC10": eps_data[eps_data['Label']=="NC10"],
    "ANME2d": eps_data[eps_data['Label']=="ANME2d"],
    "NC10+ANME": eps_data[eps_data['Label']=="NC10+ANME"],
    "Oct": eps_data[eps_data['Label']=="pMMO_inhibit"],
    "BES": eps_data[eps_data['Label']=="mcr_inhibit"],
    "S-AOM": eps_data[eps_data['Label']=="S-AOM"],
    "pMMO_P": eps_data[eps_data['Label']=="pMMO_P"],
    "OH": eps_data[eps_data['Label']=="OH"],
    "Cl": eps_data[eps_data['Label']=="Cl"],
    "red_mass": eps_data[eps_data['Label']=="red_mass"],
}


        
plot_isotope_eps(ax_eps_c_d,"ln(c/c0)","ln(d/d0)","lncse","lndse",maskf)
set_axis(ax_eps_c_d, r"ln$\frac{\delta^{13}C+1000}{\delta^{13}C_{ini}+1000}$",r"ln$\frac{\delta D+1000}{\delta D_{ini}+1000}$")
ax_eps_c_d.set_xlim([-0.002,0.02])
ax_eps_c_d.set_ylim([-0.01,0.18])
ax_eps_c_d.xaxis.set_minor_locator(MultipleLocator(0.002))
ax_eps_c_d.yaxis.set_minor_locator(MultipleLocator(0.01))

plot_isotope_eps(ax_eps_cd_dd,"ln(cd/cd0)","ln(dd/dd0)","lncdse","lnddse",maskf)
set_axis(ax_eps_cd_dd,r"ln$\frac{\delta^{13}CH_3D+1000}{\delta^{13}CH_3D_{ini}+1000}$",r"ln$\frac{\delta^{12}CH_2D_2+1000}{\delta^{13}CH_2D_{2 ini}+1000}$")
ax_eps_cd_dd.set_xlim([-0.01,0.2])
ax_eps_cd_dd.xaxis.set_minor_locator(MultipleLocator(0.01))
ax_eps_cd_dd.set_ylim([-0.02,0.35])
ax_eps_cd_dd.yaxis.set_minor_locator(MultipleLocator(0.02))

# Calculate the lines
mask_eps={
    "NC10": [0,"black","--",2],
    "NC10+ANME":[0,"black","-",2],
    "ANME2d": [0,"black",":",2],
    "Oct":[0,"red","--",1],
    "BES":[0,"blue","--",1],
    "S-AOM":[1,"black","-",-1],
    "pMMO_P":[1,"red","--",-1],
    "OH":[1,"blue",":",-2],
    "Cl":[1,"purple","-.",-2],
    "red_mass":[0,"gray","-",-1]
}
x1=np.arange(0.0,0.021,0.001) # x for C-D plot
x2=np.arange(0.0,0.21,0.01) # x for CD-DD plot
line_data={}
for key in eps.keys():
    temp=eps[key].to_numpy()
    r_eps=np.zeros((len(temp),2))
    l1_eps=np.zeros((len(temp),len(x1)))
    l2_eps=np.zeros((len(temp),len(x2)))
    if mask_eps[key][0]==1:
        for i in range(len(temp)):
            r_eps[i,0]=temp[i,3]/temp[i,1]
            r_eps[i,1]=temp[i,7]/temp[i,5]
            l1_eps[i,:]=r_eps[i,0]*x1
            l2_eps[i,:]=r_eps[i,1]*x2
            if i==0:
                ax_eps_c_d.plot(x1,l1_eps[i,:],color=mask_eps[key][1], linestyle=mask_eps[key][2],zorder=mask_eps[key][3],linewidth=2.0, label=key)
                ax_eps_cd_dd.plot(x2,l2_eps[i,:],color=mask_eps[key][1], linestyle=mask_eps[key][2],zorder=mask_eps[key][3],linewidth=2.0, label=key)
            else:
                ax_eps_c_d.plot(x1,l1_eps[i,:],color=mask_eps[key][1], linestyle=mask_eps[key][2],zorder=mask_eps[key][3],linewidth=2.0)
                ax_eps_cd_dd.plot(x2,l2_eps[i,:],color=mask_eps[key][1], linestyle=mask_eps[key][2],zorder=mask_eps[key][3],linewidth=2.0)


ax_eps_cd_dd.legend(fontsize=18, bbox_to_anchor=(1.04,0.9))

# Plot the methane oxidation rates with different experimental conditions
raw_rate=pd.read_csv("methanotrophy_rate.csv")
rate_data = {
    'NC10+ANME': raw_rate["NC10+ANME"],
    'NC10+ANME+Oct': raw_rate["NC10+ANME+Oct"],
    'NC10+ANME+BES': raw_rate["NC10+ANME+BES"]
}

# Compute means and standard deviations
groups = list(rate_data.keys())
means = [np.mean(rate_data[g]) for g in groups]
stds = [np.std(rate_data[g]) for g in groups]

# Plot bar chart with error bars
plt.figure(figsize=(8, 5))
plt.bar(groups, means, yerr=stds, capsize=5, color='lightcoral', edgecolor='black')
plt.ylabel(r'Methanotrophy rate ($\mu$mol/day/g dw)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Model curves
model_data=pd.read_csv("model_output.csv")
fig_bulk_model,ax_bulk_model=plt.subplots(figsize=(8,8))
fig_clump_model,ax_clump_model=plt.subplots(figsize=(8,8))

ax_clump_model.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax_clump_model.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)

ax_clump_model.errorbar(ANME2d["D13CH3D"],ANME2d["D12CH2D2"],xerr=ANME2d["cdse"],yerr=ANME2d["ddse"], markersize=14,label=r'N-AOM', fmt='s',
        markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
ax_clump_model.errorbar(AOM_P["D13CH3D"],AOM_P["D12CH2D2"],xerr=AOM_P["cdse"],yerr=AOM_P["ddse"], markersize=14,label=r'S-AOM', fmt='s', 
        markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
ax_bulk_model.errorbar(ANME2d["d13C"],ANME2d["dD"],xerr=ANME2d["cse"],yerr=ANME2d["dse"], markersize=14,label=r'N-AOM', fmt='s',
        markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
ax_bulk_model.errorbar(AOM_P["d13C"],AOM_P["dD"],xerr=AOM_P["cse"],yerr=AOM_P["dse"], markersize=14,label=r'S-AOM', fmt='s', 
        markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=3)
for i in range(5):
    ax_clump_model.plot(model_data.iloc[1::,4*i+2].astype(float),model_data.iloc[1::,4*i+3].astype(float), linewidth=2.0, color="black", alpha=0.12*i+0.4)
    ax_bulk_model.plot(model_data.iloc[1::,4*i].astype(float),model_data.iloc[1::,4*i+1].astype(float), linewidth=2.0, color="black", alpha=0.12*i+0.4)

ax_clump_model.set_ylim([-25,100])
ax_clump_model.set_xlim([-5,50])
set_axis(ax_clump_model,'$\Delta^{13}$CH$_3$D (\u2030)','$\Delta^{12}$CH$_2$D$_2$ (\u2030)')
ax_clump_model.legend(fontsize=16)
set_axis(ax_bulk_model,'$\delta^{13}$C (\u2030)','$\delta$D (\u2030)')



if mask["save_fig"]==1:
    fig1.savefig('clumped_sum.pdf', bbox_inches='tight')
    fig2.savefig('bulk_sum.pdf', bbox_inches='tight')
    fig_eps_c_d.savefig("eps_c_d.pdf", bbox_inches='tight')
    fig_eps_cd_dd.savefig("eps_cd_dd.pdf", bbox_inches='tight')
    figcf.savefig('frac_c.pdf', bbox_inches='tight')
    figdf.savefig('frac_d.pdf', bbox_inches='tight')
    figcdf.savefig('frac_cd.pdf', bbox_inches='tight')
    figddf.savefig('frac_dd.pdf', bbox_inches='tight')
    fig_bulk_model.savefig('bulk_model.pdf', bbox_inches='tight')
    fig_clump_model.savefig('clump_model.pdf', bbox_inches='tight')