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

T0=data[data['label']=='0'] # T0, tank gas and controls
NC10=data[data['label']=='1'] # NC10
NC10_ANME=data[data['label']=='2'] # NC10+ANME
ANME2d=data[data['label']=='3'] # ANME-2d
Oct=data[data['label']=='4'] # pMMO inhibited experiments
BES=data[data['label']=='5'] # MCR inhibited experiments
AeOM_P=data[data['label']=='P'] # Previous data from Li et al., 2024
AeOM_P1=data[data['label']=='P1'] # Previous data from Krause et al., 2022
AeOM_P2=data[data['label']=='P2'] # wang et al., 2016
AOM_P=data[data['label']=='AOM_P'] # Previous data from Liu et al., 2023
AeOM_m=AeOM_P[AeOM_P['f']>=0.4]

# Create a mask to determine which series of data to lot
mask={
    "save_fig":1,
    "T0": 1,
    "NC10":1,
    "NC10+ANME":1,
    "ANME2d":1,
    "Oct":1,
    "BES":0,
    "previous data":1
}

fig1,ax1=plt.subplots(figsize=(12,12))
fig2,ax2=plt.subplots(figsize=(12,12))
ax1.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax1.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)

def quick_plot(ax,x,y,xerr,yerr,msk):
    if msk["T0"]==1:
        ax.errorbar(T0[x],T0[y],xerr=T0[xerr],yerr=T0[yerr], markersize=35,label=r'T0', fmt='*', 
                    markerfacecolor='purple', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["NC10"]==1:
        ax.errorbar(NC10[x],NC10[y],xerr=NC10[xerr],yerr=NC10[yerr], markersize=18,label=r'NC10', fmt='o', 
                    markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["NC10+ANME"]==1:
        ax.errorbar(NC10_ANME[x],NC10_ANME[y],xerr=NC10_ANME[xerr],yerr=NC10_ANME[yerr], markersize=18,label=r'NC10+ANME', fmt='^', 
                    markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["ANME2d"]==1:
        ax.errorbar(ANME2d[x],ANME2d[y],xerr=ANME2d[xerr],yerr=ANME2d[yerr], markersize=18,label=r'ANME', fmt='s', 
                    markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["Oct"]==1:
        ax.errorbar(Oct[x],Oct[y],xerr=Oct[xerr],yerr=Oct[yerr], markersize=18,label=r'NC10+ANME (pMMO inhibited)', fmt='D', 
                    markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["BES"]==1:
        ax.errorbar(BES[x],BES[y],xerr=BES[xerr],yerr=BES[yerr], markersize=18,label=r'NC10+ANME (mcr inhibited)', fmt='d', 
                    markerfacecolor='purple', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["previous data"]==1:
        ax.errorbar(AeOM_m[x],AeOM_m[y],xerr=AeOM_m[xerr],yerr=AeOM_m[yerr], markersize=18,label=r'AeOM (Li et al., 2024)', fmt='o', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
        ax.errorbar(AeOM_P1[x],AeOM_P1[y],xerr=AeOM_P1[xerr],yerr=AeOM_P1[yerr], markersize=18,label=r'AeOM (Krause et al., 2022)', fmt='^', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)  
        ax.errorbar(AeOM_P2[x],AeOM_P2[y],xerr=AeOM_P2[xerr],yerr=AeOM_P2[yerr], markersize=18,label=r'AeOM (Wang et al., 2016)', fmt='D', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)          
        ax.errorbar(AOM_P[x],AOM_P[y],xerr=AOM_P[xerr],yerr=AOM_P[yerr], markersize=18,label=r'AOM (Liu et al., 2023)', fmt='s', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=-1)
      

quick_plot(ax1,"D13CH3D","D12CH2D2","cdse","ddse",mask)
quick_plot(ax2,"d13C","dD","cse","dse",mask)

ax1.legend(fontsize=20)
ax1.set_xlabel('$\Delta^{13}$CH$_3$D (\u2030)', fontdict = font_labels)
ax1.set_ylabel('$\Delta^{12}$CH$_2$D$_2$ (\u2030)', fontdict = font_labels)
ax1.set_ylim([-23,50])
ax1.set_xlim([-4,16])
ax1.yaxis.set_minor_locator(MultipleLocator(2))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax1.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

ax2.set_xlabel('$\delta^{13}$C (\u2030)', fontdict = font_labels)
ax2.set_ylabel('$\delta$D (\u2030)', fontdict = font_labels)
ax2.yaxis.set_minor_locator(MultipleLocator(10))
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax2.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)
# Fractionation factors
aC_anme_hs=0.9951
aD_anme_hs=0.851
aCD_anme_hs=0.834
aDD_anme_hs=0.660

# 0.9778	0.8019	0.7842746	0.629615
aC_nc10=0.9778	
aD_nc10=0.8019	
aCD_nc10=0.7843	
aDD_nc10=0.6296

# OH radical ab initio (Haghnegahdar et al., 2017)
aC_OH=1/1.0063
aD_OH=1/1.32
aCD_OH=1/1.33
aDD_OH=1/1.92

# Cl radical ab initio
aC_Cl=1/1.028
aD_Cl=1/1.41
aCD_Cl=1/1.46
aDD_Cl=1/2.2

# sMMO ab initio model
aC_smo_model=0.98704
aD_smo_model=0.72548
aCD_smo_model=0.71581	
aDD_smo_model=0.48299

# sMMO ab initio model, no Wigner correction
aC_smo_model_nc=0.9875826748616653 
aD_smo_model_nc= 0.7431504785710069 
aCD_smo_model_nc= 0.7336593668868868 
aDD_smo_model_nc= 0.512046881532376

# The isotope fractionation factors in previous studies
# Li et al., 2024, 37C, 27C, 21C, Wang et al., 2016 (30 oC, 37 oC), Krause et al., 2022, Haghneghadar et al., 2017, OH and Cl
alphas=np.array([[aC_nc10,aC_smo_model,aC_smo_model_nc, 0.9671, 0.9713, 0.9757, 0.988, 0.978, 0.98485, aC_OH, aC_Cl],
        [aD_nc10,aD_smo_model,aD_smo_model_nc,0.6967, 0.7452, 0.7742, 0.8950,0.7980,0.7265, aD_OH, aD_Cl],
        [aCD_nc10,aCD_smo_model,aCD_smo_model_nc,0.6716, 0.7249, 0.7580, 0.8847,0.7804,0.7141, aCD_OH, aCD_Cl],
        [aDD_nc10,aDD_smo_model,aDD_smo_model_nc,0.4309, 0.5291, 0.5841, 0, 0, 0.4757, aDD_OH, aDD_Cl]])

# T0 -41.098	0.04	-168.932	0.032	2.495	0.137	7.722	0.791 
dC0=-41.098
dD0=-168.932
Dcd0=2.495
Ddd0=7.722

f=np.arange(0.2,1.1,0.1)
def plt_model(data):
    m=np.zeros([len(f),4])
    C0,D0,cd0,dd0,aC_model,aD_model,aCD_model,aDD_model=data
    for i in range(len(f)):
        m[i,0]=(C0+1000)*f[i]**(aC_model-1)-1000 # Carbon isotope
        m[i,1]=(D0+1000)*f[i]**(aD_model-1)-1000 # Hydrogen isotope
        m[i,2]=cd0+1000*(aCD_model-aC_model-aD_model+1)*np.log(f[i]) # 13CH3D
        m[i,3]=dd0+1000*(aDD_model-2*aD_model+1)*np.log(f[i]) # 12CH2D2
    return m
model_par=[dC0,dD0,Dcd0,Ddd0,aC_smo_model,aD_smo_model,aCD_smo_model,aDD_smo_model]
smmo_m=plt_model(model_par)
model_par=[dC0,dD0,Dcd0,Ddd0,aC_OH,aD_OH,aCD_OH,aDD_OH]
OH_m=plt_model(model_par)
model_par=[dC0,dD0,Dcd0,Ddd0,aC_Cl,aD_Cl,aCD_Cl,aDD_Cl]
Cl_m=plt_model(model_par)

fig3,ax3=plt.subplots(figsize=(12,12))
fig4,ax4=plt.subplots(figsize=(12,12))

# Clumped isotope 
c=np.arange(0.2,1.1,0.1)
ax3.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 3.5, markersize = 15)
# for i in range(len(r)):
#     ax3.plot(Dcd[i,:],Ddd[i,:], 'ro--', linewidth=3.0, alpha=c[i])
quick_plot(ax3,"D13CH3D","D12CH2D2","cdse","ddse",mask)
ax3.plot(smmo_m[:,2],smmo_m[:,3], 'ro--',linewidth=4.0, label="Ab initio (Wigner correction)")
# ax3.plot(smmo_m[:,6],smmo_m[:,7], 'ko--',linewidth=4.0, label="Ab initio (no correction)")
ax3.plot(OH_m[:,2],OH_m[:,3],"bo--",linewidth=4.0, label="OH radical")
ax3.plot(Cl_m[:,2],Cl_m[:,3],"go-.",linewidth=4.0, label="Cl radical")
ax3.set_xlim([-5,10])
ax3.set_ylim([-36,28])
ax3.legend(fontsize=20, bbox_to_anchor=(1.6,0.8))
ax3.set_xlabel('$\Delta^{13}$CH$_3$D (\u2030)', fontdict = font_labels)
ax3.set_ylabel('$\Delta^{12}$CH$_2$D$_2$ (\u2030)', fontdict = font_labels)
ax3.yaxis.set_minor_locator(MultipleLocator(2))
ax3.xaxis.set_minor_locator(MultipleLocator(0.5))
ax3.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax3.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)
# Bulk isotope
quick_plot(ax4,"d13C","dD","cse","dse",mask)
ax4.plot(smmo_m[:,0],smmo_m[:,1], 'ro--',linewidth=4.0, label="Ab initio (Wigner correction)")
# ax4.plot(smmo_m[:,4],smmo_m[:,5], 'ko--',linewidth=4.0, label="Ab initio (No correction)")
ax4.plot(OH_m[:,0],OH_m[:,1],"bo--",linewidth=4.0, label="OH radical")
ax4.plot(Cl_m[:,0],Cl_m[:,1],"go-.",linewidth=4.0, label="Cl radical")
ax4.set_xlabel('$\delta^{13}$C (\u2030)', fontdict = font_labels)
ax4.set_ylabel('$\delta$D (\u2030)', fontdict = font_labels)
ax4.yaxis.set_minor_locator(MultipleLocator(20))
ax4.xaxis.set_minor_locator(MultipleLocator(1))
ax4.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax4.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

# With f
fig5,ax5=plt.subplots(figsize=(12,6))
fig6,ax6=plt.subplots(figsize=(12,6))
fig7,ax7=plt.subplots(figsize=(12,6))
fig8,ax8=plt.subplots(figsize=(12,6))

def quick_plot_f(ax,y,ye,msk):
    if msk["T0"]==1:
        ax.errorbar(T0['f'],T0[y], xerr=T0['fse'], yerr=T0[ye], markersize=30,label=r'T0', fmt='*', 
                    markerfacecolor='purple', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["NC10"]==1:
        ax.errorbar(NC10['f'],NC10[y], xerr=NC10['fse'], yerr=NC10[ye], markersize=24,label=r'NC10', fmt='o', 
                    markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["NC10+ANME"]==1:
        ax.errorbar(NC10_ANME['f'],NC10_ANME[y], xerr=NC10_ANME['fse'], yerr=NC10_ANME[ye], markersize=24,label=r'NC10+ANME', fmt='^', 
                    markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["ANME2d"]==1:
        ax.errorbar(ANME2d['f'],ANME2d[y], xerr=ANME2d['fse'], yerr=ANME2d[ye], markersize=24,label=r'ANME-2d', fmt='s', 
                    markerfacecolor='yellow', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["Oct"]==1:
        ax.errorbar(Oct["f"],Oct[y],xerr=Oct["fse"],yerr=Oct[ye], markersize=18,label=r'NC10+ANME (pMMO inhibited)', fmt='D', 
                    markerfacecolor='red', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["BES"]==1:
        ax.errorbar(BES["f"],BES[y],xerr=BES["fse"],yerr=BES[ye], markersize=18,label=r'NC10+ANME (mcr inhibited)', fmt='d', 
                    markerfacecolor='purple', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
    if msk["previous data"]==1:
        ax.errorbar(AeOM_m['f'][AeOM_m['f']>=0.4],AeOM_m[y], xerr=AeOM_m['fse'], yerr=AeOM_m[ye], markersize=16,label=r'AeOM (Li et al., 2024)', fmt='o', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)
        ax.errorbar(AeOM_P1['f'],AeOM_P1[y], xerr=AeOM_P1['fse'], yerr=AeOM_P1[ye], markersize=16,label=r'AeOM (Krause et al., 2022)', fmt='^', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)
        ax.errorbar(AeOM_P2['f'],AeOM_P2[y], xerr=AeOM_P2['fse'], yerr=AeOM_P2[ye], markersize=16,label=r'AeOM (Wang et al., 2016)', fmt='D', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)    
        ax.errorbar(AOM_P['f'],AOM_P[y], xerr=AOM_P['fse'], yerr=AOM_P[ye], markersize=16,label=r'AOM (Liu et al., 2023)', fmt='s', 
                    markerfacecolor='white', markeredgecolor='black',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)   

quick_plot_f(ax5,"D13CH3D","cdse",mask)
# for i in range(len(r)):
#     ax5.plot(f,Dcd[i,:], 'ro--', linewidth=3.0, alpha=c[i])
# ax5.plot(f,smmo_m[:,2],'ro--',linewidth=4.0,label="Ab initio (Wigner correction)")
# ax5.plot(f,smmo_m[:,6],'ko--',linewidth=4.0,label="Ab initio (No correction)")
ax5.axes.invert_xaxis()
ax5.set_ylabel('$\Delta^{13}$CH$_3$D (\u2030)', fontdict = font_labels)
ax5.set_xlabel(r'$f$', fontdict = font_labels)
ax5.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax5.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

quick_plot_f(ax6,"D12CH2D2",'ddse',mask)
# ax6.plot(f,smmo_m[:,3],'ro--',linewidth=4.0,label="Ab initio (Wigner correction)")
# ax6.plot(f,smmo_m[:,7],'ko--',linewidth=4.0,label="Ab initio (No correction)")
# for i in range(len(r)):
#     ax6.plot(f,Ddd[i,:], 'ro--', linewidth=3.0, alpha=c[i])
ax6.axes.invert_xaxis()
ax6.set_xlim([1.08,0.08])
ax6.set_ylabel('$\Delta^{12}$CH$_2$D$_2$ (\u2030)', fontdict = font_labels)
ax6.set_xlabel(r'$f$', fontdict = font_labels)
ax6.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax6.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

quick_plot_f(ax7,"d13C","cse",mask)
# for i in range(len(r)):
#     ax7.plot(f,dC[i,:], 'ro--', linewidth=3.0, alpha=c[i])
# ax7.plot(f,smmo_m[:,0],'ro--',linewidth=4.0,label="Ab initio (Wigner correction)")
# ax7.plot(f,smmo_m[:,4],'ko--',linewidth=4.0,label="Ab initio (No correction)")
ax7.axes.invert_xaxis()
ax7.set_ylabel('$\delta^{13}$C (\u2030)', fontdict = font_labels)
ax7.set_xlabel(r'$f$', fontdict = font_labels)
ax7.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax7.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

quick_plot_f(ax8,"dD","dse",mask)
# for i in range(len(r)):
#     ax8.plot(f,dD[i,:], 'ro--', linewidth=3.0, alpha=c[i])
# ax8.plot(f,smmo_m[:,1],'ro--',linewidth=4.0,label="Ab initio (Wigner correction)")
# ax8.plot(f,smmo_m[:,5],'ko--',linewidth=4.0,label="Ab initio (No correction)")
ax8.axes.invert_xaxis()
ax8.set_ylabel('$\delta$D (\u2030)', fontdict = font_labels)
ax8.set_xlabel(r'$f$', fontdict = font_labels)
ax8.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax8.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

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

series_name=[r"$^{13}\alpha$",r"$^{D}\alpha$",r"$^{13CD}\alpha$",r"$^{DD}\alpha$"]
fig9,ax9=plt.subplots(figsize=(8,5))
for i in range(alphas.shape[1]):
    if i==0: # NC10 data
        ax9.scatter(series_name,alphas[:,i],color="orange",edgecolors="black",s=160,zorder=3,label="NC10 experiment")
    if i==1:
        ax9.scatter(series_name,alphas[:,i],color="red",edgecolors="black",s=160,marker="D", zorder=2, label="Ab initio (Wigner correction)")
    if i==2:
        ax9.scatter(series_name,alphas[:,i],color="black",edgecolors="black",s=160,marker="d", zorder=2, label="Ab initio (No correction)")
    if i==alphas.shape[1]-1:
        ax9.scatter(series_name,alphas[:,i],color="white",edgecolors="black",marker="o", s=100, label="Previous studies")
    else:
        ax9.scatter(series_name,alphas[:,i],color="white",edgecolors="black",marker="o", s=100)

ax9.set_ylim([0.38,1.05])
ax9.set_ylabel(r"Fractionation factors ($\alpha$)",fontsize=24)
ax9.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=24)
ax9.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=24)
ax9.legend(fontsize=16)


if mask["save_fig"]==1:
    fig1.savefig('clumped_sum.pdf', bbox_inches='tight')
    fig2.savefig('bulk_sum.pdf', bbox_inches='tight')
    fig3.savefig('clumped_model.pdf', bbox_inches='tight')
    fig4.savefig('bulk_model.pdf', bbox_inches='tight')
    fig5.savefig('CDf_model.pdf', bbox_inches='tight')
    fig6.savefig('DDf_model.pdf', bbox_inches='tight')
    fig7.savefig('dCf_model.pdf', bbox_inches='tight')
    fig8.savefig('deltaDf_model.pdf', bbox_inches='tight')
    fig9.savefig('compare_alpha.pdf', bbox_inches='tight')