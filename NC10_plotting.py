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

NC10=data[data['label']=='1'] # NC10
NC10_ANME=data[data['label']=='2'] # NC10+ANME
AeOM_P=data[data['label']=='P']
AeOM_m=AeOM_P[AeOM_P['f']>=0.4]

fig1,ax1=plt.subplots(figsize=(12,12))
fig2,ax2=plt.subplots(figsize=(12,12))
ax1.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 2.5, markersize = 15)
for i in range(len(equib)):
    if equib['p'].iloc[i]==1:
        ax1.scatter(equib['D13CH3D'].iloc[i], equib['D12CH2D2'].iloc[i],color='black',s=60)

ax1.errorbar(NC10['D13CH3D'],NC10['D12CH2D2'],xerr=NC10['cdse'],yerr=NC10['ddse'], markersize=16,label=r'NC10', fmt='o', 
            markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax1.errorbar(NC10_ANME['D13CH3D'],NC10_ANME['D12CH2D2'],xerr=NC10_ANME['cdse'],yerr=NC10_ANME['ddse'], markersize=16,label=r'NC10+ANME', fmt='^', 
            markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax1.errorbar(AeOM_m['D13CH3D'],AeOM_m['D12CH2D2'],xerr=AeOM_m['cdse'],yerr=AeOM_m['ddse'], markersize=9,label=r'AeOM (Li et al., 2024)', fmt='s', 
            markerfacecolor='gray', markeredgecolor='gray',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)

# Bulk isotope
ax2.errorbar(NC10['d13C'],NC10['dD'],xerr=NC10['cse'],yerr=NC10['dse'], markersize=16,label=r'NC10', fmt='o', 
            markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax2.errorbar(NC10_ANME['d13C'],NC10_ANME['dD'],xerr=NC10_ANME['cse'],yerr=NC10_ANME['dse'], markersize=16,label=r'NC10+ANME', fmt='^', 
            markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax2.errorbar(AeOM_m['d13C'],AeOM_m['dD'],xerr=AeOM_m['cse'],yerr=AeOM_m['dse'], markersize=9,label=r'AeOM (Li et al., 2024)', fmt='s', 
            markerfacecolor='gray', markeredgecolor='gray',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)

ax1.legend(fontsize=24)
ax1.set_xlabel('$\Delta^{13}$CH$_3$D (\u2030)', fontdict = font_labels)
ax1.set_ylabel('$\Delta^{12}$CH$_2$D$_2$ (\u2030)', fontdict = font_labels)
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

aC_nc10=0.9778	
aD_nc10=0.8019	
aCD_nc10=0.7842724	
aDD_nc10=0.629614

dC0=-41.098
dD0=-168.932
Dcd0=2.495
Ddd0=7.722
# mixing of two processes
f=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
r=np.arange(0.4,1.1,0.1)
aC_mix=np.zeros(len(f))
aD_mix=np.zeros(len(f))
aCD_mix=np.zeros(len(f))
aDD_mix=np.zeros(len(f))

# Resulting isotope signals
dC=np.zeros((len(f),len(r)))
dD=np.zeros((len(f),len(r)))
Dcd=np.zeros((len(f),len(r)))
Ddd=np.zeros((len(f),len(r)))

for i in range(len(f)):
    aC_mix[i]=aC_anme_hs*(1-f[i])+aC_nc10*f[i]
    aD_mix[i]=aD_anme_hs*(1-f[i])+aD_nc10*f[i]
    aCD_mix[i]=aCD_anme_hs*(1-f[i])+aCD_nc10*f[i]
    aDD_mix[i]=aDD_anme_hs*(1-f[i])+aDD_nc10*f[i]
    for j in range(len(r)):
        dC[i,j] = r[j]**(aC_mix[i]-1)*(dC0+1000)-1000
        dD[i,j] = r[j]**(aD_mix[i]-1)*(dD0+1000)-1000
        Dcd[i,j] = Dcd0+1000*(aCD_mix[i]-aC_mix[i]-aD_mix[i]+1)*np.log(r[j])
        Ddd[i,j] = Ddd0+1000*(aDD_mix[i]-2*aD_mix[i]+1)*np.log(r[j])

fig3,ax3=plt.subplots(figsize=(12,12))
fig4,ax4=plt.subplots(figsize=(12,12))

# Clumped isotope 
c=np.arange(0.2,1.1,0.1)
ax3.plot(equib['D13CH3D'],equib['D12CH2D2'],'-k', label = 'Equilibrium', linewidth = 3.5, markersize = 15)
for i in range(len(f)):
    ax3.plot(Dcd[i,:],Ddd[i,:], 'ro--', linewidth=3.0, alpha=c[i])
ax3.errorbar(NC10['D13CH3D'],NC10['D12CH2D2'],xerr=NC10['cdse'],yerr=NC10['ddse'], markersize=16,label=r'NC10', fmt='o', 
            markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax3.errorbar(NC10_ANME['D13CH3D'],NC10_ANME['D12CH2D2'],xerr=NC10_ANME['cdse'],yerr=NC10_ANME['ddse'], markersize=16,label=r'NC10+ANME', fmt='^', 
            markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax3.errorbar(AeOM_m['D13CH3D'],AeOM_m['D12CH2D2'],xerr=AeOM_m['cdse'],yerr=AeOM_m['ddse'], markersize=10,label=r'AeOM (Li et al., 2024)', fmt='s', 
            markerfacecolor='gray', markeredgecolor='gray',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)
ax3.legend(fontsize=24)
ax3.set_xlabel('$\Delta^{13}$CH$_3$D (\u2030)', fontdict = font_labels)
ax3.set_ylabel('$\Delta^{12}$CH$_2$D$_2$ (\u2030)', fontdict = font_labels)
ax3.yaxis.set_minor_locator(MultipleLocator(4))
ax3.xaxis.set_minor_locator(MultipleLocator(2))
ax3.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax3.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)
# Bulk isotope
for i in range(len(f)):
    ax4.plot(dC[i,:],dD[i,:], 'ro--', linewidth=3.0, alpha=c[i])
ax4.errorbar(NC10['d13C'],NC10['dD'],xerr=NC10['cse'],yerr=NC10['dse'], markersize=16,label=r'NC10', fmt='o', 
            markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax4.errorbar(NC10_ANME['d13C'],NC10_ANME['dD'],xerr=NC10_ANME['cse'],yerr=NC10_ANME['dse'], markersize=16,label=r'NC10+ANME', fmt='^', 
            markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax4.errorbar(AeOM_m['d13C'],AeOM_m['dD'],xerr=AeOM_m['cse'],yerr=AeOM_m['dse'], markersize=10,label=r'AeOM (Li et al., 2024)', fmt='s', 
            markerfacecolor='gray', markeredgecolor='gray',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)
ax4.set_xlabel('$\delta^{13}$C (\u2030)', fontdict = font_labels)
ax4.set_ylabel('$\delta$D (\u2030)', fontdict = font_labels)
ax4.yaxis.set_minor_locator(MultipleLocator(20))
ax4.xaxis.set_minor_locator(MultipleLocator(2))
ax4.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax4.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

# With f
fig5,ax5=plt.subplots(figsize=(12,6))
fig6,ax6=plt.subplots(figsize=(12,6))
fig7,ax7=plt.subplots(figsize=(12,6))
fig8,ax8=plt.subplots(figsize=(12,6))

ax5.errorbar(NC10['f'],NC10['D13CH3D'], xerr=NC10['fse'], yerr=NC10['cdse'], markersize=16,label=r'NC10', fmt='o', 
            markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax5.errorbar(NC10_ANME['f'],NC10_ANME['D13CH3D'], xerr=NC10_ANME['fse'], yerr=NC10_ANME['cdse'], markersize=16,label=r'NC10+ANME', fmt='^', 
            markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax5.errorbar(AeOM_m['f'][AeOM_m['f']>=0.4],AeOM_m['D13CH3D'], xerr=AeOM_m['fse'], yerr=AeOM_m['cdse'], markersize=10,label=r'AeOM (Li et al., 2024)', fmt='s', 
            markerfacecolor='gray', markeredgecolor='gray',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)
for i in range(len(f)):
    ax5.plot(r,Dcd[i,:], 'ro--', linewidth=3.0, alpha=c[i])
ax5.axes.invert_xaxis()
ax5.set_ylabel('$\Delta^{13}$CH$_3$D (\u2030)', fontdict = font_labels)
ax5.set_xlabel(r'$f$', fontdict = font_labels)
ax5.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax5.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

ax6.errorbar(NC10['f'],NC10['D12CH2D2'], xerr=NC10['fse'], yerr=NC10['ddse'], markersize=16,label=r'NC10', fmt='o', 
            markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax6.errorbar(NC10_ANME['f'],NC10_ANME['D12CH2D2'], xerr=NC10_ANME['fse'], yerr=NC10_ANME['ddse'], markersize=16,label=r'NC10+ANME', fmt='^', 
            markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax6.errorbar(AeOM_m['f'][AeOM_m['f']>=0.4],AeOM_m['D12CH2D2'], xerr=AeOM_m['fse'], yerr=AeOM_m['ddse'], markersize=10,label=r'AeOM (Li et al., 2024)', fmt='s', 
            markerfacecolor='gray', markeredgecolor='gray',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)
for i in range(len(f)):
    ax6.plot(r,Ddd[i,:], 'ro--', linewidth=3.0, alpha=c[i])
ax6.axes.invert_xaxis()
ax6.set_ylabel('$\Delta^{12}$CH$_2$D$_2$ (\u2030)', fontdict = font_labels)
ax6.set_xlabel(r'$f$', fontdict = font_labels)
ax6.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax6.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

ax7.errorbar(NC10['f'],NC10['d13C'], xerr=NC10['fse'], yerr=NC10['cse'], markersize=16,label=r'NC10', fmt='o', 
            markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax7.errorbar(NC10_ANME['f'],NC10_ANME['d13C'], xerr=NC10_ANME['fse'], yerr=NC10_ANME['cse'], markersize=16,label=r'NC10+ANME', fmt='^', 
            markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax7.errorbar(AeOM_m['f'][AeOM_m['f']>=0.4],AeOM_m['d13C'], xerr=AeOM_m['fse'], yerr=AeOM_m['cse'], markersize=10,label=r'AeOM (Li et al., 2024)', fmt='s', 
            markerfacecolor='gray', markeredgecolor='gray',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)
for i in range(len(f)):
    ax7.plot(r,dC[i,:], 'ro--', linewidth=3.0, alpha=c[i])
ax7.axes.invert_xaxis()
ax7.set_ylabel('$\delta^{13}$C (\u2030)', fontdict = font_labels)
ax7.set_xlabel(r'$f$', fontdict = font_labels)
ax7.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax7.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

ax8.errorbar(NC10['f'],NC10['dD'], xerr=NC10['fse'], yerr=NC10['dse'], markersize=16,label=r'NC10', fmt='o', 
            markerfacecolor='orange', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax8.errorbar(NC10_ANME['f'],NC10_ANME['dD'], xerr=NC10_ANME['fse'], yerr=NC10_ANME['dse'], markersize=16,label=r'NC10+ANME', fmt='^', 
            markerfacecolor='blue', markeredgecolor='black',markeredgewidth=2.5, ecolor='black', elinewidth=2.5, zorder=2)
ax8.errorbar(AeOM_m['f'][AeOM_m['f']>=0.4],AeOM_m['dD'], xerr=AeOM_m['fse'], yerr=AeOM_m['dse'], markersize=10,label=r'AeOM (Li et al., 2024)', fmt='s', 
            markerfacecolor='gray', markeredgecolor='gray',markeredgewidth=2.5, ecolor='gray', elinewidth=2.5, zorder=-1)
for i in range(len(f)):
    ax8.plot(r,dD[i,:], 'ro--', linewidth=3.0, alpha=c[i])
ax8.axes.invert_xaxis()
ax8.set_ylabel('$\delta$D (\u2030)', fontdict = font_labels)
ax8.set_xlabel(r'$f$', fontdict = font_labels)
ax8.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=32)
ax8.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=32)

fig1.savefig('clumped_sum.pdf', bbox_inches='tight')
fig2.savefig('bulk_sum.pdf', bbox_inches='tight')
fig3.savefig('clumped_model.pdf', bbox_inches='tight')
fig4.savefig('bulk_model.pdf', bbox_inches='tight')
fig5.savefig('CDf_model.pdf', bbox_inches='tight')
fig6.savefig('DDf_model.pdf', bbox_inches='tight')
fig7.savefig('dCf_model.pdf', bbox_inches='tight')
fig8.savefig('deltaDf_model.pdf', bbox_inches='tight')