#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:53:14 2019

@author: Heng Yang and Haolan Tang

Modified by Ed Young, UCLA, for general use with python3
March 18, 2021

Note: In case of "Permission denied", do not forget to change the permission mode first
      by using the following UNIX command in the .py file's directory:   
          chmod 777 ./york_regression.py

Invoke the program using >python3 york_regression_edy.py

"""


import numpy as np
from math import cos, pi, atan, tan
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2


# Input the names of data files
print('')
print('---------------------------------------')
print('           NEW YORK REGRESSION')
print('---------------------------------------')
print('Requires a data file formatted as follows: x xerr y yerr corr_coef')
print(' ...readable formats include .dat, .txt. and .csv')
print('')
input_line = input("ENTER the input filename including the extension name:")
file_1_path = input_line
print('Your data file is: ',file_1_path,'\n')

# Read the file into tables 
data_table = pd.read_csv(file_1_path, sep=' ',header=None)
data_table.columns = ['x','sigx','y','sigy','r']

minx=min(data_table['x'])
print('Minimum x =',minx)
maxx=max(data_table['x'])
print('Maximum x =',maxx)
miny=min(data_table['y'])
print('Minimum y =',miny)
maxy=max(data_table['y'])
print('Maximum y =',maxy)
print('')


#York regression function
#
#--------------------------------------------------------------------------
def York_regression(data_table):
    X = data_table['x']
    y = data_table['y']
    sigX = data_table['sigx']
    sigy = data_table['sigy']
    r = data_table['r']
    
    b = -1  
    for iter in range(500):    
        b0 = b
        #   Weights...
        wt = 1/( sigy**2 + (b**2)*(sigX**2) - 2*r*sigX*sigy )       
        #   centroid (weighted mean) 
        sXoss = np.sum(X*wt) / np.sum(wt)
        syoss = np.sum(y*wt) / np.sum(wt) 

        # Calculate intercept a and slope b.
        t1 = (X-sXoss)*(sigy**2)
        t2 = (y-syoss)*(sigX**2)*b
        t3 = sigX*sigy*r
        st2 = np.sum( (wt**2)*(y-syoss)*(t1+t2-t3*(y-syoss)) )
        st3 = np.sum( (wt**2)*(X-sXoss)*(t1+t2-b*t3*(X-sXoss)) )    
        b = st2/st3
        #print(b)
        if abs(b-b0)<10**(-8): break  

    # best fit slope and intercept 
    fit_slope = b
    fit_intercept = syoss - sXoss*fit_slope 
    
    # Calculate errors in intercept and slope
    sig_slope = np.sqrt(1/np.sum(wt*(X-sXoss)**2))
    sig_intercept = sig_slope * np.sqrt(np.sum(wt*X**2/np.sum(wt)))

    # Calculate chi squared statistic
    # (sum of squared deviations from the best-fit line)
    chi_2 = np.sum(wt*(y-fit_intercept-fit_slope*X)**2)
    df = len(y)-2
    # Compute reduced chi squared
    rchi_2 = chi_2/df 

    # Compute "goodness of fit"
    # q is the probability that chi^2 should exceed the statistic X^2
    q = 1-chi2.cdf(rchi_2,df)
    
    return fit_slope, fit_intercept, sig_slope, sig_intercept, chi_2, rchi_2, q, sXoss, syoss
#--------------------------------------------------------------------------------
    

#Print results
#
regression = York_regression(data_table)

print('Regeresson results:')
print(' ')
print("  Slope     = %9.6g +/- %8.4g " % (regression[0],regression[2]))
print("  Intercept = %9.6g +/- %8.6g " % (regression[1],regression[3]))
print('')
print('  Chi-squared = %8.5g' % regression[4])
print("  Reduced chi-squared = %8.4f" % regression[5])
print("  Goodness of fit = %8.3f" %regression[6])
print('')
print('  x centroid = %8.5g' % regression[7])
print('  y centroid = %8.5g' % regression[8])
print('')
a_file = open('York.txt', 'w')
a_file.write('Regeresson results:\n')
a_file.write('\n')
a_file.write("  Slope     = %9.6g +/- %8.4g \n" % (regression[0],regression[2]))
a_file.write("  Intercept = %9.6g +/- %8.6g \n" % (regression[1],regression[3]))
a_file.write('\n')
a_file.write('  Chi-squared = %8.5g \n' % regression[4])
a_file.write("  Reduced chi-squared = %8.4f \n" % regression[5])
a_file.write("  Goodness of fit = %8.3f \n" % regression[6])
a_file.write('\n')
a_file.write('  x Centroid = %8.5g \n' %regression[7])
a_file.write('  y centroid = %8.5g \n' % regression[8])
a_file.close()
print('Results in file York.txt')
print('2-sigma error envelope curves in plus_error.txt and minus_error.txt')
print('See plot and dismiss when ready')


#Calculate error envelopes based on equations from Ludwig(1980, EPSL 46, p. 212),
#equations 30 thourgh 31.
#
s=regression[0]
ri=regression[1]
cx=regression[7]
cy=regression[8]
ds=regression[2]
dri=regression[3]
dtheta=ds*(cos(atan(s))**2.0)
sp=tan(atan(s)+dtheta)+tan(atan(s)-dtheta)
sp=sp/2.00
rip=cy-sp*cx
dsp=tan(atan(s)+dtheta)-tan(atan(s)-dtheta)
dsp=dsp/2.00
drip=dri+(dsp-ds)*cx

#Setup x values for plotting
num=200
regxdata=np.linspace(minx-0.05*minx,maxx+0.06*maxx,num)

#Calculate predicted y values +/- errors
xplus=regxdata
yplus=rip+sp*xplus
temp=drip**2.0+(dsp**2.0)*xplus*(xplus-2.0*cx)
temp=np.sqrt(temp)
#yplus=yplus+temp #for 1 sigma envelope
yplus=yplus+2.0*temp #for 2 sigma envelope
yminus=rip+sp*xplus
#yminus=yminus-temp  #for 1 sigma envelope
yminus=yminus-2.0*temp  #for 2 sigma envelope

#Save these curves for error envelope to files
a_file = open('plus_error.txt', 'w')
for i in range(0,num):
    a_file.write("%10.5e " % xplus[i])
    a_file.write("")
    a_file.write("%13.8e\n" % yplus[i])
a_file.close()

a_file = open('minus_error.txt', 'w')
for i in range(0,num):
    a_file.write("%10.5e " % xplus[i])
    a_file.write("")
    a_file.write("%13.8e\n" % yminus[i])
a_file.close()

#Plot results
regydata=s*regxdata+ri
plt.figure(1)
plt.errorbar(data_table['x'],data_table['y'],xerr=data_table['sigx'],yerr=data_table['sigy'],fmt='o',color='black')
plt.plot(regxdata,regydata,linestyle='-')
plt.plot(regxdata,yplus,linestyle='--')
plt.plot(regxdata,yminus,linestyle='--')
plt.show()
print('End')


