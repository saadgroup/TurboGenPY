# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:08:01 2014

@author: Tony Saad
"""
#!/usr/bin/env python
from scipy import interpolate
import numpy as np
from numpy import pi
import time
import scipy
import scipy.io
from tkespec import compute_tke_spectrum
import isoturb
import isoturbo
import matplotlib.pyplot as plt
from fileformats import FileFormats
from isoio import writefile

#load an experimental specturm. Alternatively, specify it via a function call
cbcspec = np.loadtxt('cbc_spectrum.txt')
kcbc=cbcspec[:,0]*100
ecbc=cbcspec[:,1]*1e-6
especf = interpolate.interp1d(kcbc, ecbc,'cubic')

def cbc_specf(k):
  return especf(k)

def power_spec(k):
  Nu = 1*1e-3;
  L = 0.1;
  Li = 1;
  ch = 1;
  cl = 10;
  p0 = 8;
  c0 = pow(10,2);
  Beta = 2;
  Eta = Li/20.0;
  ES = Nu*Nu*Nu/(Eta*Eta*Eta*Eta);  
  x = k*Eta
  fh = np.exp(-Beta*pow(pow(x,4) + pow(ch,4), 0.25) - ch)
  x = k*L
  fl = pow( x/pow(x*x + cl, 0.5) , 5.0/3.0 + p0)
  espec = c0*pow(k,-5.0/3.0)*pow(ES,2.0/3.0)*fl*fh
  return espec

#----------------------------------------------------------------------------------------------
# __    __   ______   ________  _______         ______  __    __  _______   __    __  ________ 
#|  \  |  \ /      \ |        \|       \       |      \|  \  |  \|       \ |  \  |  \|        \
#| $$  | $$|  $$$$$$\| $$$$$$$$| $$$$$$$\       \$$$$$$| $$\ | $$| $$$$$$$\| $$  | $$ \$$$$$$$$
#| $$  | $$| $$___\$$| $$__    | $$__| $$        | $$  | $$$\| $$| $$__/ $$| $$  | $$   | $$   
#| $$  | $$ \$$    \ | $$  \   | $$    $$        | $$  | $$$$\ $$| $$    $$| $$  | $$   | $$   
#| $$  | $$ _\$$$$$$\| $$$$$   | $$$$$$$\        | $$  | $$\$$ $$| $$$$$$$ | $$  | $$   | $$   
#| $$__/ $$|  \__| $$| $$_____ | $$  | $$       _| $$_ | $$ \$$$$| $$      | $$__/ $$   | $$   
# \$$    $$ \$$    $$| $$     \| $$  | $$      |   $$ \| $$  \$$$| $$       \$$    $$   | $$   
#  \$$$$$$   \$$$$$$  \$$$$$$$$ \$$   \$$       \$$$$$$ \$$   \$$ \$$        \$$$$$$     \$$   
#----------------------------------------------------------------------------------------------

# specify whether you want to use threads or not to generate turbulence
use_threads = True

mmodes =[100, 200, 400]#, 800, 1000, 2000, 4000, 6000, 8000, 10000, 20000, 40000, 60000, 80000, 100000]
ngrid = [32]

errors = np.zeros([len(ngrid),len(mmodes)])
times = np.zeros([len(ngrid),len(mmodes)])

plt.figure(0)
plt.loglog(kcbc,ecbc,'k-')
i = 0
for N in ngrid:
  j = 0
  for nmodes in mmodes:     
    # input domain size in the x, y, and z directions. This value is typically
    # based on the largest length scale that your data has. For the cbc data,
    # the largest length scale corresponds to a wave number of 15, hence, the
    # domain size is L = 2pi/15.
    lx = 2.0*pi/15.0
    ly = 2.0*pi/15.0
    lz = 2.0*pi/15.0
    
    # input number of cells (cell centered control volumes). This will
    # determine the maximum wave number that can be represented on this grid.
    # see wnn below
    nx = N         # number of cells in the x direction
    ny = N         # number of cells in the y direction
    nz = N         # number of cells in the z direction
    
    # enter the smallest wavenumber represented by this spectrum
    wn1 = 15 #determined here from cbc spectrum properties
    
    #------------------------------------------------------------------------------
    # END USER INPUT
    #------------------------------------------------------------------------------
    t0 = time.time()
    
    if use_threads:
      u,v,w = isoturbo.generate_isotropic_turbulence(lx,ly,lz,nx,ny,nz,nmodes,wn1,cbc_specf,False, False, FileFormats.FLAT)
    else:
      u,v,w = isoturb.generate_isotropic_turbulence(lx,ly,lz,nx,ny,nz,nmodes,wn1,cbc_specf) # this doesnt support file formats yet
    
    t1 = time.time()
    times[i,j] = t1-t0
    print 'it took me ', t1 - t0, ' s to generate the isotropic turbulence.'
    
    # verify that the generated velocities fit the spectrum
    knyquist, wavenumbers, tkespec = compute_tke_spectrum(u,v,w,lx,ly,lz,False)
    
    # analyze how well we fit the input spectrum
    espec = cbc_specf(kcbc) # compute the cbc original spec
    
    #find index of nyquist limit
    idx = (np.where(wavenumbers==knyquist)[0][0]) -1
    maxfreq = 5
    exact = cbc_specf(wavenumbers[maxfreq:idx])
    num = tkespec[maxfreq:idx]
    diff = np.abs(exact - num)/exact
    errMean = np.sqrt(np.mean(diff*diff))
    errors[i,j] = errMean
    plt.figure(1)
    plt.plot(wavenumbers[maxfreq:idx],diff)
    print 'mean error = ', np.linalg.norm(diff,2)
    plt.figure(0)
    plt.loglog(wavenumbers, tkespec,'-o')
    plt.axvline(x=knyquist, linestyle='--', color='black')
    plt.grid()
    plt.show()
    j = j+1
  i = i+1
