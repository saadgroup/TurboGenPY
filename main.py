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
import isoio

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
use_parallel = True
patches=[3,2,1]

# specify whether you want to generate velocities at cell centered or staggered
cell_centered = False

#set the number of modes you want to use to represent the velocity.
nmodes =100
N = 32
# write to file
enableIO = False # enable writing to file
fileformat = FileFormats.FLAT  # Specify the file format supported formats are: FLAT, IJK, XYZ

# save the velocity field as a matlab matrix (.mat)
savemat = False

# compute the mean of the fluctuations for verification purposes
computeMean = False

# check the divergence of the generated velocity field
checkdivergence = False

# input domain size in the x, y, and z directions. This value is typically
# based on the largest length scale that your data has. For the cbc data,
# the largest length scale corresponds to a wave number of 15, hence, the
# domain size is L = 2pi/15.
lx = 9*2.0*pi/100.0
ly = 9*2.0*pi/100.0
lz = 9*2.0*pi/100.0

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
if use_parallel:
  u,v,w = isoturbo.generate_isotropic_turbulence(patches,lx,ly,lz,nx,ny,nz,nmodes,wn1,cbc_specf, cell_centered)
else:
  u,v,w = isoturb.generate_isotropic_turbulence(lx,ly,lz,nx,ny,nz,nmodes,wn1,cbc_specf, cell_centered)
t1 = time.time()
print 'it took me ', t1 - t0, ' s to generate the isotropic turbulence.'

dx = lx/nx
dy = ly/ny
dz = lz/nz

if (enableIO):
  if (use_parallel):
    isoio.writefileparallel(u,v,w,dx,dy,dz,fileformat)
  else:
    isoio.writefile('u.txt','x',dx,dy,dz,u,fileformat)
    isoio.writefile('v.txt','y',dx,dy,dz,v,fileformat)
    isoio.writefile('w.txt','z',dx,dy,dz,w,fileformat)  

if(savemat):
  data={} # CREATE empty dictionary
  data['U'] = u
  data['V'] = v
  data['W'] = w  
  scipy.io.savemat('uvw.mat',data)
  
# compute mean velocities
if computeMean:
  umean = np.mean(u)
  vmean = np.mean(v)
  wmean = np.mean(w)  
  print 'mean u = ', umean
  print 'mean v = ', vmean
  print 'mean w = ', wmean

  ufluc = umean - u
  vfluc = vmean - v
  wfluc = wmean - w  
  
  print 'mean u fluct = ', np.mean(ufluc)
  print 'mean v fluct = ', np.mean(vfluc)
  print 'mean w fluct = ', np.mean(wfluc)
  
  ufrms = np.mean(ufluc*ufluc)
  vfrms = np.mean(vfluc*vfluc)
  wfrms = np.mean(wfluc*wfluc)
  
  print 'u fluc rms = ', np.sqrt(ufrms)
  print 'v fluc rms = ', np.sqrt(vfrms)
  print 'w fluc rms = ', np.sqrt(wfrms)

# check divergence
if checkdivergence:
  count = 0
  for k in range(0,nz-1):
    for j in range(0,ny-1):
      for i in range(0,nx-1):
        src = (u[i+1,j,k] - u[i,j,k])/dx + (v[i,j+1,k] - v[i,j,k])/dy + (w[i,j,k+1] - w[i,j,k])/dz
        if(src > 1e-2):
          count += 1
  print 'cells with divergence: ', count      

# verify that the generated velocities fit the spectrum
knyquist, wavenumbers, tkespec = compute_tke_spectrum(u,v,w,lx,ly,lz,True)

# analyze how well we fit the input spectrum
espec = cbc_specf(kcbc) # compute the cbc original spec

# compute the RMS error committed by the generated spectrum
#find index of nyquist limit
idx = (np.where(wavenumbers==knyquist)[0][0]) -1
exact = cbc_specf(wavenumbers[2:idx])
num = tkespec[2:idx]
diff = np.abs(exact - num)/exact
rmsE = np.sqrt(np.mean(diff*diff))
print 'RMS Error = ', rmsE

q, ((p1,p2),(p3,p4)) = plt.subplots(2,2)

p1.plot(kcbc, espec, 'ob', kcbc, ecbc, '-')
p1.set_title('Interpolated Spectrum')
p1.grid()
p1.set_xlabel('wave number')
p1.set_ylabel('E')

p2.loglog(kcbc, ecbc, '-', wavenumbers, tkespec, 'ro-')
p2.axvline(x=knyquist, linestyle='--', color='black')
p2.set_title('Spectrum of generated turbulence')
p2.grid()

# contour plot
p3.matshow(u[:,:,nz/2])
p3.set_title('u velocity')

p4.matshow(v[:,:,nz/2])
p4.set_title('v velocity')

plt.show()
