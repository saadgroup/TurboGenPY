# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:08:01 2014

@author: Tony Saad
"""
#!/usr/bin/env python
from scipy import interpolate
import numpy as np
import time
from numpy import pi
from tkespec import compute_tke_spectrum
# ATTENTION: TO USE THE THREADED VERSION, REPLACE isoturb WITH isoturbo
from isoturb import generate_isotropic_turbulence
import matplotlib.pyplot as plt

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

#USER INPUT
#set the number of modes you want to use to represent the velocity.
nmodes =100

# write to file
enableIO = True

# compute the mean of the fluctuations for verification purposes
computeMean = False

# input domain size in the x, y, and z directions. This value is typically
# based on the largest length scale that your data has. For the cbc data,
# the largest length scale corresponds to a wave number of 15, hence, the
# domain size is L = 2pi/15.
lx = 2*pi/15
ly = 2*pi/15
lz = 2*pi/15

# input number of cells (cell centered control volumes). This will
# determine the maximum wave number that can be represented on this grid.
# see wnn below
nx = 32         # number of cells in the x direction
ny = 32         # number of cells in the y direction
nz = 32         # number of cells in the z direction

# enter the smallest wavenumber represented by this spectrum
wn1 = 15 #determined here from cbc spectrum properties
t0 = time.time()
u,v,w = generate_isotropic_turbulence(lx,ly,lz,nx,ny,nz,nmodes,wn1,cbc_specf,computeMean, enableIO)

t1 = time.time()
print 'it took me ', t1 - t0, ' s to generate the isotropic turbulence.'

# verify that the generated velocities fit the spectrum
knyquist, wavenumbers, tkespec = compute_tke_spectrum(u,v,w,lx,ly,lz, True)

q, ((p1,p2),(p3,p4)) = plt.subplots(2,2)
espec = cbc_specf(kcbc)
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
