# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:08:01 2014

@author: Tony Saad
"""
#!/usr/bin/env python
from scipy import interpolate
import numpy as np
from numpy import pi
from tkespec import compute_tke_spectrum
from isoturb import generate_isotropic_turbulence
import matplotlib.pyplot as plt

#load an experimental specturm. Alternatively, specify it via a function call
cbcspec = np.loadtxt('cbc_spectrum.txt')
kcbc=cbcspec[:,0]*100
ecbc=cbcspec[:,1]*1e-6
especf = interpolate.interp1d(kcbc, ecbc,'slinear')

#USER INPUT
#set the number of modes you want to use to represent the velocity.
nmodes =100

# write to file
enableIO = False

# compute the mean of the fluctuations for verification purposes
computeMean = True

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
u, v, w = generate_isotropic_turbulence(lx,ly,lz,nx,ny,nz,nmodes,wn1,especf,True, True)

# verify that the generated velocities fit the spectrum
wavenumbers, tkespec = compute_tke_spectrum(u,v,w,lx,ly,lz, True)
plt.loglog(kcbc, ecbc, '-', wavenumbers, tkespec, 'ro-')
plt.title('Spectrum of generated turbulence')
plt.show()