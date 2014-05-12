# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:14:44 2014

@author: tsaad
"""
import numpy as np
from numpy.fft import fftn
from numpy import sqrt, zeros, conj, pi, arange, ones, convolve

def movingaverage(interval, window_size):
    window= ones(int(window_size))/float(window_size)
    return convolve(interval, window, 'same')

def compute_tke_spectrum(u,v,w,lx,ly,lz,smooth):
  """
  Given a velocity field u, v, w, this function computes the kinetic energy
  spectrum of that velocity field in wave space.

  Parameters:
  -----------  
  u: 3D array
    The x-velocity component.
  v: 3D array
    The y-velocity component.
  w: 3D array
    The z-velocity component.    
  lx: float
    The domain size in the x-direction.
  ly: float
    The domain size in the y-direction.
  lz: float
    The domain size in the z-direction.
  smooth: boolean
    A boolean to smooth the computed spectrum for nice visualization.
  """
  nx = len(u[:,0,0])
  ny = len(v[0,:,0])
  nz = len(w[0,0,:])
  
  nt= nx*ny*nz
  n = int(np.round(np.power(nt,1.0/3.0)))
  
  uh = fftn(u)/nt
  vh = fftn(v)/nt
  wh = fftn(w)/nt
  
  tkeh = zeros((nx,ny,nz))
  tkeh = 0.5*(uh*conj(uh) + vh*conj(vh) + wh*conj(wh)).real
  
  k0x = 2.0*pi/lx
  k0y = 2.0*pi/ly
  k0z = 2.0*pi/lz
  
  knorm = (k0x + k0y + k0z)/3.0
  
  kxmax = nx/2
  kymax = ny/2
  kzmax = nz/2
  
  wave_numbers = knorm*arange(0,n)
  
  tke_spectrum = zeros(len(wave_numbers))
  
  for kx in xrange(nx):
    rkx = kx
    if (kx > kxmax):
      rkx = rkx - (nx)
    for ky in xrange(ny):
      rky = ky
      if (ky>kymax):
        rky=rky - (ny)
      for kz in xrange(nz):        
        rkz = kz
        if (kz>kzmax):
          rkz = rkz - (nz)
        rk = sqrt(rkx*rkx + rky*rky + rkz*rkz)
        k = int(np.round(rk))
        tke_spectrum[k] = tke_spectrum[k] + tkeh[kx,ky,kz]/knorm;        

  if smooth:
    tkespecsmooth = movingaverage(tke_spectrum, 5) #smooth the spectrum
    tkespecsmooth[0:4] = tke_spectrum[0:4] # get the first 4 values from the original data
    tke_spectrum = tkespecsmooth

  return wave_numbers, tke_spectrum
