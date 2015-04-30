# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:31:54 2014

@author: tsaad
"""
import numpy as np
import gzip
from numpy import sin, cos, sqrt, ones, zeros, pi, arange
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def generate_isotropic_turbulence(lx,ly,lz,nx,ny,nz,nmodes,wn1,especf):
  ## grid generation
  # generate cell centered x-grid
  dx = lx/nx
  dy = ly/ny  
  dz = lz/nz
  
  ## START THE FUN!
  # compute random angles
  phi =   2.0*pi*np.random.uniform(0.0,1.0,nmodes);
  nu = np.random.uniform(0.0,1.0,nmodes);
  theta = np.arccos(2.0*nu -1.0);
  psi   = np.random.uniform(-pi/2.0,pi/2.0,nmodes);
  alfa  = 2.0*pi*np.random.uniform(0.0,1.0,nmodes);  
#  mu = np.random.uniform(0.0,1.0,nmodes);  
#  alfa = np.arccos(2.0*mu -1.0);
  
  # highest wave number that can be represented on this grid (nyquist limit)
  wnn = max(np.pi/dx, max(np.pi/dy, np.pi/dz));
  print 'I will generate data up to wave number: ', wnn
  
  # wavenumber step
  dk = (wnn-wn1)/nmodes
  
  # wavenumber at cell centers
  wn = wn1 + 0.5*dk + arange(0,nmodes)*dk

  dkn = ones(nmodes)*dk
  
  #   wavenumber vector from random angles
  kx = sin(theta)*cos(phi)*wn
  ky = sin(theta)*sin(phi)*wn
  kz = cos(theta)*wn

  # create divergence vector
  ktx = np.sin(kx*dx/2.0)/(dx)
  kty = np.sin(ky*dy/2.0)/(dy)
  ktz = np.sin(kz*dz/2.0)/(dz)    

#  # Use Davidson's Method to enforce Divergence Free Condition
#  ktmag = sqrt(ktx*ktx + kty*kty + ktz*ktz)
#  theta = np.arccos(kzstag/kstagmag)
#  phi = np.arctan2(kystag,kxstag)
#  sxm = cos(phi)*cos(theta)*cos(alfa) - sin(phi)*sin(alfa)
#  sym = sin(phi)*cos(theta)*cos(alfa) + cos(phi)*sin(alfa)
#  szm = -sin(theta)*cos(alfa)   

  # another method to generate sigma = zeta x k_tilde, pick zeta randomly
#  np.random.seed(3)
  phi1 =   2.0*pi*np.random.uniform(0.0,1.0,nmodes);
  nu1 = np.random.uniform(0.0,1.0,nmodes);
  theta1 = np.arccos(2.0*nu1 -1.0);
  zetax = sin(theta1)*cos(phi1)
  zetay = sin(theta1)*sin(phi1)
  zetaz = cos(theta1)
  sxm =  zetay*ktz - zetaz*kty
  sym = -( zetax*ktz - zetaz*ktx  )
  szm = zetax*kty - zetay*ktx
  smag = sqrt(sxm*sxm + sym*sym + szm*szm)
  sxm = sxm/smag
  sym = sym/smag
  szm = szm/smag  
    
  # verify that the wave vector and sigma are perpendicular
  # verify that the wave vector and sigma are perpendicular
  kk = np.sum(ktx*sxm + kty*sym + ktz*szm)
  print 'Orthogonality of k and sigma (divergence in wave space):'
  print kk
  
  # get the modes   
  km = wn
  
  # now create an interpolant for the spectrum. this is needed for
  # experimentally-specified spectra
  espec = especf(km)
  espec = espec.clip(0.0)
  
  # generate turbulence at cell centers
  um = sqrt(espec*dkn)
  u_ = zeros([nx,ny,nz])
  v_ = zeros([nx,ny,nz])
  w_ = zeros([nx,ny,nz])

  xc = dx/2.0 + arange(0,nx)*dx  
  yc = dy/2.0 + arange(0,ny)*dy  
  zc = dz/2.0 + arange(0,nz)*dz
  
  for k in range(0,nz):
    for j in range(0,ny):
      for i in range(0,nx):
        #for every grid point (i,j,k) do the fourier summation 
        arg = kx*xc[i] + ky*yc[j] + kz*zc[k] - psi
        bmx = 2.0*um*cos(arg - kx*dx/2.0)
        bmy = 2.0*um*cos(arg - ky*dy/2.0)        
        bmz = 2.0*um*cos(arg - kz*dz/2.0)                
        u_[i,j,k] = np.sum(bmx*sxm)
        v_[i,j,k] = np.sum(bmy*sym)
        w_[i,j,k] = np.sum(bmz*szm)          
  
        
  print 'done. I am awesome!'
  return u_, v_, w_