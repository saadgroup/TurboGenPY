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

def generate_isotropic_turbulence(lx,ly,lz,nx,ny,nz,nmodes,wn1,especf,cellCentered):
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
  if(not cellCentered):
    kxstag = np.sin(kx*dx/2.0)/(dx)
    kystag = np.sin(ky*dy/2.0)/(dy)
    kzstag = np.sin(kz*dz/2.0)/(dz)  
    kstagmag = sqrt(kxstag*kxstag + kystag*kystag + kzstag*kzstag)
    # angles for this vector
    theta = np.arccos(kzstag/kstagmag)
    phi = np.arctan2(kystag,kxstag)

  # sigma is the unit direction which gives the direction of the synthetic velocity field
  sxm = cos(phi)*cos(theta)*cos(alfa) - sin(phi)*sin(alfa)
  sym = sin(phi)*cos(theta)*cos(alfa) + cos(phi)*sin(alfa)
  szm = -sin(theta)*cos(alfa)   
  
  # sigma is the unit direction which gives the direction of the synthetic
  # velocity field
#  sxm = cos(phi)*cos(theta)*cos(alfa) - sin(phi)*sin(alfa)
#  sym = sin(phi)*cos(theta)*cos(alfa) + cos(phi)*sin(alfa)
#  szm = -sin(theta)*cos(alfa)   

#  sxm = cos(theta)*cos(alfa)
#  sym = cos(theta)*sin(alfa)
#  szm = -sin(theta)*cos(phi)*cos(alfa)-sin(theta)*sin(phi)*sin(alfa)
  
  # verify that the wave vector and sigma are perpendicular
  if (cellCentered):
    kk = np.sum(kx*sxm + ky*sym + kz*szm)
  else:
    kk = np.sum(kxstag*sxm + kystag*sym + kzstag*szm)

  print 'Orthogonality of k and sigma (divergence in wave space):'
  print kk
  
  # get the modes   
  km = sqrt(kx*kx + ky*ky + kz*kz)
  
  # now create an interpolant for the spectrum. this is needed for
  # experimentally-specified spectra
  espec = especf(km)
  espec = espec.clip(0.0)
  
  # generate turbulence at cell centers
  um = sqrt(espec*dkn)
  u_ = zeros([nx,ny,nz])
  v_ = zeros([nx,ny,nz])
  w_ = zeros([nx,ny,nz])

  if(cellCentered):    
    # generate cell centered xyz-grid
    xc = dx/2.0 + arange(0,nx)*dx  
    yc = dy/2.0 + arange(0,ny)*dy  
    zc = dz/2.0 + arange(0,nz)*dz

    for k in range(0,nz):
      for j in range(0,ny):
        for i in range(0,nx):
          #for every grid point (i,j,k) do the fourier summation 
          arg = kx*xc[i] + ky*yc[j] + kz*zc[k] - psi
          bm = 2.0*um*cos(arg)        
          u_[i,j,k] = np.sum(bm*sxm)
          v_[i,j,k] = np.sum(bm*sym)
          w_[i,j,k] = np.sum(bm*szm)    
  else:
    xxvol =          arange(0,nx)*dx
    yxvol = dy/2.0 + arange(0,ny)*dy
    zxvol = dz/2.0 + arange(0,nz)*dz
    
    xyvol = dx/2.0 + arange(0,nx)*dx
    yyvol =          arange(0,ny)*dy
    zyvol = dz/2.0 + arange(0,nz)*dz
  
    xzvol = dx/2.0 + arange(0,nx)*dx
    yzvol = dy/2.0 + arange(0,ny)*dy
    zzvol =          arange(0,nz)*dz
    
    for k in range(0,nz):
      for j in range(0,ny):
        for i in range(0,nx):
          #for every grid point (i,j,k) do the fourier summation 
          argx = kx*xxvol[i] + ky*yxvol[j] + kz*zxvol[k] - psi
          bmx = 2.0*um*cos(argx)
          argy = kx*xyvol[i] + ky*yyvol[j] + kz*zyvol[k] - psi
          bmy = 2.0*um*cos(argy)        
          argz = kx*xzvol[i] + ky*yzvol[j] + kz*zzvol[k] - psi        
          bmz = 2.0*um*cos(argz)                
          u_[i,j,k] = np.sum(bmx*sxm)
          v_[i,j,k] = np.sum(bmy*sym)
          w_[i,j,k] = np.sum(bmz*szm)          
  
        
  print 'done. I am awesome!'
  return u_, v_, w_