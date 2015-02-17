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

def generate_isotropic_turbulence(lx,ly,lz,nx,ny,nz,nmodes,wn1,especf,computeMean,enableIO):
  ## grid generation
  # generate cell centered x-grid
  dx = lx/nx
  xc = dx/2.0 + arange(0,nx)*dx
  
  # generate cell centered y-grid
  dy = ly/ny
  yc = dy/2 + arange(0,ny)*dy
  
  # generate cell centered z-grid
  dz = lz/nz
  zc = dz/2 + arange(0,nz)*dz # cell centered coordinates
  
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
  
  # sigma is the unit direction which gives the direction of the synthetic
  # velocity field
  sxm = cos(phi)*cos(theta)*cos(alfa) - sin(phi)*sin(alfa)
  sym = sin(phi)*cos(theta)*cos(alfa) + cos(phi)*sin(alfa)
  szm = -sin(theta)*cos(alfa)   

#  sxm = cos(theta)*cos(alfa)
#  sym = cos(theta)*sin(alfa)
#  szm = -sin(theta)*cos(phi)*cos(alfa)-sin(theta)*sin(phi)*sin(alfa)
  
  # verify that the wave vector and sigma are perpendicular
  kk = kx*sxm + ky*sym + kz*szm;
  print 'Orthogonality of k and sigma (divergence in wave space):'
  print LA.norm(kk)
  
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
  
  for k in range(0,nz):
    for j in range(0,ny):
      for i in range(0,nx):
        #for every grid point (i,j,k) do the fourier summation 
        arg = kx*xc[i] + ky*yc[j] + kz*zc[k] - psi
        bm = 2.0*um*cos(arg)
        u_[i,j,k] = np.sum(bm*sxm) 
        v_[i,j,k] = np.sum(bm*sym)
        w_[i,j,k] = np.sum(bm*szm)

#  # try to enforce divergence-free conditions
#  count = 0
#  for k in range(0,nz-1):
#    for j in range(0,ny-1):
#      for i in range(0,nx-1):
#        src = (u_[i+1,j,k] - u_[i,j,k])/dx + (v_[i,j+1,k] - v_[i,j,k])/dy + (w_[i,j,k+1] - w_[i,j,k])/dz
#        if(src > 1e-5):
#          count += 1
#          src = (u_[i+1,j,k] - u_[i,j,k])/dx + (v_[i,j+1,k] - v_[i,j,k])/dy
#          #oldw = w_[i,j,k+1]
#          w_[i,j,k+1] = w_[i,j,k] - dz*src
#          #fac = w_[i,j,k+1]/oldw
#          #print 'factor = ', fac
#  print 'cells with divergence: ', count

  count = 0
  for k in range(0,nz-1):
    for j in range(0,ny-1):
      for i in range(0,nx-1):
        src = (u_[i+1,j,k] - u_[i,j,k])/dx + (v_[i,j+1,k] - v_[i,j,k])/dy + (w_[i,j,k+1] - w_[i,j,k])/dz
        if(src > 1e-2):
          count += 1
  print 'cells with divergence: ', count
  
  # compute mean velocities
  if computeMean:
    umean = np.mean(u_)
    vmean = np.mean(v_)
    wmean = np.mean(w_)  
    print 'umean = ', umean
    print 'vmean = ', vmean
    print 'wmean = ', wmean
      
  #write to disk
  if enableIO:
    print 'Writing to disk. This may take a while...'
    fu = gzip.open('u.txt.gz', 'w')
    for k in range(0,nz):
      for j in range(0,ny):
        for i in range(0,nx):
          x = i*dx
          y = j*dy + dy/2.0
          z = k*dz + dz/2.0
          uu = u_[i,j,k]
          fu.write('%.16f %.16f %.16f %.16f \n' % (x,y,z,uu))
          #fu.write('%d %d %d %.16f \n' % (i,j,k,uu))
    fu.close()
    
    fv = gzip.open('v.txt.gz', 'w')
    for k in range(0,nz):
      for j in range(0,ny):
        for i in range(0,nx):
          x = i*dx + dx/2.0
          y = j*dy
          z = k*dz + dz/2.0
          vv = v_[i,j,k]
          fv.write('%.16f %.16f %.16f %.16f \n' % (x,y,z,vv))      
          #fv.write('%d %d %d %.16f \n' % (i,j,k,vv))
    fv.close()
    
    fw = gzip.open('w.txt.gz', 'w')
    for k in range(0,nz):
      for j in range(0,ny):
        for i in range(0,nx):
          x = i*dx + dx/2.0
          y = j*dy + dy/2.0
          z = k*dz
          ww = w_[i,j,k]
          fw.write('%.16f %.16f %.16f %.16f \n' % (x,y,z,ww))      
          #fw.write('%d %d %d %.16f \n' % (i,j,k,ww))          
    fw.close()
    #end if enable IO
    
  print 'done. I am awesome!'
  return u_, v_, w_