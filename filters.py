# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:28:57 2014

@author: Tony Saad
"""
from tkespec import compute_tke_spectrum
import scipy.io
import numpy as np
from numpy import sqrt, zeros, conj, pi, arange, ones, convolve,sin

import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn

def spectralcutoff(u, kappa, cx, cy, cz):
  nx = len(u[:,0,0])
  ny = len(u[0,:,0])
  nz = len(u[0,0,:])  
  nt= nx*ny*nz  
  uh = fftn(u)/nt  
  for i in range(0,nx):
    for j in range (0,ny):
      for k in range (0,nz):      
        rk = sqrt(cx*i*i + cy*j*j + cz*k*k)
        #rk = int(np.round(rk))
        if(rk >= kappa):
          uh[i,j,k] = 0.0
  ureal = ifftn(uh)*nt
  return ureal.real

#def spectralcutoff(u, kappa):
#  nx = len(u[:,0,0])
#  ny = len(u[0,:,0])
#  nz = len(u[0,0,:])  
#  nt= nx*ny*nz  
#  uh = fftn(u)/nt  
#  for i in range(0,nx):
#    for j in range (0,ny):
#      for k in range (0,nz):      
#        rk = sqrt(i*i + j*j + k*k)
#        #rk = int(np.round(rk))
#        if(rk >= kappa):
#          uh[i,j,k] = 0.0
#  ureal = ifftn(uh)*nt
#  return ureal.real

def boxfilter(u):
  nx = len(u[:,0,0])
  ny = len(u[0,:,0])
  nz = len(u[0,0,:])
  # filter the data
  ut = np.empty([nx+2,ny+2,nz+2])
  uf = np.zeros([nx,ny,nz])
  ut[1:nx+1,1:ny+1,1:nz+1] = u
  # now make it periodic
  ut[0,:,:] = ut[nx,:,:]
  ut[nx+1,:,:] = ut[1,:,:]
  ut[:,0,:] = ut[:,ny,:]
  ut[:,ny+1,:] = ut[:,1,:]
  ut[:,:,0] = ut[:,:,nz]
  ut[:,:,nz+1] = ut[:,:,1]    

  for i in range(0,nx):
    for j in range (0,ny):
      for k in range (0,nz):      
        uf[i,j,k] = 1.0/27.0*(  ut[i,j,k] \
                  + ut[i-1,j,k] + ut[i+1,j,k] \
                   + ut[i,j+1,k] + ut[i,j-1,k] \
                   + ut[i,j,k+1] + ut[i,j,k-1]  \
                   + ut[i+1,j+1,k] + ut[i+1,j-1,k] \
                   + ut[i-1,j+1,k] + ut[i-1,j-1,k] \
                   + ut[i+1,j,k+1] + ut[i+1,j,k-1] \
                   + ut[i-1,j,k+1] + ut[i-1,j,k-1] \
                   + ut[i,j+1,k+1] + ut[i,j+1,k-1] \
                   + ut[i,j-1,k+1] + ut[i,j-1,k-1] \
                   + ut[i+1,j+1,k+1] + ut[i+1,j+1,k-1] \
                   + ut[i+1,j-1,k+1] + ut[i+1,j-1,k-1] \
                   + ut[i-1,j+1,k+1] + ut[i-1,j+1,k-1] \
                   + ut[i-1,j-1,k+1] + ut[i-1,j-1,k-1])        
  return uf

#mat = scipy.io.loadmat('uvw_32.mat')
#u = mat['U']
#v = mat['V']
#w = mat['W']
#
#lx=ly=lz=1.0
#
## verify that the generated velocities fit the spectrum
#knyquist, wavenumbers, tkespec = compute_tke_spectrum(u,v,w,lx,ly,lz, True)
#
#q, ((p1,p2),(p3,p4)) = plt.subplots(2,2)
#
#p1.loglog(wavenumbers, tkespec, 'bo-')
#p1.axvline(x=knyquist, linestyle='--', color='black')
#p1.set_title('Spectrum of generated turbulence')
#p1.grid()
#
#nx = len(u[:,0,0])
#ny = len(v[0,:,0])
#nz = len(w[0,0,:])
#
##uf1 = sectralcutoff(u,30)
##vf1 = sectralcutoff(v,30)
##wf1 = sectralcutoff(w,30)
#
#uf0 = boxfilter(u)
#vf0 = boxfilter(v)
#wf0 = boxfilter(w)
#
#uf1 = boxfilter(uf0)
#vf1 = boxfilter(vf0)
#wf1 = boxfilter(wf0)
#
## verify that the generated velocities fit the spectrum
#knyquist, wavenumbers, tkespec = compute_tke_spectrum(uf1,vf0,wf0,lx,ly,lz, True)
#p1.loglog(wavenumbers, tkespec, 'ro-')
#p2.loglog(wavenumbers, tkespec, 'ro-')
#p2.axvline(x=knyquist, linestyle='--', color='black')
#p2.set_title('Spectrum of generated turbulence')
#p2.grid()
#
#
#p3.matshow(u[:,10,:])
#p4.matshow(uf1[:,10,:])
#
#plt.draw()