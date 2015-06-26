#
#  IsoTurbGen.h
#
#  The MIT License (MIT)
#
#  Copyright (c) 2015, Tony Saad. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#

# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:31:54 2014

@author: tsaad
"""
import numpy as np
from numpy import sin, cos, sqrt, ones, zeros, pi, arange

def generate_isotropic_turbulence(lx,ly,lz,nx,ny,nz,nmodes,wn1,especf):
  """
  Given an energy spectrum, this function computes a discrete, staggered, three 
  dimensional velocity field in a box whose energy spectrum corresponds to the input energy 
  spectrum up to the Nyquist limit dictated by the grid

  This function returns u, v, w as the axial, transverse, and azimuthal velocities.
  
  Parameters:
  -----------  
  lx: float
    The domain size in the x-direction.
  ly: float
    The domain size in the y-direction.
  lz: float
    The domain size in the z-direction.  
  nx: integer
    The number of grid points in the x-direction.
  ny: integer
    The number of grid points in the y-direction.
  nz: integer
    The number of grid points in the z-direction.
  wn1: float
    Smallest wavenumber. Typically dictated by spectrum or domain size.
  espec: functor
    A callback function representing the energy spectrum.
  """

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

  # Enforce Mass Conservation
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
  kk = np.sum(ktx*sxm + kty*sym + ktz*szm)
  print 'Orthogonality of k and sigma (divergence in wave space):'
  print kk
  
  # get the modes   
  km = wn
  
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