# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:31:54 2014

@author: tsaad
"""
import multiprocessing as mp
import numpy as np
import time
from numpy import sin, cos, sqrt, ones, zeros, pi, arange
from numpy import linalg as LA

#------------------------------------------------------------------------------

def compute_turbulence(nthread,dx,dy,dz,psi,um,kx,ky,kz,sxm,sym,szm,nx,ny,nz,nxAll, nyAll, nzAll, cellCentered,ip,jp,kp, q):
  print 'Generating turbulence on thread:', nthread
  t0 = time.time()
  u_ = zeros([nx,ny,nz])
  v_ = zeros([nx,ny,nz])
  w_ = zeros([nx,ny,nz])

  xl = (ip-1)*nx
  xh = ip*nx
  yl = (jp-1)*ny
  yh = jp*ny
  zl = (kp - 1)*nz
  zh = kp * nz  
  
  if(cellCentered):    
    # generate cell centered xyz-grid
    xc = dx/2.0 + arange(xl,xh)*dx
    yc = dy/2.0 + arange(yl,yh)*dy
    zc = dz/2.0 + arange(zl,zh)*dz # cell centered coordinates

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
    xxvol =          arange(xl,xh)*dx
    yxvol = dy/2.0 + arange(yl,yh)*dy
    zxvol = dz/2.0 + arange(zl,zh)*dz
    
    xyvol = dx/2.0 + arange(xl,xh)*dx
    yyvol =          arange(yl,yh)*dy
    zyvol = dz/2.0 + arange(zl,zh)*dz
  
    xzvol = dx/2.0 + arange(xl,xh)*dx
    yzvol = dy/2.0 + arange(yl,yh)*dy
    zzvol =          arange(zl,zh)*dz
    
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


  t1 = time.time()
  print 'Thread ', nthread, ' done generating turbulence in ', t1 - t0, 's'
  q.put((ip,jp,kp,u_,v_,w_))
  return ip, jp, kp, u_, v_, w_
#------------------------------------------------------------------------------

def generate_isotropic_turbulence(patches, lx,ly,lz,nx,ny,nz,nmodes,wn1,especf, cellCentered):
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
      
  #  must use Manager queue here, or will not work
  nxthreads = patches[0];
  nythreads = patches[1];
  nzthreads = patches[2]; 
  nxt = nx/nxthreads;
  nyt = nx/nythreads;
  nzt = nx/nzthreads;
  
  manager = mp.Manager()
  mq = manager.Queue()    
  pool = mp.Pool(mp.cpu_count()*2) #assume 2 threads per core

  #fire off workers
  jobs = []
  nthread = 0
  for k in range(1,nzthreads+1):
    for j in range(1,nythreads+1):
      for i in range(1,nxthreads+1):
        nthread= nthread+1
        job = pool.apply_async(compute_turbulence, (nthread, dx,dy,dz,psi,um,kx,ky,kz,sxm,sym,szm,nxt,nyt,nzt, nx,ny,nz, cellCentered,i,j,k,mq))
        jobs.append(job)
        
  # collect results from the workers through the pool result queue
  print 'now collecting results from individual threads...'
  uarrays = []
  varrays = []
  warrays = []
  patches = []
  for job in jobs: 
    i,j,k,u,v,w  = job.get()
    uarrays.append(u)
    varrays.append(v)
    warrays.append(w)
    patches.append([i,j,k])
  del u, v, w
  
  pool.terminate()
  pool.close()

  #combine the arrays computed from threads into large arrays
  print 'now combining velocity fields generated by the individual threads...'
  uall=zeros([nx,ny,nz])
  vall=zeros([nx,ny,nz])
  wall=zeros([nx,ny,nz])
  nthread = 0
  for k in range(1,nzthreads+1):
    for j in range(1,nythreads+1):
      for i in range(1,nxthreads+1):
        uall[(i-1)*nxt:i*nxt,(j-1)*nyt:j*nyt,(k-1)*nzt:k*nzt] = uarrays[nthread]
        vall[(i-1)*nxt:i*nxt,(j-1)*nyt:j*nyt,(k-1)*nzt:k*nzt] = varrays[nthread]
        wall[(i-1)*nxt:i*nxt,(j-1)*nyt:j*nyt,(k-1)*nzt:k*nzt] = warrays[nthread]
        nthread=nthread+1
  
  return uall,vall,wall