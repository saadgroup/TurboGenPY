# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:08:01 2014

@author: Tony Saad
"""
#!/usr/bin/env python
from scipy import interpolate
import numpy as np
import gzip
from numpy import sin, cos, sqrt, ones, zeros, pi, arange
from numpy import linalg as LA
from tkespec import compute_tke_spectrum
import matplotlib.pyplot as plt

#load an experimental specturm. Alternatively, specify it via a function call
cbcspec = np.loadtxt('cbc_spectrum.txt')
kcbc=cbcspec[:,0]*100
ecbc=cbcspec[:,1]*1e-6

#USER INPUT
#set the number of modes.
nmodes =100

# file name base
fNameBase = 'cbc32_uvw'

# write to file
enableIO = True

# compute the mean of the fluctuations
computeMean = True

#Lx = 9*2*pi/100; % domain size in the x direction
#Ly = 9*2*pi/100; % domain size in the y direction
#Lz = 9*2*pi/100; % domain size in the z direction
# input domain size in the x, y, and z directions
Lx = 2*pi/15
Ly = 2*pi/15
Lz = 2*pi/15

# input number of cells (cell centered control volumes)
nx = 32         # number of cells in the x direction
ny = 32         # number of cells in the y direction
nz = 32         # number of cells in the z direction

# enter the smallest wavenumber represented by this spectrum
wn1 = 15 #determined here from cbc spectrum properties

## grid generation
# generate cell centered x-grid
dx = Lx/nx
xc = dx/2.0 + arange(0,nx)*dx

# generate cell centered y-grid
dy = Ly/ny
yc = dy/2 + arange(0,ny)*dy

# generate cell centered z-grid
dz = Lz/nz
zc = dz/2 + arange(0,nz)*dz # cell centered coordinates

## START THE FUN!
#wn = zeros(nmodes,1); % wave number array

# compute random angles
psi   = 2.0*pi*np.random.rand(nmodes);
phi   = 2.0*pi*np.random.rand(nmodes);
alfa  = 2.0*pi*np.random.rand(nmodes);
theta = pi*np.random.rand(nmodes);

# highest wave number that can be represented on this grid (nyquist limit)
wnn = max(np.pi/dx, max(np.pi/dy, np.pi/dz));
print 'I will generate data up to wave number: ', wnn

# wavenumber step
dk = (wnn-wn1)/nmodes

# wavenumber at faces
wnf = wn1 + arange(0,nmodes+1)*dk 

# wavenumber at cell centers
wn = wn1 + 0.5*dk + arange(0,nmodes)*dk
#wn = wn'
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

# verify that the wave vector and sigma are perpendicular
kk = kx*sxm + ky*sym + kz*szm;
print 'Orthogonality of k and sigma (divergence in wave space):'
print LA.norm(kk)

# get the modes   
km = sqrt(kx*kx + ky*ky + kz*kz)

# now create an interpolant for the spectrum. this is needed for
# experimentally-specified spectra
f = interpolate.interp1d(kcbc, ecbc,'cubic')
espec = f(km)   # use interpolation function returned by `interp1d`

f, ((p1,p2),(p3,p4)) = plt.subplots(2,2)

p1.plot(kcbc, ecbc, 'o', km, espec, '-')
p1.set_title('Interpolated Spectrum')

# generate turbulence at cell centers
um = 2*sqrt(espec*dkn)
u_ = zeros((nx,ny,nz))
v_ = zeros((nx,ny,nz))
w_ = zeros((nx,ny,nz))

for k in range(0,nz):
  for j in range(0,ny):
    for i in range(0,nx):
    #for every grid point (i,j,k) do the fourier summation 
      arg = kx*xc[i] + ky*yc[j] + kz*zc[k] - psi
      bm = um*cos(arg)
      u_[i,j,k] = sum(bm*sxm)
      v_[i,j,k] = sum(bm*sym) 
      w_[i,j,k] = sum(bm*szm)

### CONVERT TO STAGGERED GRID - IF NECESSARY
## This is not needed anymore since the cfd code (wasatch for example) will
## will do the proper adjustment at the boundaries. So simply store the 
## cell-centered generated velocities on the staggered volumes.
#u = zeros([nx+1,ny,nz])
#v = zeros([nx,ny+1,nz])
#w = zeros([nx,ny,nz+1])
#
## set periodic condition
#for i in range(0,nx-1):
#  u[i+1,:,:]= 0.5*(u_[i+1,:,:] + u_[i,:,:])
#u[0,:,:] = 0.5*(u_[0,:,:] + u_[nx-1,:,:])
#u[nx,:,:] = 0.5*(u_[0,:,:] + u_[nx-1,:,:])
#
## set periodic condition
#for j in range(0,ny-1):
#  v[:,j+1,:] = 0.5*(v_[:,j+1,:] + v_[:,j,:])   
#v[:,0,:] = 0.5*(v_[:,0,:] + v_[:,ny-1,:])
#v[:,ny,:] = 0.5*(v_[:,0,:] + v_[:,ny-1,:])
#
## set periodic condition
#for k in range(0,nz-1):
#  w[:,:,k+1] = 0.5*(w_[:,:,k+1] + w_[:,:,k])   
#w[:,:,0] = 0.5*(w_[:,:,0] + w_[:,:,nz-1])
#w[:,:,nz] = 0.5*(w_[:,:,0] + w_[:,:,nz-1])

# compute mean velocities
if computeMean:
  umean = np.mean(u_)
  vmean = np.mean(v_)
  wmean = np.mean(w_)  
  print 'umean = ', umean
  print 'vmean = ', vmean
  print 'wmean = ', wmean
  
# verify that the generated velocities fit the spectrum
wavenumbers, tkespec = compute_tke_spectrum(u_,v_,w_,Lx,Ly,Lz, True)
p2.loglog(kcbc, ecbc, '-', wavenumbers, tkespec, 'ro-')
p2.set_title('Spectrum of generated turbulence')
plt.show()

# contour plot
X, Y = np.meshgrid(xc, yc)
p3.contour(X, Y, u_[:,:,1])
p3.set_title('Contour lines of u velocity')

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
  fw.close()
  #end if enable IO
  
print 'done. I am awesome!'