#
#  isoturb.py
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


def generate_isotropic_turbulence(lx, ly, lz, nx, ny, nz, nmodes, wn1, especf):
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
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    # START THE FUN!

    # compute random angles
    phi = 2.0 * pi * np.random.uniform(0.0, 1.0, nmodes)
    nu = np.random.uniform(0.0, 1.0, nmodes)
    theta = np.arccos(2.0 * nu - 1.0)
    psi = np.random.uniform(0, pi, nmodes)

    # highest wave number that can be represented on this grid (nyquist limit)
    wnn = max(np.pi / dx, max(np.pi / dy, np.pi / dz))
    print('I will generate data up to wave number: ', wnn)

    # wavenumber step
    dk = (wnn - wn1) / nmodes

    # wavenumber at cell centers
    wn = wn1 + 0.5 * dk + arange(0, nmodes) * dk
    print('wn = ', wn)
    dkn = ones(nmodes) * dk

    #   wavenumber vector from random angles
    kx = sin(theta) * cos(phi) * wn
    ky = sin(theta) * sin(phi) * wn
    kz = cos(theta) * wn

    # create divergence vector
    ktx = np.sin(kx * dx / 2.0) / dx
    kty = np.sin(ky * dy / 2.0) / dy
    ktz = np.sin(kz * dz / 2.0) / dz

    # Enforce Mass Conservation
    phi1 = 2.0 * pi * np.random.uniform(0.0, 1.0, nmodes)
    nu1 = np.random.uniform(0.0, 1.0, nmodes)
    theta1 = np.arccos(2.0 * nu1 - 1.0)
    zetax = sin(theta1) * cos(phi1)
    zetay = sin(theta1) * sin(phi1)
    zetaz = cos(theta1)
    sxm = zetay * ktz - zetaz * kty
    sym = -(zetax * ktz - zetaz * ktx)
    szm = zetax * kty - zetay * ktx
    smag = sqrt(sxm * sxm + sym * sym + szm * szm)
    sxm = sxm / smag
    sym = sym / smag
    szm = szm / smag

    # verify that the wave vector and sigma are perpendicular
    kk = np.sum(ktx * sxm + kty * sym + ktz * szm)
    print('Orthogonality of k and sigma (divergence in wave space):', kk)

    # get the modes
    km = wn

    espec = especf(km)
#     espec = espec.clip(0.0)

    # generate turbulence at cell centers
    um = sqrt(espec * dkn)
    u_ = zeros([nx, ny, nz])
    v_ = zeros([nx, ny, nz])
    w_ = zeros([nx, ny, nz])

    xc = dx / 2.0 + arange(0, nx) * dx
    yc = dy / 2.0 + arange(0, ny) * dy
    zc = dz / 2.0 + arange(0, nz) * dz

    for k in range(0, nz):
        for j in range(0, ny):
            for i in range(0, nx):
                # for every grid point (i,j,k) do the fourier summation
                arg = kx * xc[i] + ky * yc[j] + kz * zc[k] - psi
                bmx = 2.0 * um * cos(arg - kx * dx / 2.0)
                bmy = 2.0 * um * cos(arg - ky * dy / 2.0)
                bmz = 2.0 * um * cos(arg - kz * dz / 2.0)
                u_[i, j, k] = np.sum(bmx * sxm)
                v_[i, j, k] = np.sum(bmy * sym)
                w_[i, j, k] = np.sum(bmz * szm)

    print('done generating turbulence.')
    return u_, v_, w_

import numpy as np

def generate_periodic_scalar_turbulence(lx, ly, lz, nx, ny, nz, nmodes,wn1, especf):
    """
    Generates a periodic 3D scalar field using discrete Fourier modes.
    """
    # 1. Setup Grid (Cell-Centered)
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    xc = np.linspace(dx/2, lx - dx/2, nx)
    yc = np.linspace(dy/2, ly - dy/2, ny)
    zc = np.linspace(dz/2, lz - dz/2, nz)
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')

    # 2. Generate Discrete Wavevectors
    # We pick random integer indices to ensure periodicity
    # max_m defines the highest frequency harmonic allowed
    max_m = max(nx, ny, nz) // 2 
    
    # Randomly sample integer modes (excluding 0,0,0)
    mx = np.random.randint(-max_m, max_m, nmodes)
    my = np.random.randint(-max_m, max_m, nmodes)
    mz = np.random.randint(-max_m, max_m, nmodes)
    
    # Frequency scaling for periodicity: k = m * (2*pi / L)
    kx = mx * (2.0 * np.pi / lx)
    ky = my * (2.0 * np.pi / ly)
    kz = mz * (2.0 * np.pi / lz)
    
    # Wavenumber magnitude
    kn = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Handle the k=0 case to avoid division by zero in spectrum
    kn[kn == 0] = 1e-10 

    # 3. Calculate Amplitudes and Phases
    # In discrete mode, energy is distributed per mode. 
    # Since we are sampling randomly, we use an approximation for dk.
    # A more robust way is to use the shell volume, but for nmodes >> 1:
    dk = np.max(kn) / nmodes 
    espec = especf(kn).clip(0.0)
    
    # um represents the amplitude of the fluctuation for this mode
    um = np.sqrt(espec * dk)
    phases = np.random.uniform(0, 2.0 * np.pi, nmodes)

    # 4. Vectorized Summation (The "Awesome" part)
    scalar_field = np.zeros((nx, ny, nz))
    
    print(f"Generating {nmodes} periodic modes on {nx}x{ny}x{nz} grid...")
    for m in range(nmodes):
        # arg = k_vec dot x_vec - phase
        arg = kx[m]*X + ky[m]*Y + kz[m]*Z - phases[m]
        scalar_field += 2.0 * um[m] * np.cos(arg)
        
    print("Done. Edge values now match perfectly!")
    return scalar_field
    
def generate_scalar_isotropic_turbulence(lx, ly, lz, nx, ny, nz, nmodes, wn1, especf):
    """
    Given an energy spectrum, this function computes a discrete, cell centered, three
    dimensional scalar field in a box whose spectrum corresponds to the input 
    spectrum up to the Nyquist limit dictated by the grid

    This function returns phi as a 3D scalar field.

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
      A callback function representing the spectrum.
    """

    # generate cell centered x-grid
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    # START THE FUN!

    # compute random angles
    phi = 2.0 * pi * np.random.uniform(0.0, 1.0, nmodes)
    nu = np.random.uniform(0.0, 1.0, nmodes);
    theta = np.arccos(2.0 * nu - 1.0);
    psi = np.random.uniform(-pi / 2.0, pi / 2.0, nmodes)

    # highest wave number that can be represented on this grid (nyquist limit)
    wnn = max(np.pi / dx, max(np.pi / dy, np.pi / dz))
    print('I will generate data up to wave number: ', wnn)

    # wavenumber step
    dk = (wnn - wn1) / nmodes

    # wavenumber at cell centers
    wn = wn1 + 0.5 * dk + arange(0, nmodes) * dk

    dkn = ones(nmodes) * dk

    #   wavenumber vector from random angles
    kx = sin(theta) * cos(phi) * wn
    ky = sin(theta) * sin(phi) * wn
    kz = cos(theta) * wn

    # get the modes
    km = wn

    espec = especf(km)
    espec = espec.clip(0.0)

    # generate turbulence at cell centers
    um = sqrt(espec * dkn)
    scalar_ = zeros([nx, ny, nz])

    xc = dx / 2.0 + arange(0, nx) * dx
    yc = dy / 2.0 + arange(0, ny) * dy
    zc = dz / 2.0 + arange(0, nz) * dz

	# Create 3D coordinate grids
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')	
	# Initialize scalar field
    scalar_ = np.zeros((nx, ny, nz))
	
	# Sum the modes one by one (vectorized over the whole 3D grid at once)
    for m in range(nmodes):
        arg = kx[m] * X + ky[m] * Y + kz[m] * Z - psi[m]
        scalar_ += 2.0 * um[m] * np.cos(arg)

#     for k in range(0, nz):
#         for j in range(0, ny):
#             for i in range(0, nx):
#                 # for every grid point (i,j,k) do the fourier summation
#                 arg = kx * xc[i] + ky * yc[j] + kz * zc[k] - psi
#                 bm = 2.0 * um * cos(arg)
#                 scalar_[i, j, k] = np.sum(bm)
# 
#                 print('done. I am awesome!')
    return scalar_

def generate_isotropic_turbulence_2d(lx, ly, nx, ny, nmodes, wn1, especf):
    """
    This is the 2D version of the isotropic turbulence generator.
    Given an energy spectrum, this function computes a discrete, staggered, three
    dimensional velocity field in a box whose energy spectrum corresponds to the input energy
    spectrum up to the Nyquist limit dictated by the grid

    This function returns u, v as the axial and transverse velocities.

    Parameters:
    -----------
    lx: float
      The domain size in the x-direction.
    ly: float
      The domain size in the y-direction.
    nx: integer
      The number of grid points in the x-direction.
    ny: integer
      The number of grid points in the y-direction.
    wn1: float
      Smallest wavenumber. Typically dictated by spectrum or domain size.
    espec: functor
      A callback function representing the energy spectrum.
    """

    # generate cell centered x-grid
    dx = lx / nx
    dy = ly / ny

    # START THE FUN!

    # compute random angles
    psi = np.random.uniform(-pi / 2.0, pi / 2.0, nmodes)

    # highest wave number that can be represented on this grid (nyquist limit)
    wnn = max(np.pi / dx, np.pi / dy)
    print('I will generate data up to wave number: ', wnn)

    # wavenumber step
    dk = (wnn - wn1) / nmodes

    # wavenumber at cell centers
    wn = wn1 + 0.5 * dk + arange(0, nmodes) * dk

    dkn = ones(nmodes) * dk

    #   wavenumber vector from random angles
    theta = np.random.uniform(0.0,2.0*np.pi,nmodes)
    kx = cos(theta) * wn
    ky = sin(theta) * wn

    # create divergence vector
    ktx = np.sin(kx * dx / 2.0) / dx
    kty = np.sin(ky * dy / 2.0) / dy

    # Enforce Mass Conservation
    sxm = -kty
    sym = ktx
    
    smag = sqrt(sxm * sxm + sym * sym)
    sxm = sxm / smag
    sym = sym / smag

    # verify that the wave vector and sigma are perpendicular
    kk = np.sum(ktx * sxm + kty * sym)
    print('Orthogonality of k and sigma (divergence in wave space):', kk)

    # get the modes
    km = wn

    espec = especf(km)
    espec = espec.clip(0.0)

    # generate turbulence at cell centers
    um = sqrt(espec * dkn)
    u_ = zeros([nx, ny])
    v_ = zeros([nx, ny])

    xc = dx / 2.0 + arange(0, nx) * dx
    yc = dy / 2.0 + arange(0, ny) * dy
    for j in range(0, ny):
    	for i in range(0, nx):
    		# for every grid point (i,j,k) do the fourier summation
    		arg = kx * xc[i] + ky * yc[j]  - psi
    		bmx = 2.0 * um * cos(arg - kx * dx / 2.0)
    		bmy = 2.0 * um * cos(arg - ky * dy / 2.0)
    		u_[i, j] = np.sum(bmx * sxm)
    		v_[i, j] = np.sum(bmy * sym)

    print('done generating turbulence.')
    return u_, v_