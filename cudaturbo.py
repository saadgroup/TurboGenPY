#
#  cudaturbo.py
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
"""
@authors: Tony Saad and Austin Richards
"""

import math
from numba import cuda
import numpy as np
from numpy import pi, ones, zeros, sin, cos, sqrt, arange


@cuda.jit
def turbo_kernel(kx, ky, kz, xc, yc, zc, psi, um, sxm, sym, szm, dx, dy, dz, u_, v_, w_):
    """
    This is the cuda kernel for the turbulence generator. Our approach is to assign one thread per spatial grid point.
    Then, the job of each thread is to compute the Fourier summation @ that grid point.
    :param kx: Fourier modes in the x direction
    :param ky: Fourier modes in the y direction
    :param kz: Fourier modes in the z direction
    :param xc: x coordinate of cell centers
    :param yc: y coordinate of cell centers
    :param zc: z coordinate of cell centers
    :param psi: Wave component
    :param um:  Fourier velocity magnitude
    :param sxm: Auxiliary vector (sigma in the paper)
    :param sym: Auxiliary vector (sigma in the paper)
    :param szm: Auxiliary vector (sigma in the paper)
    :param dx: Grid spacing in the x direction
    :param dy: Grid spacing in the y direction
    :param dz: Grid spacing in the z direction
    :param u_: x velocity component of the generated turbulence
    :param v_: y velocity component of the generated turbulence
    :param w_: z velocity component of the generated turbulence
    :return: u, v, w of the generated turbulent vector field
    """
    # calculate thread location for 3D array
    i, j, k = cuda.grid(3)

    # Each thread is assigned to a physical grid point. Each thread will compute the Fourier series @ that point
    if i < u_.shape[0] and j < v_.shape[1] and k < w_.shape[2]:
        for m in range(0, len(kx)):
            arg = kx[m] * xc[i] + ky[m] * yc[j] + kz[m] * zc[k] - psi[m]
            bmx = 2.0 * um[m] * math.cos(arg - kx[m] * dx / 2.0)
            bmy = 2.0 * um[m] * math.cos(arg - ky[m] * dy / 2.0)
            bmz = 2.0 * um[m] * math.cos(arg - kz[m] * dz / 2.0)
            u_[i, j, k] += bmx * sxm[m]
            v_[i, j, k] += bmy * sym[m]
            w_[i, j, k] += bmz * szm[m]


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

    # compute random angles
    np.random.seed(7)
    phi = 2.0 * pi * np.random.uniform(0.0, 1.0, nmodes)
    nu = np.random.uniform(0.0, 1.0, nmodes)
    theta = np.arccos(2.0 * nu - 1.0)
    psi = np.random.uniform(-pi / 2.0, pi / 2.0, nmodes)

    # highest wave number that can be represented on this grid (nyquist limit)
    wnn = max(np.pi / dx, max(np.pi / dy, np.pi / dz))
    print('I will generate data up to wave number: ', wnn)

    # wavenumber step
    dk = (wnn - wn1) / nmodes

    # wavenumber at cell centers
    wn = wn1 + 0.5 * dk + arange(0, nmodes) * dk

    dkn = ones(nmodes) * dk

    # wavenumber vector from random angles
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
    print('Orthogonality of k and sigma (divergence in wave space):')
    print(kk)

    # get the modes
    km = wn

    espec = especf(km)
    espec = espec.clip(0.0)

    # generate turbulence at cell centers
    um = sqrt(espec * dkn)
    u_ = zeros([nx, ny, nz])
    v_ = zeros([nx, ny, nz])
    w_ = zeros([nx, ny, nz])

    xc = dx / 2.0 + arange(0, nx) * dx
    yc = dy / 2.0 + arange(0, ny) * dy
    zc = dz / 2.0 + arange(0, nz) * dz

    # allocate memory on the device for u_, v_, w_ solutions
    #   cuda_u = cuda.to_device(u_)
    #   cuda_v = cuda.to_device(v_)
    #   cuda_w = cuda.to_device(w_)

    # determine the threads per block and number of blocks
    threads_per_block = (8, 8, 8)
    bpg_x = int(math.ceil(u_.shape[0] / threads_per_block[0]))  # blocks per grid in the x direction
    bpg_y = int(math.ceil(u_.shape[1] / threads_per_block[1]))
    bpg_z = int(math.ceil(u_.shape[2] / threads_per_block[2]))
    blocks_per_grid = (bpg_x, bpg_y, bpg_z)

    # run the kernel
    turbo_kernel[blocks_per_grid, threads_per_block](kx, ky, kz, xc, yc, zc, psi, um, sxm, sym, szm, dx, dy,
                                                     dz, u_, v_, w_)
    return u_, v_, w_
