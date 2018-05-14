#
#  isoturbo.py
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
import multiprocessing as mp
import numpy as np
import time
from numpy import sin, cos, sqrt, ones, zeros, pi, arange
from numpy import linalg as LA

def compute_turbulence(nthread, dx, dy, dz, psi, um, kx, ky, kz, sxm, sym, szm, nx, ny, nz, nxAll, nyAll, nzAll, ip, jp,
                       kp, q):
    print('Generating turbulence on thread:', nthread)
    t0 = time.time()
    u_ = zeros([nx, ny, nz])
    v_ = zeros([nx, ny, nz])
    w_ = zeros([nx, ny, nz])

    xl = (ip - 1) * nx
    xh = ip * nx
    yl = (jp - 1) * ny
    yh = jp * ny
    zl = (kp - 1) * nz
    zh = kp * nz

    xc = dx / 2.0 + arange(xl, xh) * dx
    yc = dy / 2.0 + arange(yl, yh) * dy
    zc = dz / 2.0 + arange(zl, zh) * dz  # cell centered coordinates

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

    t1 = time.time()
    print('Thread ', nthread, ' done generating turbulence in ', t1 - t0, 's')
    q.put((ip, jp, kp, u_, v_, w_))
    return ip, jp, kp, u_, v_, w_


def generate_isotropic_turbulence(patches, lx, ly, lz, nx, ny, nz, nmodes, wn1, especf):
    ## grid generation
    # generate cell centered x-grid
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    ## START THE FUN!
    # compute random angles
    np.random.seed(0)
    phi = 2.0 * pi * np.random.uniform(0.0, 1.0, nmodes);
    nu = np.random.uniform(0.0, 1.0, nmodes);
    theta = np.arccos(2.0 * nu - 1.0);
    psi = np.random.uniform(-pi / 2.0, pi / 2.0, nmodes);
    alfa = 2.0 * pi * np.random.uniform(0.0, 1.0, nmodes);

    # highest wave number that can be represented on this grid (nyquist limit)
    wnn = max(np.pi / dx, max(np.pi / dy, np.pi / dz));
    print('I will generate data up to wave number: ', wnn)

    # wavenumber step
    dk = (wnn - wn1) / nmodes

    # wavenumber at cell centers
    wn = wn1 + arange(0, nmodes) * dk
    #  wn = wn1 + np.arange(0,nmodes)*dk*np.log(np.arange(0,nmodes) + 1)/np.log(nmodes)
    dkn = ones(nmodes) * dk
    #  dkn = wn[1:nmodes] - wn[0:nmodes-1]
    #  dkn = np.append(dkn,dkn[nmodes-2])

    #   wavenumber vector from random angles
    kx = sin(theta) * cos(phi) * wn
    ky = sin(theta) * sin(phi) * wn
    kz = cos(theta) * wn

    # create divergence vector
    ktx = np.sin(kx * dx / 2.0) / (dx)
    kty = np.sin(ky * dy / 2.0) / (dy)
    ktz = np.sin(kz * dz / 2.0) / (dz)

    #  # Use Davidson's Method to enforce Divergence Free Condition
    #  ktmag = sqrt(ktx*ktx + kty*kty + ktz*ktz)
    #  theta = np.arccos(kzstag/kstagmag)
    #  phi = np.arctan2(kystag,kxstag)
    #  sxm = cos(phi)*cos(theta)*cos(alfa) - sin(phi)*sin(alfa)
    #  sym = sin(phi)*cos(theta)*cos(alfa) + cos(phi)*sin(alfa)
    #  szm = -sin(theta)*cos(alfa)

    # another method to generate sigma = zeta x k_tilde, pick zeta randomly
    #  np.random.seed(3)
    phi1 = 2.0 * pi * np.random.uniform(0.0, 1.0, nmodes);
    nu1 = np.random.uniform(0.0, 1.0, nmodes);
    theta1 = np.arccos(2.0 * nu1 - 1.0);
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

    # now create an interpolant for the spectrum. this is needed for
    # experimentally-specified spectra
    #  espec = especf(km + dk/2) + especf(km))*0.5
    espec = especf(km)
    espec = espec.clip(0.0)

    # generate turbulence at cell centers
    um = sqrt(espec * dkn)

    #  must use Manager queue here, or will not work
    nxthreads = patches[0];
    nythreads = patches[1];
    nzthreads = patches[2];
    nxt = nx // nxthreads;
    nyt = nx // nythreads;
    nzt = nx // nzthreads;

    manager = mp.Manager()
    mq = manager.Queue()
    pool = mp.Pool(mp.cpu_count())  # assume 2 threads per core

    # fire off workers
    jobs = []
    nthread = 0
    for k in range(1, nzthreads + 1):
        for j in range(1, nythreads + 1):
            for i in range(1, nxthreads + 1):
                nthread = nthread + 1
                job = pool.apply_async(compute_turbulence, (
                nthread, dx, dy, dz, psi, um, kx, ky, kz, sxm, sym, szm, nxt, nyt, nzt, nx, ny, nz, i, j, k, mq))
                jobs.append(job)

    # collect results from the workers through the pool result queue
    print('now collecting results from individual threads...')
    uarrays = []
    varrays = []
    warrays = []
    patches = []
    for job in jobs:
        i, j, k, u, v, w = job.get()
        uarrays.append(u)
        varrays.append(v)
        warrays.append(w)
        patches.append([i, j, k])
    del u, v, w

    pool.terminate()
    pool.close()

    # combine the arrays computed from threads into large arrays
    print('now combining velocity fields generated by the individual threads...')
    uall = zeros([nx, ny, nz])
    vall = zeros([nx, ny, nz])
    wall = zeros([nx, ny, nz])
    nthread = 0
    for k in range(1, nzthreads + 1):
        for j in range(1, nythreads + 1):
            for i in range(1, nxthreads + 1):
                uall[(i - 1) * nxt:i * nxt, (j - 1) * nyt:j * nyt, (k - 1) * nzt:k * nzt] = uarrays[nthread]
                vall[(i - 1) * nxt:i * nxt, (j - 1) * nyt:j * nyt, (k - 1) * nzt:k * nzt] = varrays[nthread]
                wall[(i - 1) * nxt:i * nxt, (j - 1) * nyt:j * nyt, (k - 1) * nzt:k * nzt] = warrays[nthread]
                nthread = nthread + 1

    return uall, vall, wall
