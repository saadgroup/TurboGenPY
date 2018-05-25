# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:08:01 2014

@author: Tony Saad
"""
# !/usr/bin/env python
from scipy import interpolate
from scipy import integrate
import numpy as np
from numpy import pi
import time
import scipy.io
from tkespec import compute_tke_spectrum
import isoturb
import isoturbo
from fileformats import FileFormats
import isoio
import cudaturbo

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

#plt.interactive(True)

# load an experimental specturm. Alternatively, specify it via a function call
cbcspec = np.loadtxt('cbc_spectrum.txt')
kcbc = cbcspec[:, 0] * 100
ecbc = cbcspec[:, 1] * 1e-6
especf = interpolate.interp1d(kcbc, ecbc, 'cubic')


def cbc_spec(k):
    return especf(k)


def karman_spec(k):
    nu = 1.0e-5
    alpha = 1.452762113
    urms = 0.25
    ke = 40.0
    kappae = np.sqrt(5.0 / 12.0) * ke
    L = 0.746834 / kappae  # integral length scale - sqrt(Pi)*Gamma(5/6)/Gamma(1/3)*1/ke
    #  L = 0.05 # integral length scale
    #  Kappae = 0.746834/L
    epsilon = urms * urms * urms / L
    kappaeta = pow(epsilon, 0.25) * pow(nu, -3.0 / 4.0)
    r1 = k / kappae
    r2 = k / kappaeta
    espec = alpha * urms * urms / kappae * pow(r1, 4) / pow(1.0 + r1 * r1, 17.0 / 6.0) * np.exp(-2.0 * r2 * r2)
    return espec


def power_spec(k):
    Nu = 1 * 1e-3
    L = 0.1
    Li = 1
    ch = 1
    cl = 10
    p0 = 8
    c0 = pow(10, 2)
    Beta = 2
    Eta = Li / 20.0
    ES = Nu * Nu * Nu / (Eta * Eta * Eta * Eta)
    x = k * Eta
    fh = np.exp(-Beta * pow(pow(x, 4) + pow(ch, 4), 0.25) - ch)
    x = k * L
    fl = pow(x / pow(x * x + cl, 0.5), 5.0 / 3.0 + p0)
    espec = c0 * pow(k, -5.0 / 3.0) * pow(ES, 2.0 / 3.0) * fl * fh
    return espec


# ----------------------------------------------------------------------------------------------
# __    __   ______   ________  _______         ______  __    __  _______   __    __  ________
# |  \  |  \ /      \ |        \|       \       |      \|  \  |  \|       \ |  \  |  \|        \
# | $$  | $$|  $$$$$$\| $$$$$$$$| $$$$$$$\       \$$$$$$| $$\ | $$| $$$$$$$\| $$  | $$ \$$$$$$$$
# | $$  | $$| $$___\$$| $$__    | $$__| $$        | $$  | $$$\| $$| $$__/ $$| $$  | $$   | $$
# | $$  | $$ \$$    \ | $$  \   | $$    $$        | $$  | $$$$\ $$| $$    $$| $$  | $$   | $$
# | $$  | $$ _\$$$$$$\| $$$$$   | $$$$$$$\        | $$  | $$\$$ $$| $$$$$$$ | $$  | $$   | $$
# | $$__/ $$|  \__| $$| $$_____ | $$  | $$       _| $$_ | $$ \$$$$| $$      | $$__/ $$   | $$
# \$$    $$ \$$    $$| $$     \| $$  | $$      |   $$ \| $$  \$$$| $$       \$$    $$   | $$
#  \$$$$$$   \$$$$$$  \$$$$$$$$ \$$   \$$       \$$$$$$ \$$   \$$ \$$        \$$$$$$     \$$
# ----------------------------------------------------------------------------------------------


import argparse
__author__ = 'Tony Saad'
parser = argparse.ArgumentParser(description='This is the Turbulence Generator.')
parser.add_argument('-l','--length', help='Domain size, lx ly lz',required=False, nargs='+', type=float)
parser.add_argument('-n','--res', help='Grid resolution, nx ny nz',required=True, nargs='+', type=int)
parser.add_argument('-m','--modes',help='Number of modes', required=False,type=int)
parser.add_argument('-gpu','--cuda',help='Use a GPU if availalbe', required = False, action='store_true')
parser.add_argument('-mp','--multiprocessor',help='Use the multiprocessing package', required = False,nargs='+', type=int)
parser.add_argument('-o','--output', help='Write data to disk', required = False,action='store_true')
args = parser.parse_args()

# parse grid resolution (nx, ny, nz)
nx = 32
ny = 32
ny = 32
N = args.res
if len(N) == 1:
	nx = ny = nz = N[0]
elif len(N) == 2:
	print('Error! You must specify either all three grid resolutions or just one.')
	exit()
else:
	nx = N[0]
	ny = N[1]
	nz = N[2]

# Default values for domain size in the x, y, and z directions. This value is typically
# based on the largest length scale that your data has. For the cbc data,
# the largest length scale corresponds to a wave number of 15, hence, the
# domain size is L = 2pi/15.
lx = 9 * 2.0 * pi / 100.0
ly = 9 * 2.0 * pi / 100.0
lz = 9 * 2.0 * pi / 100.0
# parse domain length, lx, ly, and lz
L = args.length
if L:
	if len(L) == 1:
		lx = ly = lz = L[0]
	elif len(L) == 2:
		print('Error! You must specify either all three grid resolutions or just one.')
		exit()
	elif len(L) == 3:
		lx = L[0]
		ly = L[1]
		lz = L[2]

# parse number of modes
nmodes = 100
m = args.modes
if m:
	nmodes = int(m)
	print(m)

# specify whether you want to use threads or not to generate turbulence
use_threads = False
patches = [1,1,8]

# specify whether you want to use CUDA or not
use_cuda = False

if args.multiprocessor:
    use_threads = True
    patches = args.multiprocessor
    print('patches = ', patches)
elif args.cuda:
    use_cuda = True

# specify which spectrum you want to use. Options are: cbc_spec, karman_spec, and power_spec
whichspec = locals()['karman_spec']

# specify the spectrum name to append to all output filenames
filespec = 'vkp'

# write to file
enableIO = False  # enable writing to file
io = args.output
if io:
	enableIO = io
fileformat = FileFormats.FLAT  # Specify the file format supported formats are: FLAT, IJK, XYZ

# save the velocity field as a matlab matrix (.mat)
savemat = False

# compute the mean of the fluctuations for verification purposes
computeMean = False

# check the divergence of the generated velocity field
checkdivergence = False

# enter the smallest wavenumber represented by this spectrum
wn1 = min(2.0*pi/lx, min(2.0*pi/ly, 2.0*pi/lz))
# wn1 = 15  # determined here from cbc spectrum properties

# summarize user input
print('-----------------------------------')
print('SUMMARY OF USER INPUT:')
print('Domain size:', lx, ly, lz)
print('Grid resolution:', nx, ny, nz)
print('Fourier accuracy (modes):', nmodes)
print('Using cuda:', use_cuda)
print('Using CPU threads:', use_threads)
if use_threads:
	print('\t patch layout:', patches)

# ------------------------------------------------------------------------------
# END USER INPUT
# ------------------------------------------------------------------------------

# input number of cells (cell centered control volumes). This will
# determine the maximum wave number that can be represented on this grid.
# see wnn below
dx = lx / nx
dy = ly / ny
dz = lz / nz

t0 = time.time()
if use_threads:
    u, v, w = isoturbo.generate_isotropic_turbulence(patches, lx, ly, lz, nx, ny, nz, nmodes, wn1, whichspec)
elif use_cuda:
    u, v, w = cudaturbo.generate_isotropic_turbulence(lx, ly, lz, nx, ny, nz, nmodes, wn1, whichspec)
else:
    u, v, w = isoturb.generate_isotropic_turbulence(lx, ly, lz, nx, ny, nz, nmodes, wn1, whichspec)
t1 = time.time()
elapsed_time = t1 - t0
print('it took me ', elapsed_time, 's to generate the isotropic turbulence.')

if enableIO:
    if use_threads:
        isoio.writefileparallel(u, v, w, dx, dy, dz, fileformat)
    else:
        isoio.writefile('u.txt', 'x', dx, dy, dz, u, fileformat)
        isoio.writefile('v.txt', 'y', dx, dy, dz, v, fileformat)
        isoio.writefile('w.txt', 'z', dx, dy, dz, w, fileformat)

if savemat:
    data = {}  # CREATE empty dictionary
    data['U'] = u
    data['V'] = v
    data['W'] = w
    scipy.io.savemat('uvw.mat', data)

# compute mean velocities
if computeMean:
	umean = np.mean(u)
	vmean = np.mean(v)
	wmean = np.mean(w)
	print('mean u = ', umean)
	print('mean v = ', vmean)
	print('mean w = ', wmean)

	ufluc = umean - u
	vfluc = vmean - v
	wfluc = wmean - w

	print('mean u fluct = ', np.mean(ufluc))
	print('mean v fluct = ', np.mean(vfluc))
	print('mean w fluct = ', np.mean(wfluc))

	ufrms = np.mean(ufluc * ufluc)
	vfrms = np.mean(vfluc * vfluc)
	wfrms = np.mean(wfluc * wfluc)

	print('u fluc rms = ', np.sqrt(ufrms))
	print('v fluc rms = ', np.sqrt(vfrms))
	print('w fluc rms = ', np.sqrt(wfrms))

# check divergence
if checkdivergence:
    count = 0
    for k in range(0, nz - 1):
        for j in range(0, ny - 1):
            for i in range(0, nx - 1):
                src = (u[i + 1, j, k] - u[i, j, k]) / dx + (v[i, j + 1, k] - v[i, j, k]) / dy + (w[i, j, k + 1] - w[
                    i, j, k]) / dz
                if src > 1e-2:
                    count += 1
    print('cells with divergence: ', count)

# verify that the generated velocities fit the spectrum
knyquist, wavenumbers, tkespec = compute_tke_spectrum(u, v, w, lx, ly, lz, False)
# save the generated spectrum to a text file for later post processing
np.savetxt('tkespec_' + filespec + '_' + str(N) + '_' + str(nmodes) + '.txt', np.transpose([wavenumbers, tkespec]))

# -------------------------------------------------------------
# compare spectra
# integral comparison:
# find index of nyquist limit
idx = (np.where(wavenumbers == knyquist)[0][0]) - 2

# km0 = 2.0 * np.pi / lx
# km0 is the smallest wave number
km0 = wn1
# use a LOT of modes to compute the "exact" spectrum
#exactm = 10000
#dk0 = (knyquist - km0) / exactm
#exactRange = km0 + np.arange(0, exactm + 1) * dk0
dk = wavenumbers[1] - wavenumbers[0]
exactE = integrate.trapz(whichspec(wavenumbers[1:idx]), dx=dk)
print(exactE)
numE = integrate.trapz(tkespec[1:idx], dx=dk)
diff = np.abs((exactE - numE)/exactE)
integralE = diff*100.0
print('Integral Error = ', integralE, '%')

# analyze how well we fit the input spectrum
# compute the RMS error committed by the generated spectrum
exact = whichspec(wavenumbers[4:idx])
num = tkespec[4:idx]
diff = np.abs((exact - num) / exact)
meanE = np.mean(diff)

print('Mean Error = ', meanE * 100.0, '%')
rmsE = np.sqrt(np.mean(diff * diff))
print('RMS Error = ', rmsE * 100, '%')


#create an array to save time and error values
array_toSave = np.zeros(4)
array_toSave[1] = integralE
array_toSave[0] = elapsed_time
array_toSave[2] = meanE*100.0
array_toSave[3] = rmsE*100.0

# save time and error values in a txt file
np.savetxt('time_error_' + filespec + '_' + str(N) + '_' + str(nmodes) + '.txt',array_toSave)
#np.savetxt('cpuTime_' + filespec + '_' + str(N) + '_' + str(nmodes) + '.txt',time_elapsed)

# -------------------------------------------------------------

# plt.rc('text', usetex=True)
plt.rc("font", size=10, family='serif')

fig = plt.figure(figsize=(3.5, 2.8), dpi=200, constrained_layout=True)

wnn = np.arange(wn1, 2000)

l1, = plt.loglog(wnn, whichspec(wnn), 'k-', label='input')
l2, = plt.loglog(wavenumbers[1:6], tkespec[1:6], 'bo--', markersize=3, markerfacecolor='w', markevery=1, label='computed')
plt.loglog(wavenumbers[5:], tkespec[5:], 'bo--', markersize=3, markerfacecolor='w', markevery=4, label='computed')
plt.axis([8, 10000, 1e-7, 1e-2])
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
plt.axvline(x=knyquist, linestyle='--', color='black')
plt.xlabel('$\kappa$ (1/m)')
plt.ylabel('$E(\kappa)$ (m$^3$/s$^2$)')
plt.grid()
# plt.gcf().tight_layout()
if nx == ny == nz:
	plt.title(str(nx) + '$^3$')
else:
	plt.title(str(nx) + 'x' + str(ny) + 'x' + str(nz))
plt.legend(handles=[l1, l2], loc=1)
fig.savefig('tkespec_' + filespec + '_' + str(N) + '.pdf')
plt.show()
