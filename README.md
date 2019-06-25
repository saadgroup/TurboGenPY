# TurboGenPY
TurboGenPY is a synthetic isotropic turbulence generator for structured grids. It can take in an arbitrary
turbulent energy spectrum and produce a three dimensional vector field (e.g. velocity) whose energy spectrum
matches the input spectrum up to the discrete nyquist limit.

As the name indicates, this generator is written for Python, and specifically, Python 3.x.

To cite this work, please use the following citations:
1. Tony Saad, Derek Cline, Rob Stoll, and James C. Sutherland. “Scalable Tools for Generating Synthetic Isotropic Turbulence with Arbitrary Spectra”. ,http://dx.doi.org/10.2514/1.J055230. (available online: Aug 25, 2016).
2. Saad, T., & Sutherland, J. C. (2016). Comment on “Diffusion by a random velocity field” [Phys. Fluids 13, 22 (1970)]. Physics of Fluids (1994-Present), 28(11), 119101. https://doi.org/10.1063/1.4968528.
3. Austin Richards, Tony Saad, and James C. Sutherland. “A Fast Turbulence Generator using Graphics Processing Units”, 2018 Fluid Dynamics Conference, AIAA AVIATION Forum, (AIAA 2018-3559). https://doi.org/10.2514/6.2018-3559.

# Supported Spectra
TurboGenPY supports the following turbulent energy spectra:
1. CBC (Comte-Bellot, Corrsin)
2. von-Karman Pao
3. Kang, Chester, and Meneveau
4. Your experimentally measured data

You can certainly add your own spectrum. If you are using experimentally observed data, then mimic how the CBC spectrum is implemented via an interpolant.

# Usage
See examply.py for an example of how to use the generator.

```
> python example.py -h

usage: example.py [-h] [-l LENGTH [LENGTH ...]] -n RES [RES ...] [-m MODES]
                  [-gpu] [-mp MULTIPROCESSOR [MULTIPROCESSOR ...]] [-o]
                  [-spec SPECTRUM]

This is the Utah Turbulence Generator.

optional arguments:
  -h, --help            show this help message and exit
  -l LENGTH [LENGTH ...], --length LENGTH [LENGTH ...]
                        Domain size, lx ly lz
  -n RES [RES ...], --res RES [RES ...]
                        Grid resolution, nx ny nz
  -m MODES, --modes MODES
                        Number of modes
  -gpu, --cuda          Use a GPU if availalbe
  -mp MULTIPROCESSOR [MULTIPROCESSOR ...], --multiprocessor MULTIPROCESSOR [MULTIPROCESSOR ...]
                        Use the multiprocessing package
  -o, --output          Write data to disk
  -spec SPECTRUM, --spectrum SPECTRUM
                        Select spectrum. Defaults to cbc. Other options
                        include: vkp, and kcm.
```

# Examples

```
python example.py -n 32
```
generates turbulence in a $32^3$ box using the default CBC spectrum along with the default length for that spectrum

```
python example.py -n 32 34 32 -l 0.565 0.6 0.565
```
generates turbulence in a domain of size lx = 0.2, ly = 0.3, and lz = 0.2 with grid resolution nx = 32, ny = 34, and nz = 32.

```
python example.py -n 32 -l 0.565 -m 1000 -spec vkp
```
generates turbulence in a box of size 0.565^3 with resolution of 32^3 along with 1000 modes (resolution in wave space) and according to the von-Karman-Pao spectrum

```
python example.py -n 32 -mp 2 1 2
```
uses multiprocessors (or threads). Here, each spatial direction will have its own thread layout: 2 threads in the x direction, 1 thread in the y direction, and 2 threads in the z direction for atotal of 2x1x2 = 4 threads
