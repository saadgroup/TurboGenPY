# TurboGenPY
TurboGenPY is a synthetic isotropic turbulence generator for structured grids. It can take in an arbitrary
turbulent energy spectrum and produce a three dimensional vector field (e.g. velocity) whose energy spectrum
matches the input spectrum up to the discrete nyquist limit.

As the name indicates, this generator is written for Python, and specifically, Python 3.x.

# Supported Spectra
TurboGenPY supports the following turbulent energy spectra:
1. CBC (Comte-Bellot, Corrsin)
2. von-Karman Pao
3. Power spectrum

You can certainly add your own spectrum. If you are using experimentally observed data, then mimic how the CBC spectrum is implemented via an interpolant.

# Usage
See examply.py for an example of how to use the generator. Here are some typical scenarios:
