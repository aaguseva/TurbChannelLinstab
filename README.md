# Linear stability of turbulent mean profile with control

This repository contains subroutines to analyse linear stability of an arbitrary mean velocity profile, including those for turbulent channel flows. A generalised eigenvalue problem is solved for wall-normal vorticity and Laplacian of wall-normal velocity.

The main subroutine is OSS_control, which requires a call to Dmat(N), constructing Chebyshev differetiation matrices, and turbulent velocity and turbulent viscosity profiles (e.g. from turprof_generic).
