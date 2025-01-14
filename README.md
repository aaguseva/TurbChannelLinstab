# Linear stability of turbulent mean profile with control

This repository contains subroutines to analyse linear stability of an arbitrary mean velocity profile, including those for turbulent channel flows. A generalised eigenvalue problem is solved for wall-normal vorticity and Laplacian of wall-normal velocity in a periodic channel. Equations are discretized using spectral Chebyshev method in wall-normal direction, and solved for a fixed wavenumbers $\alpha$ and $\beta$ in streamwise and spanwise directions.

The main subroutine is OSS_control, which requires a call to Dmat(N), constructing Chebyshev differetiation matrices, and turbulent velocity and turbulent viscosity profiles. The latter can be calculated from semi-empirical expressions for turbulent channel flows, see e.g. turprof_generic(...). 

The resulting eigenvectors u,v,w can be obtained with get_vel(...)
