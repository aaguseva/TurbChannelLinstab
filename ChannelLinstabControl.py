import numpy as np
from scipy import interpolate,integrate
from scipy.linalg import eig,inv,eigvals,svd



# Chebyshev differentiation matrices

def Dmat(N):
    # initialize differentiation matrices
    D0 = np.zeros([N+1,N+1])
    D1 = np.zeros([N+1,N+1])
    D2 = np.zeros([N+1,N+1])
    D3 = np.zeros([N+1,N+1])
    D4 = np.zeros([N+1,N+1])
    # vector 0...1...N
    vec = np.arange(N+1)
    # mesh
    y = np.cos(np.pi*vec/N)
    # filling in D0
    for j in range(0,N+1):
        D0[:,j] = np.cos(j*np.pi*vec/N)
    # starting D1 and D2    
    D1[:,1] = D0[:,0]; D1[:,2] = 4*D0[:,1];
    D2[:,2] = 4*D0[:,0]
    # looping over remaining columns
    for j in range(3,N+1):
        D1[:,j] = 2*j*D0[:,j-1] + j*D1[:,j-2]/(j-2)
        D2[:,j] = 2*j*D1[:,j-1] + j*D2[:,j-2]/(j-2)
        D3[:,j] = 2*j*D2[:,j-1] + j*D3[:,j-2]/(j-2)
        D4[:,j] = 2*j*D3[:,j-1] + j*D4[:,j-2]/(j-2)
    # return matrices and mesh
    return D0, D1, D2, D3, D4, y

# central finite difference scheme for derivative of a function
def deriv(f,y):
    dfdy = np.zeros(len(y))
    dfdy[1:-1] = (f[2:] - f[0:-2])/(y[2:] - y[0:-2])
    dfdy[0] = (f[1] - f[0])/(y[1] - y[0])
    dfdy[-1] = (f[-2] - f[-1])/(y[-2] - y[-1])
    return dfdy

# calculate velocity profile and it's derivatives based on the grid
# needs number of points and mesh from Dmat
# laminar velocity profile
def lam_velprof(N,y):
    Nos = N + 1
    u=np.ones(Nos)-y**2 # u = 1 - x**2
    du=-2*y # du = -2*x
    ddu = -2*np.ones(Nos) # ddu = -2
    return u, du, ddu
   
# generic turbulent velocity profile with variable eddy viscosity
def turprof_generic(N,Aint,kapa,Rt,y):
    Nos=N+1
    # use eddy viscosity to compute profile
    y0 = 1 - y
    nut=0.5*(1 + kapa**2*Rt**2/9*(2*y0-y0**2)**2.*(3-4*y0+2*y0**2)**2*(1-np.exp(-y0*Rt/Aint))**2)**0.5+0.5;
    u=Rt*integrate.cumtrapz((1-y0)/nut,y0, initial=0);
    u = u - u[0]
    du  = -y/nut
    # make symmetric
    n0  = np.int(np.fix(Nos/2))
    n1  = Nos-n0
    du  = np.hstack((du[0:n1], -du[-(n1+1)::-1]))
    nut = np.hstack((nut[0:n1], nut[-(n1+1)::-1]))
    u = np.hstack((u[0:n1], u[-(n1+1)::-1]))
    #print(u)
    # rescale with bulk velocity 
    Ub = np.trapz(u,x=y0)/np.trapz(np.ones(Nos),x=y0)
    #print(Ub)
    u = u/Ub
    Re = Ub*Rt
    #print(Re)
    nut = nut/Re
    du = du*Rt/Ub
    # get high-order derivatives of u and eddy viscosity, p's denote "primes"
    upp = deriv(du,y)
    nutp = deriv(nut,y)
    nutpp = deriv(nutp,y)
    nutppp = deriv(nutpp,y)
    return np.vstack((u,du,upp)), np.vstack((nut,nutp,nutpp,nutppp))

# Orr-Sommerfeld + Squire matrices with variable viscosity, changed to take into account control
# Using Chebyshev pseudospectral discretization for plane Poiseuille flow profile
#N = number of modes
#alp, bet = wave numbers
#R = Reynolds number
#D0, D1, D2, D4 = zero'th, first, second, fourth derivative matrix
# Control: v(0) = Ad*v(yd), yd - number of plane
# Ad - complex: np.abs(Ad) - magnitude of control, np.angle(Ad) - phase 
def OSS_control(N,alp,bet,D0,D1,D2,D3,D4,u,nut,Rt,yd,Ad,ftype='sym'):
    
    # allocate arrays as complex values
    A11 = np.zeros([N+1,N+1],dtype='c16')
    B11 = np.zeros([N+1,N+1],dtype='c16')
    A21 = np.zeros([N+1,N+1],dtype='c16')
    A22 = np.zeros([N+1,N+1],dtype='c16')
    B22 = np.zeros([N+1,N+1],dtype='c16')
    
    # kx^2 + kz^2
    ak2=alp**2+bet**2 
    
    # unpacking velocity
    du = u[1,:]; upp = u[2,:]; u = u[0,:];
    nutppp = nut[3,:]; nutpp = nut[2,:]; nutp = nut[1,:]; nut = nut[0,:];
    
    # placing of spurious eigenvalues
    er = -Rt*1j 
    # Orr-Sommerfeld matrices
    B11[...] = D2 - ak2*D0
    A11[...] = alp*B11*u[:,np.newaxis] - alp*D0*upp[:,np.newaxis]
    A11[...] = A11[...] + 1j*(D4 - 2*ak2*D2 + ak2**2*D0)*nut[:,np.newaxis] + 1j*2*(D3 - ak2*D1)*nutp[:,np.newaxis] + 1j*(D2 + ak2*D0)*nutpp[:,np.newaxis] 
    
    # make solutions symmetric or asymmetric on the upper wall
    if ftype =='sym':
        Atop = Ad
    elif ftype == 'asym':
        Atop = -Ad
        
    # BC (Orr-Sommerfeld + control) v(0) = Ad*v(yd)      
    A11[0,:]  = er*(D0[0,:] - Ad*D0[yd,:]); A11[1,:]  = er*D1[0,:]; 
    A11[-1,:] = er*(D0[-1,:] - Atop*D0[-(yd+1),:]); A11[-2,:] = er*D1[-1,:];     
    B11[0,:]  = D0[0,:] - Ad*D0[yd,:];   B11[1,:]  = D1[0,:];    
    B11[-1,:] = D0[-1,:] - Atop*D0[-(yd+1),:];  B11[-2,:] = D1[-1,:];
    
    # Squire matrices
    A21[...] = bet*D0*du[:,np.newaxis]
    A22[...] = -(D2 - ak2*D0)*nut[:,np.newaxis]/1j
    A22[...] = A22[...] - D1*nutp[:,np.newaxis]/1j
    A22[...] = A22[...] + alp*D0*u[:,np.newaxis]
    B22[...] = D0;
    
    # BC (Squire)
    A22[0,:] = er*D0[0,:]; A22[-1,:] = er*D0[-1,:]
    A21[0,:] = 0.0; A21[-1,:] = 0.0 # this is here because the velocity term is absent in the BC for vorticity
    
    # combine two systems in one
    A = np.vstack((np.hstack((A11, np.zeros([N+1,N+1]))), np.hstack((A21,A22))))
    B = np.vstack((np.hstack((B11, np.zeros([N+1,N+1]))), np.hstack((np.zeros([N+1,N+1]),B22))))
    return A11,B11,A,B


# Inviscid Rayleigh equation, changed to take into account control
# Control: v(0) = Ad*v(yd), yd - number of plane
# Ad - complex: np.abs(Ad) - magnitude of control, np.angle(Ad) - phase 
def OSS_inviscid_control(N,alp,bet,D0,D1,D2,D3,D4,u,Rt,yd,Ad):
    
    # allocate arrays as complex values
    A11 = np.zeros([N+1,N+1],dtype='c16')
    B11 = np.zeros([N+1,N+1],dtype='c16')

    # kx^2 + kz^2
    ak2=alp**2+bet**2 
    
    # unpacking velocity
    du = u[1,:]; upp = u[2,:]; u = u[0,:];
    
    # placing of spurious eigenvalues
    er = -Rt*1j 
    # Orr-Sommerfeld matrices
    B11[...] = D2 - ak2*D0
    A11[...] = alp*B11*u[:,np.newaxis] - alp*D0*upp[:,np.newaxis]
    
    # BC (Orr-Sommerfeld + control) v(0) = Ad*v(yd)       
    A11[0,:]  = er*(D0[0,:] - Ad*D0[yd,:]); 
    A11[-1,:] = er*(D0[-1,:] - Ad*D0[-(yd+1),:]);   
    B11[0,:]  = D0[0,:] - Ad*D0[yd,:];   
    B11[-1,:] = D0[-1,:] - Ad*D0[-(yd+1),:];
    
    return A11,B11

# given eigenvector vec, calculate velocity
def get_vel(alp,bet,D0,D1,vec):
    ak2 = alp**2 + bet**2 # sum of wavenumbers squared
    v = D0@vec[0:N+1] # wall-normal velocity
    eta = D0@vec[N+1:] # omega_y vorticity
    u = 1j*(alp*D1@vec[0:N+1] - bet*D0@vec[N+1:])/ak2 # streamwise velocity component
    w = 1j*(bet*D1@vec[0:N+1] + alp*D0@vec[N+1:])/ak2 # spanwise velocity
    uv = -0.5*(Rw**2)*np.real(u.conj()*v) 
