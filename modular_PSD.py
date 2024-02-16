import numpy as np
import gc
import full_expression_PSD as full

def Rayleigh_greens_function_vertical(f, rho_s=2400, rx=9.5e3, vc0=1400, f0=1.0,
                                      xi=0.55, Q=50):
    '''
    vertical component of Rayleigh-waves Green's functions in frequency-space
    from equation 4 in Gestrich et al (2020)
    assumes source-receiver azimuthal angle is 0

    ---INPUTS---
    f : frequencies to evaluate Green's functions at

    optional
        rho_s  : solid density (kg / m^3)
        rx     : horizontal source reciever distance (m)
        vc0    : Rayleigh-wave reference velocity (m/s)
        f0     : Rayleigh-wave reference frequency (Hz)
        xi     : Rayleigh-wave velocity exponent
        Q      : Rayleigh-wave quality factor

    ---RETURNS---
    GF : vertical components of Green's functions to convolve w/ arbitrary force
         (3, #frequencies)
    '''
    vc, vu = full.Rayleigh_velocities(f, vc0, f0, xi)
    k = 2 * np.pi * f / vc

    gf_fac = k / (8 * rho_s * vc * vu)
    gf_fac *= np.sqrt(2 / (np.pi * k * rx))
    gf_fac *= np.exp(-np.pi * f * rx / (vu * Q))

    GF = np.zeros((3,) + f.shape)
    GF[0] = 0.8 * gf_fac # x-component of force
    GF[1] = 0 * gf_fac # y-component of force
    GF[2] = 0.6 * gf_fac # z-component

    return GF

def particle_impact_rate(u_mean, phi_p, D, D_dist):
    '''
    returns particle impact rate calculated by eq 7 in Gestrich et al (2020)

    ---INPUTS---
    u_mean : mean vertical velocity (m/s)
    phi_p  : particle volume fraction
    D      : grain size (m)
    D_dist : grain size distribution (1/m)
    ---RETURNS---
    rate : impact rate
    '''
    rate = 0.1 * u_mean * phi_p * D_dist / D**3
    return rate

def particle_impact_force(eb, m, u_mean):
    '''
    calculates Fourier transform of impact force after integrating over
    impact angles, following eq 15 and 16

    ---INPUTS---
    eb     : coefficient of restitution
    m      : mass of particle
    u_mean : mean vertical velocity (m/s)
    ---RETURNS---
    force : x,y,z components of impact force from particle
    '''
    impact = (1 + eb) * m * u_mean

    if isinstance(u_mean, np.ndarray):
        force = np.zeros((3,)+u_mean.shape)
    else:
        force = np.zeros(3)
    force[0] = 0.36 * impact
    force[1] = 0.36 * impact
    force[2] = 0.29 * impact

    return force

def impact_PSD(f, u_mean=120, eb=0.8, phi_p=0.1, Dr=1.4e-3, R=9, rho_s=2400, df=300,
               rx=9.5e3, vc0=1400, f0=1.0, xi=0.55, Q=50):
    '''
    calculates seismic PSD from particle impacts following eq 3 in Gestrich et al (2020)

    ---INPUTS---
    f : frequencies to evaluate PSD

    optional
        u_mean : mean vertical velocity (m/s)
        eb     : coefficient of restitution
        phi_p  : particle volume fraction
        Dr     : representative grain size (m)
        R      : conduit radius (m)
        rho_s  : solid density (kg / m^3)
        df     : fragmentation depth (m)
        rx     : horizontal source reciever distance (m)
        vc0    : Rayleigh-wave reference velocity (m/s)
        f0     : Rayleigh-wave reference frequency (Hz)
        xi     : Rayleigh-wave velocity exponent
        Q      : Rayleigh-wave quality factor

    ---RETURNS---
    PSD : array of PSD values at specified frequencies
    '''
    Dr_dist = 1 # for representative grain size
    m = rho_s * (4/3) * np.pi * Dr**3 # mass of representative grain size

    impact_force = particle_impact_force(eb, m, u_mean)
    impact_rate = particle_impact_rate(u_mean, phi_p, Dr, Dr_dist)
    RW_GF = Rayleigh_greens_function_vertical(f, rho_s, rx, vc0, f0, xi, Q)

    FjGjz = impact_force[0] * RW_GF[0] + impact_force[1] * RW_GF[1] + impact_force[2] * RW_GF[2]

    PSD = 2 * np.pi * R * df * impact_rate * (2 * np.pi * f * FjGjz)**2

    return PSD

def dissipation_rate(u_star, Db, kappa=0.4):
    '''
    eq 23 in Gestrich et al (2020)

    ---INPUTS---
    u_star : shear velocity in turbulent layer (m/s)
    Db     : roughness size (m)

    optional
        kappa : Von Karman constant
    ---RETURNS---
    epsilon : dissipation rate
    '''
    epsilon = u_star**3 / (kappa * Db / 4)
    return epsilon

def Kolmogorov_velocity_spectrum(f, u_mean, Db, K=0.5, kappa=0.4):
    '''
    eq 24 in Gestrich et al (2020)

    ---INPUTS---
    f      : frequency (Hz)
    u_mean : mean vertical velocity (m/s)
    Db     : roughness size (m)

    optional
        K     : Kolmogorov universal constant
        kappa : Von Karman constant
    ---RETURNS---
    Ek_tilda : velocity spectrum for given frequencies
    '''
    kt = 15 * f / u_mean
    epsilon = dissipation_rate(0.06 * u_mean, Db, kappa)

    Ek = K * epsilon**(2/3) * kt**(-5/3) # eq 18 in Gestrich
    Ek_tilda = 2 * np.pi * Ek / (0.42 * u_mean)

    return Ek_tilda

def force_spectrum(f, u_mean, rho_g, Db, C=0.5, K=0.5, kappa=0.4, chi_fl=1.0):
    '''
    eq 25 and 26 in Gestrich et al (2020)

    ---INPUTS---
    f      : frequency (Hz)
    u_mean : mean vertical velocity (m/s)
    rho_g  : gas density (kg / m^3)
    Db     : roughness size (m)

    optional
        C      : drag coefficient:
        K      : Kolmogorov universal constant
        kappa  : Von Karman constant
        chi_fl : fluid-dynamic admittance
    ---RETURNS---
    F_tilda : force spectrum for given frequencies
    '''
    A = np.pi * Db**2 / 4
    Ek_tilda = Kolmogorov_velocity_spectrum(f, u_mean, Db, K, kappa)
    Fp = (C * rho_g * 0.42 * u_mean * A)**2 * Ek_tilda * chi_fl**2

    F_tilda = Fp * Db**(-2)
    return F_tilda

def turbulence_PSD(f, u_mean=120, Db=0.5, rho_g=0.5, rho_s=2400, R=9, df=300,
                   rx=9.5e3, vc0=1400, f0=1.0, xi=0.55, Q=50):
    '''
    calculates turbulence PSD according to equation 17 from Gestrich et al (2020)

    ---INPUTS---
    f : frequencies to evaluate PSD

    optional
        u_mean : mean vertical velocity (m/s)
        Db     : roughness size (m)
        rho_g  : gas density (kg / m^3)
        rho_s  : solid density (kg / m^3)
        R      : conduit radius (m)
        df     : fragmentation depth (m)
        rx     : horizontal source reciever distance (m)
        vc0    : Rayleigh-wave reference velocity (m/s)
        f0     : Rayleigh-wave reference frequency (Hz)
        xi     : Rayleigh-wave velocity exponent
        Q      : Rayleigh-wave quality factor

    ---RETURNS---
    PSD : array of PSD values at specified frequencies
    '''
    F_tilda = force_spectrum(f, u_mean, rho_g, Db)
    RW_GF = Rayleigh_greens_function_vertical(f, rho_s, rx, vc0, f0, xi, Q)

    FGjz2 = F_tilda * (RW_GF[0] + RW_GF[1] + RW_GF[2])**2

    PSD = 8 * np.pi**3 * R * df * f**2 * FGjz2

    return PSD

