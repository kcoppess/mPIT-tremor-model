import numpy as np
import gc

def Rayleigh_velocities(f, vc0, f0, xi):
    '''
    calculates the Rayleigh phase and group velocities following
    equations 30 & 31 in Gestrich et al (2020)

    ---INPUTS---
    f   : frequency (Hz)
    vc0 : Rayleigh reference velocity (m/s)
    f0  : reference frequency (Hz)
    xi  : exponent defining power-law relation

    ---RETURNS---
    vc : Rayleigh-wave phase velocity (m/s)
    vu : Rayleigh-wave group velocity (m/s)
    '''
    vc = vc0 * (f / f0) ** (-xi)
    vu = vc / (1 + xi)
    return vc, vu

def impact_full_expression(f, u_mean=120, eb=0.8, phi_p=0.1, Dr=1.4e-3, R=9,
                          df=300, rx=9.5e3, vc0=1400, f0=1.0, xi=0.55, Q=50):
    '''
    equation for impact PSD given in equation 28 from Gestrich et al (2020)

    ---INPUTS---
    f : frequencies to evaluate PSD

    optional
        u_mean : mean vertical velocity (m/s)
        eb     : coefficient of restitution
        phi_p  : particle volume fraction
        Dr     : representative grain size (m)
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
    vc, vu = Rayleigh_velocities(f, vc0, f0, xi)

    fac1 = 0.088
    fac2 = (u_mean * f)**3
    fac3 = (1 + eb)**2 * phi_p * Dr**3
    fac4 = np.exp(-2 * np.pi * f * rx / (vu * Q)) * R * df / (rx * vc**3 * vu**2)

    PSD = fac1 * fac2 * fac3 * fac4

    return PSD

def turbulence_full_expression(f, u_mean=120, Db=0.5, rho_g=0.5, rho_s=2400, R=9,
                              df=300, rx=9.5e3, vc0=1400, f0=1.0, xi=0.55, Q=50):
    '''
    equation for turbulence PSD given in equation 29 from Gestrich et al (2020)

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
    vc, vu = Rayleigh_velocities(f, vc0, f0, xi)

    fac1 = 5.8e-4
    fac2 = u_mean**(14/3) * f**(4/3)
    fac3 = Db**(4/3) * (rho_g / rho_s)**2
    fac4 = np.exp(-2 * np.pi * f * rx / (vu * Q)) * R * df / (rx * vc**3 * vu**2)

    PSD = fac1 * fac2 * fac3 * fac4

    return PSD
