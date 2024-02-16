import numpy as np
import gc
from scipy.interpolate import griddata
import scipy.io as sio
import scipy.interpolate as si
import modular_PSD as mod

def turbulence_PSD_extended(f, u_mean, rx, Db=0.5, rho_g=0.5, rho_s=2400, R=9, 
                            dz=2, vc0=1400, f0=1.0, xi=0.55, Q=50):
    '''
    calculates turbulence PSD according to equation 17 from Gestrich et al (2020) for
    extended source (rather than point source)
    
    ---INPUTS---
    f      : frequencies to evaluate PSD
    u_mean : mean vertical velocity (m/s)
    rx     : horizontal source reciever distance for each source segment (m)
    
    optional
        Db     : roughness size (m)
        rho_g  : gas density (kg / m^3)
        rho_s  : solid density (kg / m^3)
        R      : conduit radius (m)
        dz     : segment vertical width (m)
        vc0    : Rayleigh-wave reference velocity (m/s)
        f0     : Rayleigh-wave reference frequency (Hz)
        xi     : Rayleigh-wave velocity exponent
        Q      : Rayleigh-wave quality factor
    
    ---RETURNS---
    PSD : array of PSD values at specified frequencies for each source segment
    '''
    ff = len(f)
    ll = len(u_mean)
    
    f_matrix = np.tile(f, (ll, 1))
    u_matrix = np.tile(u_mean, (ff, 1)).transpose()
    F_tilda = mod.force_spectrum(f_matrix, u_matrix, rho_g, Db)
    gc.collect()
    
    rx_matrix = np.tile(rx, (ff, 1)).transpose()
    RW_GF = mod.Rayleigh_greens_function_vertical(f_matrix, rho_s, rx_matrix, vc0, f0, xi, Q)
    gc.collect()
    
    FGjz2 = F_tilda * (RW_GF[0] + RW_GF[1] + RW_GF[2])**2
    
    PSD = 8 * np.pi**3 * R * dz * f_matrix**2 * FGjz2
    
    return PSD

def impact_PSD_extended(f, u_mean, Dr, rx, eb=0.8, phi_p=0.1, R=9, rho_s=2400,
                        dz=2, vc0=1400, f0=1.0, xi=0.55, Q=50):
    '''
    calculates seismic PSD from particle impacts following eq 3 in Gestrich et al (2020) for
    extended source (rather than point source)

    ---INPUTS---
    f      : frequencies to evaluate PSD
    u_mean : mean vertical velocity (m/s)
    Dr     : representative grain size (m)
    rx     : horizontal source reciever distance for each source segment (m)

    optional
        eb     : coefficient of restitution
        phi_p  : particle volume fraction
        R      : conduit radius (m)
        rho_s  : solid density (kg / m^3)
        dz     : segment vertical width (m)
        vc0    : Rayleigh-wave reference velocity (m/s)
        f0     : Rayleigh-wave reference frequency (Hz)
        xi     : Rayleigh-wave velocity exponent
        Q      : Rayleigh-wave quality factor

    ---RETURNS---
    PSD : array of PSD values at specified frequencies
    '''
    ff = len(f)
    ll = len(u_mean)

    f_matrix = np.tile(f, (ll, 1))
    u_matrix = np.tile(u_mean, (ff, 1)).transpose()
    Dr_matrix = np.tile(Dr, (ff, 1)).transpose()
    rx_matrix = np.tile(rx, (ff, 1)).transpose()

    Dr_dist = 1 # for representative grain size
    m = rho_s * (4/3) * np.pi * Dr_matrix**3 # mass of representative grain size

    impact_force = mod.particle_impact_force(eb, m, u_matrix)
    impact_rate = mod.particle_impact_rate(u_matrix, phi_p, Dr_matrix, Dr_dist)
    RW_GF = mod.Rayleigh_greens_function_vertical(f_matrix, rho_s, rx_matrix, vc0, f0, xi, Q)

    FjGjz = impact_force[0] * RW_GF[0] + impact_force[1] * RW_GF[1] + impact_force[2] * RW_GF[2]

    PSD = 2 * np.pi * R * dz * impact_rate * (2 * np.pi * f_matrix * FjGjz)**2

    return PSD

def load_gfs_ES(directory, depths, RETURNTIME=False, INTERPOLATE=True, REPEATED=0):
    '''
    loads in extended source single force Green's functions in fourier domain
    and can interpolate in space to get compatible array dimensions with desired
    depth array

    original Green's functions must be stored such that the columns correspond below:
            time    vertical    radial   transverse

    --INPUTS--
    directory    : string            : path to folder holding Green's function files
    depths       : (# grid points)   : desired depth array
    INTERPOLATE  : bool              : if True, interpolate to get values at desired
                                                and depths/frequencies
    --RETURNS--
    gf_freq : (# freq points)                       : sampled frequencies for GFs
    gfs     : [ (# grid points, # freq points, 3) ] : list of final Green's functions (ver, rad, tra)
                                                        single force, 2 arrays
    '''
    components = ['horizontal_force.mat', 'vertical_force.mat']

    if not INTERPOLATE:
        gfs = []
        gf_time = sio.loadmat(directory+'time.mat')['out'][0]
        for com in components:
            gf = sio.loadmat(directory+com)['out']
            gfs.append(gf)
        gc.collect()
        return gf_time, gfs

    gfs = []
    gfs_hat = []
    gf_time = sio.loadmat(directory+'time.mat')['out'][0]
    gf_tt = len(gf_time)
    if 'interpolated' in directory:
        gf_depths = sio.loadmat(directory+'depths.mat')['out'][0].astype(float)
    else:
        gf_depths = sio.loadmat(directory+'depths.mat')['out'][:].astype(float)*1e3
    gf_hh = len(gf_depths)
    hh = len(depths)
    gf_dt = gf_time[1] - gf_time[0]
    gf_freq = np.fft.fftfreq(gf_tt, gf_dt)
    ind0 = np.argwhere(gf_freq < 0)[0,0]

    for com in components:
        gf = np.zeros((gf_hh, gf_tt, 3))
        gf[:,:] = np.real(sio.loadmat(directory+com)['out'])
        for ii in range(REPEATED):
            gf[-ii] = gf[-REPEATED]
        gfs.append(gf)
        gfs_hat.append(np.fft.fft(gf, axis=1) * gf_dt)
    gc.collect()

    new_gfs_hat = []
    for func, lab in zip(gfs_hat, components):
        smooth = si.interp1d(gf_depths, func, axis=0, kind='linear', fill_value='extrapolate') #bounds_error=False, fill_value=(func[-1], func[0])    )
        gf_hat_sm = smooth(depths)
        new_gfs_hat.append(gf_hat_sm[:,:ind0])

    if RETURNTIME:
        return gf_freq[:ind0], new_gfs_hat, gf_time
    return gf_freq[:ind0], new_gfs_hat

def impact_PSD_ES_numGF(f, depths, gfs, u_mean, Dr, phi_p, dz=1, eb=0.8, R=9, rho_s=2700):
    '''
    calculates seismic PSD from particle impacts following eq 3 in Gestrich et al (2020) for
    extended source (rather than point source) using numerical Green's functions
    rather than closed-form expression

    ---INPUTS---
    f      : frequencies to evaluate PSD
    depths : source depths (m)
    gfs    : numerical Green's functions for extended source
    u_mean : mean vertical velocity (m/s)
    Dr     : representative grain size (m)
    phi_p  : particle volume fraction

    optional
        dz     : segment vertical width (m)
        eb     : coefficient of restitution
        R      : conduit radius (m)
        rho_s  : solid density (kg / m^3)

    ---RETURNS---
    PSD : array of PSD values at specified frequencies
    '''
    ff = len(f)
    ll = len(depths)

    f_matrix = np.tile(f, (ll, 1))
    u_matrix = np.tile(u_mean, (ff, 1)).transpose()
    Dr_matrix = np.tile(Dr, (ff, 1)).transpose()
    phi_p_matrix = np.tile(phi_p, (ff, 1)).transpose()

    Dr_dist = 1 # for representative grain size
    m = rho_s * (4/3) * np.pi * Dr_matrix**3 # mass of representative grain size

    impact_force = mod.particle_impact_force(eb, m, u_matrix)
    impact_rate = mod.particle_impact_rate(u_matrix, phi_p_matrix, Dr_matrix, Dr_dist)
    gc.collect()

    FjGjz = np.sqrt(impact_force[0]**2 + impact_force[1]**2) * gfs[0][:,:,0] + impact_force[2] * gfs[1][:,:,0]
    #FjGjz = impact_force[0] * gfs[:,:,1] + impact_force[1] * gfs[:,:,2] + impact_force[2] * gfs[:,:,0]

    PSD = 2 * np.pi * R * dz * impact_rate * np.abs((2 * np.pi * f_matrix * FjGjz)**2)

    return PSD

def turbulence_PSD_ES_numGF(f, depths, gfs, u_mean, Db, rho_g, dz=1, R=9):
    '''
    calculates turbulence PSD according to equation 17 from Gestrich et al (2020) for
    extended source (rather than point source) using numerical Green's functions
    rather than closed-form expression

    ---INPUTS---
    f      : frequencies to evaluate PSD
    depths : source depths (m)
    gfs    : numerical Green's functions for extended source
    u_mean : mean vertical velocity (m/s)
    Db     : roughness size (m)
    rho_g  : gas density (kg / m^3)

    optional
        dz     : segment vertical width (m)
        R      : conduit radius (m)

    ---RETURNS---
    PSD : array of PSD values at specified frequencies for each source segment
    '''
    ff = len(f)
    ll = len(depths)

    f_matrix = np.tile(f, (ll, 1))
    u_matrix = np.tile(u_mean, (ff, 1)).transpose()
    Db_matrix = np.tile(Db, (ff, 1)).transpose()
    rho_g_matrix = np.tile(rho_g, (ff, 1)).transpose()

    F_tilda = mod.force_spectrum(f_matrix, u_matrix, rho_g_matrix, Db_matrix)
    gc.collect()

    FGjz2 = F_tilda * (np.sqrt(2) * gfs[0][:,:,0] + gfs[1][:,:,0])**2
    #FGjz2 = F_tilda * (gfs[:,:,1] + gfs[:,:,2] + gfs[:,:,0])**2

    PSD = 8 * np.pi**3 * R * dz * f_matrix**2 * np.abs(FGjz2)

    return PSD

