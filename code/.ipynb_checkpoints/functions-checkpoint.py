#Included: polyfit2d, mkpol, select_isochrone, fit_bkg, delve_movie_maker, select_cutout, select_stream_cutout, spherical_harmonic_background_fit
#Included: movie_maker, get_filter_splines, filter_data, table_maker, object2mutable, make_it_big, make_gif

#Included: stream data, font properties
#Imports
#from __future__ import division
import astropy
import hpgeom as hp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.io import fits
import os
from astropy.table import Table, vstack
import skyproj
import pyproj
import healsparse as hsp
import healpy as hp
import ugali
import warnings
from matplotlib.path import Path
from ugali.analysis.isochrone import factory as isochrone_factory
from ugali.utils.shell import get_iso_dir
import hats
import lsdb
from matplotlib.colors import LogNorm
from astropy.table import QTable
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.polynomial import polynomial
from numpy.polynomial.polynomial import polyvander2d
import numpy.ma as ma
import glob
import astropy.io.fits as fitsio
from astropy import table
from scipy.interpolate import interp1d
import subprocess

#Polyfit
def polyfit2d(x, y, f, deg):
    """
    Fit a 2d polynomial.

    Parameters:
    -----------
    x : array of x values
    y : array of y values
    f : array of function return values
    deg : polynomial degree (length-2 list)

    Returns:
    --------
    c : polynomial coefficients
    """
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f)[0]
    return c.reshape(deg+1)

# Original Define Isochrone Functions
def mkpol(mu, age=12., z=0.0004, dmu=0.5, C=[0.05, 0.05], E=4., clip=True):
    """Make the the isochrone matched-filter polygon.

    Parameters
    ----------
    mu : distance modulus
    age: isochrone age [Gyr]
    z  : isochrone metallicity [Z_sol]
    dmu: distance modulus spread
    C  : additive color spread
    E  : multiplicative error spread
    err: photometric error model; function that returns magerr as 
         a function of magnitude.
    survey: key to load error model
    clip: maximum absolute magnitude
    
    Returns
    -------
    verts : vertices of the polygon
    """
            #warnings.warn('Using DES photometric error model!')
    err=lambda x: 0.0010908679647672335 + np.exp((x - 27.091072029215375) / 1.0904624484538419)
    #err=lambda x: 0.0004 + np.exp((x - 26.8) / 1.2) #Sam's error function
    #print(err)
    #print('i am error')
            #err = surveys.surveys['DES_DR1']['err']
    """ Builds ordered polygon for masking """
    iso = isochrone_factory('Dotter', survey='DES', age=age, distance_modulus=mu, z=z)

    color=iso.color
    mag=iso.mag
    #print(color)
    #print(mag)


    if clip is not None:
        # Clip for plotting, use gmin otherwise
        # clip abs mag
        cut = (mag > clip) & ((mag + mu) < 24) & \
            (color > 0) & (color < 1)
        color = color[cut]
        mag = mag[cut]

    # Spread in magnitude     
    mnear = mag + mu - dmu / 2.
    mfar = mag + mu + dmu / 2.
    #print(mnear)
    #print('hi')
    #print(mfar)
    #print('i am mfar')
    # Spread in color
    C = np.r_[color + E * err(mfar) + C[1], color[::-1] - E * err(mnear[::-1]) - C[0]]
    #print(err(mfar - mu))
    #print(C[0],C[1])
    #print('i am mfrae error')
    M = np.r_[mag, mag[::-1]]
    return np.c_[C, M]


def select_isochrone(mag_g, mag_r, iso_params=[17.0, 12.5, 0.0001], dmu=0.5, C=[0.01, 0.01], E=2, gmin=None):
    #select iso parameters from paper
    #C is width of g-r
    #E multiplies error dw
    #gmin IF cut off RGB, other bright stars
    """Create the isochrone matched-fitler polygon and select stars that
    reside within it.

    Parameters
    ----------
    mag_g, mag_r : measured magnitudes of the stars
    err: photometric error model; function that returns magerr as 
         a function of magnitude.
    iso_params : isochrone parameters [mu, age, Z]
    dmu: distance modulus spread
    C  : additive color spread
    E  : multiplicative error spread
    gmin : bright magnitude limit in g-band
    survey: key to load error model
    
    Returns
    -------
    selection : boolean array indicating if the object is in the polygon

    """
    mu, age, z = iso_params
    
    
    mk = mkpol(mu=mu, age=age, z=z, dmu=dmu, C=C, E=E, clip=None)
    #print(mk)
    pth = Path(mk)
    #print(pth)
    cm = np.vstack([mag_g - mag_r, mag_g - mu]).T
    idx = pth.contains_points(cm)
    if gmin:
        idx &= (mag_g > gmin)
        #mk = mk[mk[:,1] >= gmin]
    return idx #Sam: return mk, idk

## Original Define isochrone functions

def zach():
    
    ''' Makes a zachary object'''
    
    zach = 'I am Zachary'
    return zach

def fit_bkg(data, proj, sigma=0.1, percent=[2, 95], deg=5):
    nside = hp.get_nside(data.mask)
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)

    vmin, vmax = np.percentile(data.compressed(), q=percent)
    data = np.clip(data, vmin, vmax)
    data.fill_value = np.ma.median(data)

    smoothed = hp.smoothing(data, sigma=np.radians(sigma), verbose=False)
    data = np.ma.array(smoothed, mask=data.mask)

    sel = ~data.mask
    x, y = proj.ang2xy(lon[sel], lat[sel], lonlat=True)

    xmin, xmax, ymin, ymax = proj.get_extent()
    sel2 = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
    # sel2 = ~np.isnan(x) & ~np.isnan(y)

    v = data[sel][sel2]
    x = x[sel2]
    y = y[sel2]

    c = polyfit2d(x, y, v, [deg, deg])

    # Evaluate the polynomial
    x, y = proj.ang2xy(lon, lat, lonlat=True)
    bkg = polynomial.polyval2d(x, y, c)
    bkg = np.ma.array(bkg, mask=data.mask, fill_value=np.nan)

    return bkg

def delve_movie_maker(data, mu_start=15, mu_end=18, mu_step=0.5, save=False, show=True, age=11., z=0.0007, smoothing=0, title='Delve'):
    arange = np.arange(mu_start, mu_end + mu_step, mu_step)
    for mu in arange:
        selector = select_isochrone(data['MAG_PSF_SFD_G'], data['MAG_PSF_SFD_R'], iso_params=[mu, age, z])
        iso_selection = data[selector]
        
        # RA/Dec plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ra = iso_selection['RA']
        dec = iso_selection['DEC']
        sp = skyproj.McBrydeSkyproj(ax=ax1)
        sp.legend()
        hpxbin = sp.draw_hpxbin(ra, dec, nside=512)
        sp.draw_hpxmap(hp.smoothing(hpxbin[0], fwhm=np.radians(smoothing)))
        ax1.set_title('SMC Spatial Plot', fontsize=14)
        
        #Color/Magnitude Plot
        box = mkpol(mu=mu, age=age, z=z, dmu=0.5, C=[0.04, 0.04], E=4., clip=0)
        hist = ax2.hist2d(
            (data['MAG_PSF_SFD_G'] - data['MAG_PSF_SFD_R']),
            data['MAG_PSF_SFD_G'],
            bins=(100, 100),
            cmap=plt.cm.viridis
        )
        ax2.set_title(f'{title}, m-M = {mu:.1f}', fontsize=14)
        ax2.set_xlabel('g-r', fontsize=12)
        ax2.set_ylabel('g', fontsize=12)
        ax2.invert_yaxis()

        cbar = plt.colorbar(hist[3], ax=ax2)
        cbar.set_label('N stars (10000 bins)', fontsize=12)
        x, y = zip(*box)
        x = list(x) + [x[0]]  # Close the polygon
        y = list(y) + [y[0]]
        ax2.plot(x, y, marker=None, markersize=20, color='red')
        ax2.fill(x, y, color='lightblue', alpha=0.2)
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{mu}_slice.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            
#Stream Data
stream_dists = {'Tucana III': 25.1,
                'ATLAS': 22.9,
                'Phoenix': 19.1,
                'Indus': 16.6,
                'Jhelum': 13.2,
                'Chenab': 39.8,
                'Elqui': 50.1,
                'Aliqa Uma': 28.8,
                'Turranburra': 27.5,
                '300S':18,
                'Ravi':22.9,
                'Herc':132}

stream_rvs = {'ATLAS': -99.8,
              'Phoenix': 47.9,
              'Indus': -47.2,
              'Jhelum': (0.20, -34.0),
              'Chenab': -145.5,
              'Elqui': -33.9,
              'Aliqa Uma': -34.0,
              'Ravi':0.,
              'Herc':46.81}


stream_masses = {'Tucana III': 2e-7,
                 'ATLAS': 2e-6,
                 'Phoenix': 2e-6,
                 'Indus': 0.001,
                 'Jhelum': 0.002,
                 'Chenab': 0.001,
                 'Elqui': 0.0001,
                 'Aliqa Uma': 2e-6,
                 'Turranburra': 0.0001,
                 '300S':2e-6,
                 'Ravi':0.0001,
                 'Herc':2e-6}

stream_sigmas = {'Tucana III': 0.05,
                 'ATLAS': 0.01,
                 'Phoenix': 0.01,
                 'Indus': 0.1,
                 'Jhelum': 0.1,
                 'Chenab': 0.5,
                 'Elqui': 0.1,
                 'Aliqa Uma': 0.01,
                 'Turranburra': 0.1,
                 '300S':0.01,
                 'Ravi':0.01,
                 'Herc':0.01}


stream_matrices = {'Aliqa Uma': [[0.66315359, 0.48119409, -0.57330582], [0.74585903, -0.36075668, 0.5599544], [-0.06262284, 0.79894109, 0.59814004]],
                   'ATLAS': [[0.83697865, 0.29481904, -0.4610298], [0.51616778, -0.70514011, 0.4861566], [0.18176238, 0.64487142, 0.74236331]],
                   'Chenab': [[0.51883185, -0.34132444, -0.78378003], [-0.81981696, 0.06121342, -0.56934442], [-0.24230902, -0.93795018, 0.2480641]],
                   'Elqui': [[0.74099526, 0.20483425, -0.63950681], [0.57756858, -0.68021616, 0.45135409], [0.34255009, 0.70381028, 0.62234278]],
                   'Indus': [[0.47348784, -0.22057954, -0.85273321], [0.25151201, -0.89396596, 0.37089969], [0.84412734, 0.39008914, 0.3678036]],
                   'Jhelum': [[0.60334991, -0.20211605, -0.7714389], [-0.13408072, -0.97928924, 0.15170675], [0.78612419, -0.01190283, 0.61795395]],
                   'Phoenix': [[0.5964467, 0.27151332, -0.75533559], [-0.48595429, -0.62682316, -0.60904938], [0.63882686, -0.73032406, 0.24192354]],
                   'Tucana III': [[0.505715, -0.007435, -0.862668], [-0.078639, -0.996197, -0.037514], [0.859109, -0.086811, 0.504377]],
                   'Turranburra': [[0.36111266, 0.85114984, -0.38097455], [0.87227667, -0.16384562, 0.46074725], [-0.32974393, 0.49869687, 0.80160487]],
                   'Jet': [[-0.69798645,  0.61127501, -0.37303856], [-0.62615889, -0.26819784,  0.73211677], [0.34747655,  0.744589,  0.56995374]],
                   '300S': [[-0.88197819,  0.38428506,  0.27283596], [ 0.43304104,  0.88924457,  0.14737555], [-0.18598367,  0.24813119, -0.95070552]],
                   'Ravi': [[0.57336113, -0.22475898, -0.78787081], [0.57203155, -0.57862539, 0.58135407], [0.58654661, 0.78401279, 0.20319208]],
                   'Herc': [[-0.36919252, -0.90261961,  0.22130235], [-0.75419217,  0.43013129,  0.49616655], [-0.54303872,  0.01627647, -0.8395499 ]]}

stream_lengths = {'Aliqa Uma': 10.0,
                  'ATLAS': 22.6,
                  'Chenab': 18.5,
                  'Elqui': 9.4,
                  'Indus': 20.3,
                  'Jhelum': 29.2,
                  'Phoenix': 13.6,
                  'Tucana III': 4.8,
                  'Turranburra': 16.9,
                  'Jet':20.0,
                  '300S':17.0,
                  'Ravi':60.,
                  'Herc':1.}

stream_widths = {'Aliqa Uma': 0.26,
                 'ATLAS': 0.24,
                 'Chenab': 0.71,
                 'Elqui': 0.54,
                 'Indus': 0.83,
                 'Jhelum': 1.16,
                 'Phoenix': 0.16,
                 'Tucana III': 0.18,
                 'Turranburra': 0.60,
                 'Jet':0.2,
                 '300S':0.47,
                 'Ravi':0.72,
                 'Herc':6.3/60.}

stream_vrs = {'Aliqa Uma': 0,
              'ATLAS': 0,
              'Chenab': -150,
              'Elqui': 0,
              'Indus': 0,
              'Jhelum': 0,
              'Phoenix': 0,
              'Tucana III': 0,
              'Turranburra': 0,
              'Jet':272.5,
              '300S':300,
              'Ravi': 0,
              'Herc':46.81}

stream_vr_widths = {'Aliqa Uma': 0,
                    'ATLAS': 0,
                    'Chenab': 4.32,
                    'Elqui': 8.40,
                    'Indus': 5.76,
                    'Jhelum': 13.30,
                    'Phoenix': 0,
                    'Tucana III': 0,
                    'Turranburra': 0,
                    'Jet':0,
                    '300S':0,
                    'Ravi':0,
                    'Herc':0}

stream_phi2s = {'Aliqa Uma': 0,
                'ATLAS': 0.66,
                'Chenab': 0,
                'Elqui': 0,
                'Indus': 0,
                'Jhelum': 0,
                'Phoenix': 0,
                'Tucana III': 0,
                'Turranburra': 0,
                'Jet':0,
                '300S':0,
                'Ravi':0,
                'Herc':0}

stream_mids = {'Aliqa Uma': (35.96519575304428, -34.98107408877647),
               'ATLAS': (19.40440473586367, -27.453578354852883),
               'Chenab': (-33.339759611694205, -51.60798647784715),
               'Elqui': (15.452462557149355, -39.75505320589948),
               'Indus': (-24.978978038471325, -58.510205017947726),
               'Jhelum': (-18.520322938617255, -50.483277585344595),
               'Phoenix': (24.475886399510763, -49.054701430221975),
               'Tucana III': (-1.4880868099542681, -59.641037041735935),
               'Turranburra': (67.01021150038846, -22.394061648029155),}

stream_phi12_pms = {'Aliqa Uma': {'pm1': 0.9803, 'e_pm1': 0.0350, 'pm2': -0.3416, 'e_pm2': 0.0277, 'grad_pm1': -0.0224, 'e_grad_pm1': 0.0209, 'grad_pm2': -0.0362, 'e_grad_pm2': 0.0212},
                    'ATLAS': {'pm1': 1.6602, 'e_pm1': 0.0428, 'pm2': -0.1537, 'e_pm2': 0.0351, 'grad_pm1': 0.0155, 'e_grad_pm1': 0.0049, 'grad_pm2': -0.0179, 'e_grad_pm2': 0.0044},
                    'Chenab': {'pm1': 1.0336, 'e_pm1': 0.0454, 'pm2': -0.5975, 'e_pm2': 0.0287, 'grad_pm1': 0.0440, 'e_grad_pm1': 0.0130, 'grad_pm2': -0.0213, 'e_grad_pm2': 0.0084},
                    'Elqui': {'pm1': 0.5584, 'e_pm1': 0.0606, 'pm2': -0.0280, 'e_pm2': 0.0491, 'grad_pm1': -0.0270, 'e_grad_pm1': 0.0199, 'grad_pm2': -0.0433, 'e_grad_pm2': 0.0141},
                    'Indus': {'pm1': -3.0886, 'e_pm1': 0.0319, 'pm2': 0.2053, 'e_pm2': 0.0285, 'grad_pm1': 0.0542, 'e_grad_pm1': 0.0043, 'grad_pm2': 0.0436, 'e_grad_pm2': 0.0041},
                    # 'Jhelum': {'pm1': -5.9330, 'e_pm1': 0.9163, 'pm2': -0.7612, 'e_pm2': 0.7987, 'grad_pm1': 0.0258, 'e_grad_pm1': 0.1969, 'grad_pm2': 0.0347, 'e_grad_pm2': 0.1505},
                    'Jhelum': {'pm1': -5.9330, 'e_pm1': 0.03, 'pm2': -0.7612, 'e_pm2': 0.05, 'grad_pm1': 0.0, 'e_grad_pm1': 0.0, 'grad_pm2': 0.0, 'e_grad_pm2': 0.0},
                    'Phoenix': {'pm1': -1.9439, 'e_pm1': 0.0216, 'pm2': -0.3649, 'e_pm2': 0.0227, 'grad_pm1': -0.0091, 'e_grad_pm1': 0.0062, 'grad_pm2': 0.0088, 'e_grad_pm2': 0.0068},
                    'Tucana III': {'pm1': 1.0835, 'e_pm1': 0.0311, 'pm2': -0.0260, 'e_pm2': 0.0343, 'grad_pm1': 0.1200, 'e_grad_pm1': 0.0309, 'grad_pm2': -0.0618, 'e_grad_pm2': 0.0319},
                    'Turranburra': {'pm1': 0.6922, 'e_pm1': 0.0455, 'pm2': -0.2223, 'e_pm2': 0.0436, 'grad_pm1': 0.0016, 'e_grad_pm1': 0.0159, 'grad_pm2': -0.0287, 'e_grad_pm2': 0.0138}}

stream_radec0_pms = {'Aliqa Uma': {'pmra': 0.2465, 'e_pmra': 0.0330, 'pmdec': -0.7073, 'e_pmdec': 0.0517},
                     'ATLAS': {'pmra': 0.0926, 'e_pmra': 0.0326, 'pmdec': -0.8783, 'e_pmdec': 0.0328},
                     'Chenab': {'pmra': 0.3223, 'e_pmra': 0.0365, 'pmdec': -2.4659, 'e_pmdec': 0.0434},
                     'Elqui': {'pmra': 0.1311, 'e_pmra': 0.0387, 'pmdec': -0.3278, 'e_pmdec': 0.0923},
                     'Phoenix': {'pmra': 2.7572, 'e_pmra': 0.0217, 'pmdec': -0.0521, 'e_pmdec': 0.0222},
                     'Tucana III': {'pmra': -0.0995, 'e_pmra': 0.0390, 'pmdec': -1.6377, 'e_pmdec': 0.0373},
                     'Turranburra': {'pmra': 0.4348, 'e_pmra': 0.0386, 'pmdec': -0.8875, 'e_pmdec': 0.0426}}


stream_phi120_pms = {'Aliqa Uma': {'pm1': -0.6634, 'pm2': -0.3479},
                     'ATLAS': {'pm1': -0.5586, 'pm2': -0.6841},
                     'Chenab': {'pm1': 2.1318, 'pm2': -1.2805},
                     'Elqui': {'pm1': -0.2986, 'pm2': -0.1883},
                     'Phoenix': {'pm1': -0.9694, 'pm2': -2.5817},
                     'Indus': {'pm1': -6.33, 'pm2': -1.34},
                     'Jhelum': {'pm1': -8.04, 'pm2': -3.98},
                     'Tucana III': {'pm1': 0.2048, 'pm2': -1.6279},
                     'Turranburra': {'pm1': -0.8193, 'pm2': -0.5528}}

stream_peri_apo = {'Aliqa Uma': (15.82, 46.04),
                   'ATLAS': (14.62, 47.64),
                   'Chenab': (33.09, 81.53),
                   'Elqui': (7.25, 66.37),
                   'Phoenix': (11.40, 17.33),
                   'Indus': (11.63, 19.25),
                   'Jhelum': (8.83, 34.17),
                   'Jet': (8.94, 38.98)}

def select_cutout(data, extent):
    """ extent: first two in array are RA, second two are DEC """
    data = data[(data['RA'] > extent[0]) & (data['RA'] < extent[1]) & (data['DEC'] > extent[2]) & (data['DEC'] < extent[3])]


def select_stream_cutout(data, stream, extent):
    
    R = stream_matrices[stream]
    phi1, phi2 = phi12_rotmat(data['RA'], data['DEC'], R)

    new_dtype = np.dtype(data.dtype.descr + [('PHI1', '>f8'), ('PHI2', '>f8')])
    new_data = np.zeros(data.shape, dtype=new_dtype)
    for descr in data.dtype.descr:
        new_data[descr[0]] = data[descr[0]]

    new_data['PHI1'] = phi1
    new_data['PHI2'] = phi2

    data = new_data
    data = data[(data['PHI1'] > extent[0]) & (data['PHI1'] < extent[1]) & (data['PHI2'] > extent[2]) & (data['PHI2'] < extent[3])]
    return data

def spherical_harmonic_background_fit(masked_array, lmax=3):
    """Fit and subtract background using spherical harmonics expansion while maintaining masks.
    
    Args:
        masked_array: Masked HEALPix array
        lmax: Maximum l value for spherical harmonics expansion
        
    Returns:
        residual_data: Masked array with background subtracted
        background_model: Masked array containing the spherical harmonic background model
    """
    nside = hp.get_nside(masked_array)
    
    # Get original alm (up to lmax)
    alm_original = hp.map2alm(masked_array, lmax=lmax)
    
    # Create background model (this will be unmasked)
    full_background = hp.alm2map(alm_original, nside, verbose=False)
    
    # Convert to masked array using original mask
    background_model = np.ma.array(full_background, mask=masked_array.mask, fill_value=hp.UNSEEN)
    
    # Subtract background model from data (preserves mask)
    residual_data = masked_array - background_model
    
    return residual_data, background_model

def movie_maker(data, mu_start=15, mu_end=18, mu_step=0.5, save=False, show=True, age=11., z=0.0007, sigma=0, title='Delve', gmag_title='gmag', rmag_title='rmag', ra_title='RA', dec_title='Dec', cmap=plt.cm.Greys, percentile_cut = [5, 95]):
    font_properties = {'family': 'serif', 'weight': 'bold', 'size': 14}
    arange = np.arange(mu_start, mu_end + mu_step, mu_step)
    for mu in arange:
        selector = select_isochrone(data[gmag_title], data[rmag_title], iso_params=[mu, age, z], gmin=(3+mu))
        iso_selection = data[selector]

        # RA/Dec plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ra = iso_selection[ra_title]
        dec = iso_selection[dec_title]
            
        pix = hp.ang2pix(512, ra, dec, lonlat=True) #gives pixel ids - healpix ids of each object (resolution-dependent)
        #get a list of healpix
        upix, counts = np.unique(pix, return_counts=True) #upix is unique pixels
        pix512 = (np.ones(hp.nside2npix(nside=512)) * hp.UNSEEN)
        pix512[upix] = counts
        min1, max1 = np.percentile(pix512[~(pix512 == hp.UNSEEN)], [5,95])
        mask_sel = (pix512 == hp.UNSEEN) | (pix512 < min1 -1) | (pix512 > max1)
        masked_array_here = np.ma.array(pix512, mask=mask_sel, fill_value=hp.UNSEEN)

        sp = skyproj.McBrydeSkyproj(ax=ax1)
        valid = masked_array_here[~(masked_array_here.mask | (masked_array_here == hp.UNSEEN))]
        vmin, vmax = np.percentile(valid, percentile_cut)
        smooth1 = hp.smoothing(masked_array_here.filled(0), np.radians(sigma))
        thing1 = sp.draw_hpxmap(smooth1, cmap=cmap, vmin=vmin, vmax=vmax) # lon and lat range
        sp.ax.set_title(f'{title}, m-M = {mu:.1f}', fontsize=16, fontdict=font_properties, y=1.05, pad=15)
        sp.ax.set_xlabel('RA', fontsize=10, fontdict=font_properties)
        sp.ax.set_ylabel('Dec', fontsize=10, fontdict=font_properties)
        sp.draw_inset_colorbar(loc=4)
        
        #Color/Magnitude Plot
        box = mkpol(mu=mu, age=age, z=z, dmu=0.5, C=[0.04, 0.04], E=4., clip=3)
        hist = ax2.hist2d(
            (data[gmag_title] - data[rmag_title]),
            data[gmag_title],
            bins=(100, 100),
            cmap=cmap
        )
        #ax2.set_title(f'CMD', fontsize=14)
        ax2.set_xlabel('g-r', fontsize=10, fontdict=font_properties)
        ax2.set_ylabel('g', fontsize=10, fontdict=font_properties)
        ax2.invert_yaxis()

        cbar = plt.colorbar(hist[3], ax=ax2)
        cbar.set_label('N stars (10000 bins)', fontsize=12, fontdict=font_properties)
        x, y = zip(*box)
        x = list(x) + [x[0]]  # Close the polygon
        y = list(y) + [y[0]]
        
        ax2.plot(x, y+mu, marker=None, markersize=20, color='red')
        ax2.fill(x, y+mu, color='lightblue', alpha=0.2)
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{mu}_slice.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            
def get_filter_splines(age, mu, z, abs_mag_min=2.9, app_mag_max = 23.5, color_min=0, color_max=1, dmu=0.5, C=[0.05, 0.1], E=2.):
    from scipy.interpolate import interp1d
    iso = isochrone_factory('Dotter2016', survey='DES', age=age, distance_modulus=mu, z=z)
    err = lambda x: 0.0010908679647672335 + np.exp((x - 27.091072029215375) / 1.0904624484538419)
    gsel = (iso.mag > abs_mag_min) & (iso.mag + mu < app_mag_max)
    color =iso.color[gsel]
    mag = iso.mag[gsel]
    # Spread in magnitude     
    mnear = mag + mu - dmu / 2.
    mfar = mag + mu + dmu / 2.
    color_far = np.clip(color + E * err(mfar) + C[1], color_min, color_max)
    color_near = np.clip(color - E * err(mnear) - C[0], color_min,color_max)
    spline_far = interp1d(mag + mu, color_far , bounds_error=False, fill_value=np.nan)
    spline_near = interp1d(mag + mu, color_near, bounds_error=False, fill_value=np.nan)
    return spline_near, spline_far

def filter_data(color, mag, age, mu, z):
    sp_near, sp_far = get_filter_splines(age=age, mu = mu, z=z)
    near_vals = sp_near(mag)
    far_vals =  sp_far(mag)
    sel = (color > near_vals) & (color < far_vals)
    return sel #Sel is the boolean array

def astropy_table_maker(data, mu_start=15, mu_end=18, mu_step=0.5, age=11., z=0.0007, gmag_title='gmag', rmag_title='rmag', ra_title='RA', dec_title='Dec'):
    '''Makes a table with counts for each mu value in each healpixel'''
    ids = np.arange(1, hp.nside2npix(nside=512) + 1)
    table = Table()
    table['Healpix Id'] = ids
    
    arange = np.arange(mu_start, mu_end + mu_step, mu_step)
    for mu in arange:
        
        selector = filter_data(color=(data[gmag_title]-data[rmag_title]), mag=data[gmag_title], age=age, mu=mu, z=z)
        iso_selection = data[selector]
        
        ra = iso_selection[ra_title]
        dec = iso_selection[dec_title]
            
        pix = hp.ang2pix(512, ra, dec, lonlat=True)
       
        upix, counts = np.unique(pix, return_counts=True)
        pix512 = np.zeros(hp.nside2npix(nside=512))
        pix512[upix] = counts
        
        table[f'Mu = {mu}'] = pix512
    
    ra = data[ra_title]
    dec = data[dec_title]
            
    pix = hp.ang2pix(512, ra, dec, lonlat=True)
       
    upix, counts = np.unique(pix, return_counts=True)
    pix512 = np.zeros(hp.nside2npix(nside=512))
    pix512[upix] = counts
    table['Totals'] = pix512
    #table = table[table['Totals'] != 0]
    return table

def table_maker(data, mu_start=15, mu_end=18, mu_step=0.5, age=11., z=0.0007, gmag_title='gmag', rmag_title='rmag', ra_title='RA', dec_title='Dec', nside=512):
    
    ra = data[ra_title]
    dec = data[dec_title]
    mu_arange = np.arange(mu_start, mu_end + mu_step, mu_step) # mu range
    
    pix = hp.ang2pix(512, ra, dec, lonlat=True) # all pixel ids in range
    upix = np.unique(pix, return_counts=False) # all unique pixel ids in range
    
    out_col_list = [(f'pix{nside}', 'int')] # this column will hold all the ids
    out_col_list += [f'{mu:.1f}'.replace('.','p') for mu in mu_arange] # now it's all the columns
    
    dtype_list = [(name, 'int') for name in out_col_list] # list of 'int' for each column
    hpx_array = np.recarray(shape=len(upix), dtype=dtype_list) # Make hpx array
    hpx_array.fill(0) # Make the default count value zero
    hpx_array[f'pix{nside}'] = upix # Set the hpx ids in the first column
    
    for mu in mu_arange:
        
        selector = filter_data(color=(data[gmag_title]-data[rmag_title]), mag=data[gmag_title], age=age, mu=mu, z=z) # Make matched-filter selection
        upix_sel, counts_sel = np.unique(pix[selector], return_counts=True) # Get the unique hpx ids and counts for the filter selection

        col_name = f'{mu:.1f}'.replace('.','p') # Get the column name
        hpx_array[col_name][np.searchsorted(upix, upix_sel)] = counts_sel # As far as I can tell, this works, but ask Peter
          
    return hpx_array

def object2mutable(vertices, hats_catalog, args):
    '''
    Makes table of counts at each mu value across hpx ids
    
    vertices: (ra1, ra2, dec1, dec2)
    hats_catalog: data to work with
    '''
    ra1 = vertices[0]
    ra2 = vertices[1]
    dec1 = vertices[2]
    dec2 = vertices[3]
    catalog_box = hats_catalog.box_search((ra1,ra2),(dec1,dec2)).compute()
    mu_start, mu_end, mu_step, age, z, gmag, rmag, ra, dec, nside = args
    small_array = table_maker(data=catalog_box, mu_start=mu_start, mu_end=mu_end, mu_step=mu_step, age=age, z=z, gmag_title=gmag, rmag_title=rmag, ra_title=ra, dec_title=dec, nside=nside)
    
    return small_array

def make_it_big(hats_catalog, ra_step, dec_step, args):
    '''
    Makes a mu table across all RA and Dec Values given a hats catalog.
    '''
    ra_list = np.arange(0, 360, ra_step)
    dec_list = np.arange(-90, 90, dec_step)
    small_array_list = []
    for ra in ra_list:
        for dec in dec_list:
            vertices = (ra, ra+ra_step, dec, dec+dec_step)
            small_array = object2mutable(vertices=vertices, hats_catalog=hats_catalog, args=args)
            small_array_list.append(small_array)
            print(f'Loading RA: {vertices[0]} -> {vertices[1]}, DEC: {vertices[2]} -> {vertices[3]}          ', end='\r')
    big_array = np.concatenate(small_array_list)
    return big_array

def make_gif(infiles, outfile=None, delay=40, queue='local'):
    print("Making movie...")
    infiles = np.atleast_1d(infiles)
    if not len(infiles):
        msg = "No input files found"
        raise ValueError(msg)

    infiles = ' '.join(infiles)
    if not outfile:
        outfile = infiles[0].replace('.png', '.gif')
    cmd = 'convert -delay %i -quality 100 %s %s' % (delay, infiles, outfile)
    if queue != 'local':
        cmd = 'csub -q %s ' % (queue) + cmd
    print(cmd)
    subprocess.check_call(cmd, shell=True)