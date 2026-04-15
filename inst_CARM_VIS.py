#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:28:59 2026

This file was written by Naïm Teriitehau-Martin. For any issues, please contact : naimteriitehau@gmail.com 

"""

###### CARM_VIS ######
''' THIS IS A WORK IN PROGRESS '''

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.constants import c
from scipy.interpolate import CubicSpline

# from inst.template import read_tpl   #### DO NOT NEED IF APPLY BARYCENTRIC CORRECTION IN THIS FILE?
from inst.FTS_resample import resample, FTSfits 
from inst.airtovac import airtovac
import timeit

oset = '20:50'    # Relative orders that are analyzed (Here we keep only the 20th to 49th available orders, ) (for CARM_VIS, the absolute orders start at 118 [low wavlengths] and end at 58 [high wavelengths])
iset='400:3600'    # Keep the pixels between X and Y ; Do not analyze the ones outside of that range -- Chosen arbitrarily and can be changed by the user

# Convert FWHM resolution to sigma
''' The IP is currently set to a very low number, because CARMENES usually uses SERVAL templates, and those have the spectrum 
of the star already convoluted with the IP. But the viper code tries to do another convolution, which we do not want.
'''
#ip_guess = {'s' : 300_000/(39_594_600*2*np.sqrt(2*np.log(2)))}    # speed of light divided by (spectrograph_resolution x factor) -- Assuming Gaussian IP, FWHM defined in speed v
ip_guess = {'s' : 300_000/(94_600*2*np.sqrt(2*np.log(2)))}  # Regular ip_guess as it should be

# Location of CARMENES Spectrograph -- Obtained from the data headers (hdr keys 'HIERARCH CAHA TEL GEOLAT' and 'HIERARCH CAHA TEL GEOLON' and 'HIERARCH CAHA TEL GEOELEV')
location = carmenes = EarthLocation.from_geodetic(lat=37.2236*u.deg, lon=-2.54625*u.deg, height=2168.*u.m)


def Spectrum(filename='', order=None, targ=None):
    hdu = fits.open(filename, ignore_blank=True)
    hdr = hdu[0].header
    
    dateobs = hdr.get('DATE-OBS')   # UTC datetime at observation start (ISOT format)
    #exptime = hdr.get('EXPTIME')    # Exposure time in seconds
    ''' HERE WE GET TMEAN FROM HEADERS (so it is an absolute value, also less reliable because it is in header)
    I HAVE TO FIND A WAY TO GET IT FROM THE serval DIR (which is the same as the CARMENES_templates btw) -- 
    The problem being that I have to somehow call that file, which implies the user has the file and it is in the exact same path as how I call it'''
    exptime_tmean = hdr.get('HIERARCH CARACAL TMEAN') * u.s  # Flux-weighted midpoint of exposure (more accurate than exptime)

    # If target not specified while calling Spectrum() : Define target from data file header info
    ra = hdr.get('RA', np.nan)     # RightAscension of target [degrees]
    dec = hdr.get('DEC', np.nan)    # Declination of target [degrees]
    '''The line below seems to be necessary for some SERVAL templates'''
    dec = (dec+90) % 180 - 90   #Declination needs to be between -90 and 90 deg, but headers give values way past that
    targdrs = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    if not targ: targ = targdrs
    
    # Apply barycentric RV correction halfway through exposure (offers the least error?)
    ''' for midtime : instead of using exptime/2 (can cause errors ex if there are clouds then flux info can be wrong etc etc can cause shift of something like 5mins which is like 20m/s) 
        use TMMEAN INSTEAD? (CF. /data/svn/SERVAL for info) from brv.dat files'''
    #midtime = Time(dateobs, format='isot', scale='utc') + exptime/2 * u.s #REPLACE THIS *tmean
    midtime = Time(dateobs, format='isot', scale='utc') + exptime_tmean
    berv = targ.radial_velocity_correction(obstime=midtime, location=carmenes)  #Barycentric Earth RV : This is the RV correction to apply
    berv = berv.to_value(u.km/u.s)
    bjd = midtime.tdb    # Convert midtime scale from utc to Barycentric Dynamical Time
    
    # Read file data to obtain spec, wavelen, err
    spec = hdu['SPEC'].data    # Flux for spectrum
    wavelen = hdu['WAVE'].data    # Vacuum wavelength for each pixel [angstrom]
    err_spec = hdu['SIG'].data    # Get error estimate for spec
    
    
    # If specific order is selected, only keep that order
    if order is not None:
        wavelen, spec, err_spec = wavelen[order], spec[order], err_spec[order]
    
    # Build pixel array and bad pixel map
    pixel = np.arange(spec.size)    # As many pixels as there are values in spec (Including bad pixels)
    flag_pixel = 1 * np.isnan(spec)  # Bad pixels (spec value undefined) have a value of 1
    
    return pixel, wavelen, spec, err_spec, flag_pixel, bjd, berv



def Tpl(tplname, order=None, targ=None):
    '''
    Tpl should return barycentric corrected wavelengths.
    read_tpl(tplname) tries to call Spectrum(tplname=template_file) from the indicated inst file and apply a
    barycentric correction to the returned wavelen (if tplname doesn't end with "_tpl")
    
    Currently this only works with templates that store wavelen data as natural log wavelen (like the ones produced by serval)
    Because the wavelen are converted into linear wavelen via np.exp()
    '''
    
    
    '''
    SERVAL templates store spec and natural log of vacuum wavelen as the "knot positions of uniform B-spline"
    so we need to convert to linear wavelen then apply cubic spline to get data between the knots
    '''
    if tplname.endswith('.fits'):   #The only type of SERVAL template I am aware of
        try:
            pixel, wavelen_k, spec_k, err_spec_k, flag_pixel, bjd, berv = Spectrum(tplname, order=order, targ=targ)
            '''SERVAL already applies barycentric correction to templates, so the line below can be removed'''
            #wavelen_k *= 1 + (berv*u.km/u.s/c).to_value('') # Apply barycentric correction
            
            # CubicSpline interpolation to artificially improve sampling in the template
            wavelen = np.linspace(wavelen_k[0], wavelen_k[-1], 4*wavelen_k.size)    # Interpolate with 4 times as many points
            spec = CubicSpline(wavelen_k, spec_k, bc_type='natural')(wavelen)   
            wavelen = np.exp(wavelen) # Convert log knots to linear knots
        except:
            print('Error : Barycentric correction for the selected .fits template has failed.')
            exit()
            
    elif tplname.endswith('.all'):    # for PEPSI templates
        try:
            hdu = fits.open(tplname)
            wavelen = hdu[1].data.field('Arg')
            spec = hdu[1].data.field('Fun')
            wavelen = airtovac(wavelen)    # convert the wavelength values from air to vacuum. Unlike SERVAL templates which already take this into account
            
        except:
            print('Error : Barycentric correction for the selected .all template has failed.')
            exit()
            
    else:
        print('\x1b[0;31;40m' +'Error: Template format is not known for the selected instrument. \nPlease select a .fits file or .all file'+ '\x1b[0m')    
        exit()
    

    
    
    
    print(f'tplname is {tplname}')
    hdr = fits.open(str(tplname), ignore_blank=True)[0].header
    def thing1():
        if any(["SERVAL COADD" in key for key in  hdr]):
            a = True
    def thing2():
        if any(["SERVAL COADD" in key for key in  list(hdr.keys())]):
            a = True
    def thing3():
        if 'HIERARCH SERVAL COADD SN010' in hdr:
            a = True
    t1 = timeit.timeit(stmt = thing1, number=100)
    t2 = timeit.timeit(stmt = thing2, number=100)
    t3 = timeit.timeit(stmt = thing3, number=100)
    print(f'\n.\n.\n {t1} WITH ANY REVERSE \n.\n.\n')
    print(f'\n.\n.\n {t2} WITH ANY \n.\n.\n')
    print(f'\n.\n.\n {t3} WITH otherthang \n.\n.\n')
    return wavelen, spec



# If using a cell (default = No cell)
def FTS(ftsname='None', dv=100):
    '''FTSFits() reads FTS of the cell and obtains wavenumber and flux (f) from data headers
    Converts wavenumbers [cm] to wavelengths (w) [angstrom], also inverts w and f arrays ( [::-1] ) so that wavelengths are in ascending order
    resample() returns w, f, uj= np.arange(ln(w)) with step dv/c, iod_j=flux interpolated from uj
    '''
    print('fts was called')
    return resample(*FTSfits(ftsname), dv=dv)



# If we want to create a template
def write_fits(wtpl_all, tpl_all, e_all, list_files, file_out):
    print(' nuh uh write fit')
    
    
    # Get data from header of the first fits file
    file_in = list_files[0]
    hdu = fits.open(file_in, ignore_blank=True)
    f = hdu[0].data
    
    # Write template data to the file
    for o in range(1, len(f), 1): 
        if o in tpl_all:
            f[o] = tpl_all[o]
        else:
            f[o] = np.ones(len(f[o]))

    hdu.writeto(file_out+'_tpl.model', overwrite=True)




