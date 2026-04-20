#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:28:59 2026

This file was written by Naïm Teriitehau-Martin. For any issues, please contact : naimteriitehau@gmail.com 

"""

###### CARM_VIS ######
''' THIS IS STILL A WORK IN PROGRESS as of 20/04/2026'''

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

oset = '20:50'    # Relative orders that are analyzed (Here we keep only the 20th to 50th available orders. For CARM_VIS, the absolute orders start at 118 [low wavlengths] and end at 58 [high wavelengths])
iset='400:3600'    # Keep the pixels between X and Y ; Do not analyze the ones outside of that range -- Chosen arbitrarily and can be changed by the user (goes up to 0:4000)


# Convert FWHM resolution to sigma
ip_guess = {'s' : c.to_value(u.km/u.s)/(94_600*2*np.sqrt(2*np.log(2)))}  # speed of light [km/s] divided by (spectrograph_resolution x factor) -- Assuming Gaussian IP for factor ; FWHM is defined in terms of speed dv [km/s] (which is ~= delta(ln(wavelen)))

# Location of CARMENES Spectrograph -- Obtained from the data headers (hdr keys 'HIERARCH CAHA TEL GEOLAT' and 'HIERARCH CAHA TEL GEOLON' and 'HIERARCH CAHA TEL GEOELEV')
location = carmenes = EarthLocation.from_geodetic(lat=37.2236*u.deg, lon=-2.54625*u.deg, height=2168.*u.m)


def Spectrum(filename='', order=None, targ=None):
    hdu = fits.open(filename, ignore_blank=True)
    hdr = hdu[0].header
    
    dateobs = hdr.get('DATE-OBS')   # UTC datetime at observation start (ISOT format)
    #exptime = hdr.get('EXPTIME')    # Exposure time in seconds
    ''' HERE WE GET TMEAN FROM HEADERS (so it is an absolute value, also less reliable because it is in header)
    I HAVE TO FIND A WAY TO GET IT FROM THE serval DIR (which is the same as the CARMENES_templates btw) -- 
    The problem being that I have to somehow call that file, which implies the user has the file and it is in the exact same path as how I call it
    
    ---> In the meantime, just check if there is a file in the given path (path will automatically be generated, like I did in SCRIPT_checks_if_serval_in_hdr.py
         and assuming that it is only used by people in the IAG intranet? And if no file found (ex. path not exist because used by external user) then just use hdr TMEAN?'''
    exptime_tmean = hdr.get('HIERARCH CARACAL TMEAN') * u.s  # Flux-weighted midpoint of exposure (more accurate than the exptime)

    # If target not specified while calling Spectrum() : Define target from data file header info
    ra = hdr.get('RA', np.nan)     # RightAscension of target [degrees]
    dec = hdr.get('DEC', np.nan)    # Declination of target [degrees]
    dec = (dec+90) % 180 - 90   #Declination needs to be between -90 and 90 deg, but some SERVAL tpl headers give values way bigger than that so we fix it
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
    err_spec = hdu['SIG'].data    # Error estimate for flux
    
    
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
    '''
    # Check if template already convolved with the IP (If yes, will need to apply a different convolution in utils/model.py) ;
    # IMPORTANT REMARK : Currently it only checks if template comes from SERVAL (no idea how it could be generalized to other sources of templates)
    hdr = fits.open(tplname, ignore_blank=True)[0].header
    tpl_is_serval = 'HIERARCH SERVAL COADD NUM' in hdr    # all SERVAL templates contain this key to indicate number of spectra used for coadd, but none of the data files contain it
    
    global tpl_IP_isconv    # define as global so it can be called from other files
    tpl_IP_isconv = tpl_is_serval    # currently it only actually checks if the template comes from SERVAL
    

    if tplname.endswith('.fits'):
        try:
            pixel, wavelen_k, spec_k, err_spec_k, flag_pixel, bjd, berv = Spectrum(tplname, order=order, targ=targ)
            
            #SERVAL templates store flux and natural log of vacuum wavelen as the "knot positions of uniform B-spline" 
            #so we need to apply cubic spline to get data between the knots, and convert to linear wavelen
            if tpl_is_serval:
                # CubicSpline interpolation to artificially improve sampling in the template
                wavelen = np.linspace(wavelen_k[0], wavelen_k[-1], 4*wavelen_k.size)    # Interpolate with 4 times as many points
                spec = CubicSpline(wavelen_k, spec_k, bc_type='natural')(wavelen)   
                wavelen = np.exp(wavelen) # Convert log knots to linear knots
                
            # tpl_R = 91_000
        except:
            print('\x1b[0;31;40mError : Barycentric correction for the selected .fits template has failed. \x1b[0m')
            exit()
            
    elif tplname.endswith('.all'):    # for PEPSI templates
        try:
            hdu = fits.open(tplname)
            wavelen = hdu[1].data.field('Arg')  # Wavelengths for template spectrum
            spec = hdu[1].data.field('Fun') # Flux for template spectrum
            wavelen = airtovac(wavelen)    # convert the wavelength values from air to vacuum. Unlike SERVAL templates which already take this into account
            
            '''
            Check if template is from serval using something like the lines below?
            there was Also something like if the resolution of the tpl is high enough then do not convolve with IP???
                
            global tpl_has_IP
            # tpl_R = 200_000
            tpl_has_IP = True
            spec.tpl_has_IP = True
            '''
        except:
            print('\x1b[0;31;40mError : Barycentric correction for the selected .all template has failed. \x1b[0m')
            exit()
            
    else:
        print('\x1b[0;31;40m' +'Error: Template format is not known for the selected instrument. \nPlease select a .fits file or .all file'+ '\x1b[0m')    
        exit()
        
    #flux
    #spec = (wavelen, flux, R) reutrn dict ? would need to change the other inst files and viper.py
    return wavelen, spec



# If using a cell (default = No cell)
def FTS(ftsname='None', dv=100):
    '''FTSFits() reads FTS of the cell and obtains wavenumber and flux (f) from data headers
    Converts wavenumbers [cm] to wavelengths (w) [angstrom], also inverts w and f arrays ( [::-1] ) so that wavelengths are in ascending order
    resample() returns w, f, uj= np.arange(ln(w)) with step dv/c, iod_j=flux interpolated from uj
    '''
    return resample(*FTSfits(ftsname), dv=dv)



# If we want to create a template
def write_fits(wtpl_all, tpl_all, e_all, list_files, file_out):

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




