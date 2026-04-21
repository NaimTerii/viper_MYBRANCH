#! /usr/bin/env python3
# Licensed under a GPLv3 style license - see LICENSE

import numpy as np
import sys
import os
from astropy.io import fits
from astropy.time import Time
import datetime
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.constants import c

from .template import read_tpl
from .readmultispec import readmultispec
from .airtovac import airtovac

from .FTS_resample import resample, FTSfits

# see https://github.com/mzechmeister/serval/blob/master/src/inst_FIES.py

location = lasilla = EarthLocation.from_geodetic(lat=-29.2543*u.deg, lon=-70.7346*u.deg, height=2335*u.m)

oset = '8:24'
iset = '380:1700'

ip_guess = {'s': 300_000/80_000/ (2*np.sqrt(2*np.log(2))) }   # convert FHWM resolution to sigma

def Spectrum(filename='', order=None, targ=None):
    hdu = fits.open(filename, ignore_blank=True)
    hdr = hdu[0].header

    dateobs = hdr['DATE-OBS']
    exptime = hdr['EXPOSURE']
    
    ra = hdr.get('RA', np.nan)                        
    de = hdr.get('DEC', np.nan)
    
    offs = de.startswith('-')
    ra = ra.split(':')
    de = de[offs:].split(':')
    ra = (float(ra[0]) + float(ra[1])/60 + float(ra[2])/3600) * 15
    de = float(de[0]) + float(de[1])/60 + float(de[2])/3600
    if offs: de *= -1

    targdrs = SkyCoord(ra=ra*u.deg, dec=de*u.deg)
    if not targ: targ = targdrs
    midtime = Time(dateobs, format='isot', scale='utc') + exptime/2. * u.s
 
    berv = targ.radial_velocity_correction(obstime=midtime, location=lasilla)
    berv = berv.to(u.km/u.s).value
    bjd = midtime.tdb

    if len(hdu) == 1:
        # for spectra reduced with the IRAF package
        spec = hdu[0].data
        spec /= np.nanmean(spec)
        gg = readmultispec(filename, reform=True, quiet=True)
        wave = gg['wavelen']
        wave = airtovac(wave)
    else:
        # for spectra reduced with the ceres+ pipeline
        wave = hdu[0].data
        spec = hdu[3].data	# de-blazed spectrum
       #spec = hdu[5].data	# continuum-normalized spectrum
    
    if order is not None:
         wave, spec= wave[order], spec[order]
     
    if len(hdu) == 1:     
        wave = wave[::-1]
        spec = spec[::-1]

    pixel = np.arange(spec.size) 
    err = np.ones(spec.size)*0.1
    flag_pixel = 1 * np.isnan(spec) # bad pixel map
    
    # for HARPS
    flag_pixel[(5300<wave) & (wave<5343)] |= 256

    return pixel, wave, spec, err, flag_pixel, bjd, berv


def Tpl(tplname, order=None, targ=None):
    '''Tpl should return barycentric corrected wavelengths'''
    
    wave, spec = read_tpl(tplname, inst=os.path.basename(__file__), order=order, targ=targ) 

    return wave, spec

    
def FTS(ftsname='lib/PlatoSpec/FTS_PlatoSpec.fits', dv=100):

    w, f = FTSfits(ftsname)
    f = f[::-1]

    return resample(w, f, dv=dv)


def write_fits(wtpl_all, tpl_all, e_all, list_files, file_out):

    file_in = list_files[0]

    # copy header from first fits file 
    hdu = fits.open(file_in, ignore_blank=True)[0]
    f = hdu.data

    # write the template data to the file
    for order in range(1, len(f), 1): 
        if order in tpl_all:
            f[order] = tpl_all[order]
        else:
            f[order] = np.ones(len(f[order]))

    hdu.writeto(file_out+'_tpl.model', overwrite=True)  
  #  hdu.close()  

