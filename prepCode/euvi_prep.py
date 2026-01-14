"""
Module for functions related to EUVI processing that are 
called by secchi_prep. Largely a port of the corresponding
IDL routines and we have kept names matching and indicated
what portions have been left out to facilitate comparison to
the other version. 

External calls:
    scc_funs, cor_prep

"""

import numpy as np
import os
import sys
from astropy.io import fits
from scc_funs import scc_sebip
import datetime
from scipy.interpolate import griddata
from cor_prep import get_calfac, get_calimg

#|---------------------------|
#|--- Check Filter Status ---|
#|---------------------------|
def euvi_get_normal(hdr):
    """
    Function to grab the filter factor from an EUVI header
    
    Input:
        hdr: an EUVI header
    
    Output:
        filter_factor: a value based on 'WAVELNTH' and 'FILTER'

    """
    # |--- Check if a known wavelength ---|
    if hdr['WAVELNTH'] == 171:
        filtDict = {'OPEN':1, 'S1':0.49, 'S2':0.49, 'DBL':0.41}
    elif hdr['WAVELNTH'] == 195:
        filtDict = {'OPEN':1, 'S1':0.49, 'S2':0.49, 'DBL':0.41}
    elif hdr['WAVELNTH'] == 284:
        filtDict = {'OPEN':1, 'S1':0.33, 'S2':0.33, 'DBL':0.24}
    elif hdr['WAVELNTH'] == 304:
        filtDict = {'OPEN':1, 'S1':0.29, 'S2':0.29, 'DBL':0.22}
    else:
        sys.exit('WAVELNTH not found, exiting euvi_get_normal')
    
    # |--- Pull filter val for that wavelength ---|
    if hdr['FILTER'] in filtDict:
        filter_factor = filtDict[hdr['FILTER']]
    else:
        sys.exit('FILTER not found')
    return filter_factor


#|-------------------------|
#|--- Main Prep Routine ---|
#|-------------------------|
def euvi_correction(img, hdr, prepDir, sebipOff=False, exptimeOff=False, biasOff=False, normalOff=False, dn2pOff=False, calImgOff=False):
    """
    Main wrapper function for EUVI image correction
    
    Input:
        img: an EUVI image
    
        hdr: the corresponding header
    
        prepDir: the path where the extra files needed for prep are stored
       
    Optional Input:   
        sebipOff: flag to turn off the seb ip correction (defaults false)
        
        exptimeOff: flag to turn off the exposure time correction (defaults false)
    
        biasOff: flag to turn off the bias correction (defaults false)
    
        normalOff: flag to turn off normalizing to open filter (defaults false)

        dn2pOff: flag to turn off the dN to photons correction (defaults false)

        calImgOff: flag to turn off the calibration image correction (defaults false)

    Output:
        img: the calibrated image 
        
        hdr: the updated header for the calbrated image 

    """
    #|------------------|
    #|--- EUVI Check ---|
    #|------------------|
    if hdr['detector'] != 'EUVI':
        sys.exit('Passed non EUVI obs to euvi_correcton')
    hdr['history'] = 'Applied port of euvi_correction.pro'
    
    #|-------------------------|
    #|--- SEB IP correction ---|
    #|-------------------------|
    if not sebipOff:
        img, hdr, sebipFlag = scc_sebip(img, hdr)
    
    #|--------------------------------|
    #|--- Exposure time correction ---|
    #|--------------------------------|
    exptime = 1.0    
    if not exptimeOff:
        exptime = float(hdr['exptime'])
        if exptime != 1.:
            hdr['history'] = 'Exposure Normalized to 1 Second from ' + str(exptime)
            
    #|-----------------------|
    #|--- Bias correction ---|
    #|-----------------------|
    biasmean = 0.        
    if not biasOff:
        biasmean = float(hdr['biasmean'])
        if biasmean != 0: 
            hdr['history'] = 'Bias subtracted '+ str(biasmean)
        hdr['OFFSETCR'] = biasmean
    
    #|--------------------------------|
    #|--- Filter/normal correction ---|
    #|--------------------------------|
    normal = 1.
    if not normalOff:
        normal = euvi_get_normal(hdr)
        if normal != 0: 
            hdr['history'] = 'Normalized to Open Filter Position ' + str(normal)
    
    #|--------------------------------|
    #|--- DN to photons correction ---|
    #|--------------------------------|
    photons_dn = 1.0
    if not dn2pOff:
        photons_dn = get_calfac(hdr)
        hdr['history'] = 'Photometric Correction - DN to Detected Photons'+ str(photons_dn)
        
    #|------------------------------------|
    #|--- Calibration image correction ---|
    #|------------------------------------|
    calimg = 1.0
    if not calImgOff:
        calimg, hdr = get_calimg(hdr, prepDir+'calimg/')
  
    #|-------------------------|
    #|--- Applu corrections ---|
    #|-------------------------|
    img = ((img - biasmean)*photons_dn/(exptime*normal))*calimg
        
    return img, hdr

#|-------------------------|
#|--- Main Prep Wrapper ---|
#|-------------------------|
def euvi_prep(im, hdr, prepDir, pointingOff=True, calibrateOff=False):
    """
    Main wrapper for prepping EUVI images. IDL has more options but
    this is essentiall a call to euvi_correction
    
    Input:
        img: an EUVI image
    
        hdr: the corresponding header
    
        prepDir: the path where the extra files needed for prep are stored
    
    Optional Input:
        pointingOff: flag to turn off pointing correction (not implemented)
    
        calibrateOff: flag to turn off calibration (defaults false)
                        -> this makes this function simply pass input files back as is

    Output:
        im: the calibrated image
    
        hdr: the corresponding header 
    
    Notes:
        The EUVI has been very minimally ported at this point


    """
    #|------------------|
    #|--- EUVI Check ---|
    #|------------------|
    if hdr['detector'] != 'EUVI':
        sys.exit('Calibration for EUVI DETECTOR only')
    
        
    # Pointing already checked in the trimming 
    
    # Not doing CRs or missing
    
    #|-----------------------------------|
    #|--- Send to calibration routine ---|
    #|-----------------------------------|
    if not calibrateOff:
        im, hdr = euvi_correction(im, hdr, prepDir)
        
    # Skipping missing
    # Not doing dejitter
    # Not rolling north up
    # No mask
    # No color table
    
    return im, hdr