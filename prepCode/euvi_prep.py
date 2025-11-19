"""
Module for functions related to EUVI processing that are 
called by secchi_prep. Largely a port of the corresponding
IDL routines and we have kept names matching and indicated
what portions have been left out to facilitate comparison to
the other version. 

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
    
    if hdr['FILTER'] in filtDict:
        filter_factor = filtDict[hdr['FILTER']]
    else:
        sys.exit('FILTER not found')
    return filter_factor


#|-------------------------|
#|--- Main Prep Routine ---|
#|-------------------------|
def euvi_correction(img, hdr, prepDir, sebipOff=False, exptimeOff=False, biasOff=False, normalOff=False, dn2pOff=False, calImgOff=False):
    if hdr['detector'] != 'EUVI':
        sys.exit('Passed non EUVI obs to euvi_correcton')
    hdr['history'] = 'Applied port of euvi_correction.pro'
    
    if not sebipOff:
        img, hdr, sebipFlag = scc_sebip(img, hdr)
    
    exptime = 1.0    
    if not exptimeOff:
        exptime = float(hdr['exptime'])
        if exptime != 1.:
            hdr['history'] = 'Exposure Normalized to 1 Second from ' + str(exptime)
            
    biasmean = 0.        
    if not biasOff:
        biasmean = float(hdr['biasmean'])
        if biasmean != 0: 
            hdr['history'] = 'Bias subtracted '+ str(biasmean)
        hdr['OFFSETCR'] = biasmean
    
    normal = 1.
    if not normalOff:
        normal = euvi_get_normal(hdr)
        if normal != 0: 
            hdr['history'] = 'Normalized to Open Filter Position ' + str(normal)
    
    photons_dn = 1.0
    if not dn2pOff:
        photons_dn = get_calfac(hdr)
        hdr['history'] = 'Photometric Correction - DN to Detected Photons'+ str(photons_dn)
        
    calimg = 1.0
    if not calImgOff:
        calimg, hdr = get_calimg(hdr, prepDir+'calimg/')
  
    # Apply correction
    img = ((img - biasmean)*photons_dn/(exptime*normal))*calimg
        
    return img, hdr

#|-------------------------|
#|--- Main Prep Wrapper ---|
#|-------------------------|
def euvi_prep(im, hdr, prepDir, pointingOff=True, calibrateOff=False):
    if hdr['detector'] != 'EUVI':
        sys.exit('Calibration for EUVI DETECTOR only')
    
        
    # Pointing already checked in the trimming 
    
    # Not doing CRs or missing
    
    # Correction DN
    if not calibrateOff:
        im, hdr = euvi_correction(im, hdr, prepDir)
        
    # Skipping missing
    # Not doing dejitter
    # Not rolling north up
    # No mask
    # No color table
    
    return im, hdr