"""
Module for functions related to LASCO C2/C3 processing.
Largely a port of the corresponding IDL routines and we 
have kept names matching and indicatedwhat portions have
been left out to facilitate comparison to the other version. 

"""
import numpy as np
import sunpy.map
import sys, os
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.time import parse_time

#|---------------------------|
#|--- Get solar ephemeris ---|
#|---------------------------|
def get_solar_ephem(yymmdd, isSOHO=False):
    dte = parse_time(yymmdd).utc
    j2000 = parse_time('2000/01/01').utc
    n = dte.mjd - j2000.mjd
    lon = 280.460 + 0.9856474 * n
    lon = lon % 360
    g = 357.528 + 0.9856003 * n
    g = g * np.pi / 180.
    dist = 1.00014 - 0.01672 * np.cos(g) - 0.00014 * np.cos(2*g)
    if isSOHO:
        dist = dist * 0.99
    radius = 0.2666 / dist
    
    return radius, dist, lon
    
#|----------------------------------|
#|--- Get exponential (?) factor ---|
#|----------------------------------|
def get_exp_factor(hdr, efacDir):
    tel = hdr['detector'].lower()
    mjd = hdr['mid_date']
    jd = mjd + 2400000.5
    t = Time(jd, format='jd')
    dt = t.to_datetime()
    yymm = dt.strftime('%y%m%d') # idl has days so we do too despite the name    
    fn = tel + '_expfactor_'+yymm+'.dat'
    
    # Going rogue from here bc IDL is using common blocks but found the source files
    myDir = efacDir + yymm[:4] + '/'
    efac = 1 # bc if we don't find it, it's nearly one anyway
    bias = None
    if 'offset' in hdr:
        bias = hdr['offset']
    if os.path.exists(myDir+fn):
        data = np.genfromtxt(myDir+fn, dtype=str)
        if hdr['filename'] in data[:,0]:
            idx = np.where(data[:,0] == hdr['filename'])[0]
            efac = float(data[idx[0], 1])
        if type(bias) != type(None):
            bias = float(data[idx[0], 2])            
    return efac, bias

#|---------------------------------|
#|--- Get c2 calibration factor ---|
#|---------------------------------|
def c2_calfactor(hdr, nosum=False):
    filt = hdr['filter'].upper()
    polar = hdr['POLAR'].upper()
    mjd = hdr['mid_date']
    
    if filt == 'ORANGE':
        cal_factor=4.60403e-07*mjd+0.0374116
        polref=cal_factor/.25256
        if polar in ['+60DEG', '0DEG', '-60DEG', 'nd']:
            cal_factor = polref
    elif filt == 'BLUE':
        cal_factor = 0.1033
        polref=cal_factor / 0.25256
        if polar in ['+60DEG', '0DEG', '-60DEG', 'nd']:
            cal_factor = polref
    elif filt == 'DEEPRD':
        cal_factor = 0.1033
        polref=cal_factor / 0.25256
        if polar in ['+60DEG', '0DEG', '-60DEG', 'nd']:
            cal_factor = polref
        else:
            cal_factor = 0.
    elif filt in ['HALPHA', 'LENS']:
        cal_factor = 0.1055 # IDL labels this as wrong but doesn't provide a right
        polref=cal_factor / 0.25256
        if polar in ['+60DEG', '0DEG', '-60DEG', 'nd']:
            cal_factor = polref
       
    else:
        sys.exit('Unrecognized filter in c3_calfactor')
        
    if not nosum:
        if (hdr['sumcol'] > 0): cal_factor = cal_factor / hdr['sumcol']
        if (hdr['sumrow'] > 0): cal_factor = cal_factor / hdr['sumrow']
        if (hdr['lebxsum'] > 1): cal_factor = cal_factor / hdr['lebxsum']
        if (hdr['lebysum'] > 1): cal_factor = cal_factor / hdr['lebysum']
    
    cal_factor = cal_factor * 1e-10    
    return cal_factor
    
#|---------------------------------|
#|--- Get c3 calibration factor ---|
#|---------------------------------|
def c3_calfactor(hdr, nosum=False):
    filt = hdr['filter'].upper()
    polar = hdr['POLAR'].upper()
    mjd = hdr['mid_date']
    
    if filt == 'ORANGE':
        cal_factor = 0.0297
        polref = cal_factor/.25256	
        if polar == '+60DEG':	
            cal_factor=polref
        elif polar =='0DEG':	
            cal_factor=polref * 0.9648
        elif polar == '-60DEG':	
            cal_factor=polref * 1.0798
    elif filt == 'BLUE':
        cal_factor = 0.0975	
        polref = cal_factor / 0.25256
        if polar == '+60DEG':	
            cal_factor=polref
        elif polar =='0DEG':	
            cal_factor=polref * 0.9734
        elif polar == '-60DEG':	
            cal_factor=polref * 1.0613
    elif filt == 'CLEAR':
        cal_factor=7.43e-8 * (mjd - 50000) + 5.96e-3
        polref = cal_factor / 0.25256
        if polar == '+60DEG':	
            cal_factor=polref
        elif polar =='0DEG':	
            cal_factor=polref * 0.9832
        elif polar == '-60DEG':	
            cal_factor=polref * 1.0235
        elif polar == 'H_ALPHA':
            cal_factor = 1.541
    elif filt == 'DEEPRD':
        cal_factor = 0.0259	
        polref = cal_factor / 0.25256
        if polar == '+60DEG':	
            cal_factor=polref
        elif polar =='0DEG':	
            cal_factor=polref * 0.9983
        elif polar == '-60DEG':	
            cal_factor=polref * 1.0300
    elif filt == 'IR':
        cal_factor = 0.0887	
        polref = cal_factor / 0.25256
        if polar == '+60DEG':	
            cal_factor=polref
        elif polar =='0DEG':	
            cal_factor=polref * 0.9833
        elif polar == '-60DEG':	
            cal_factor=polref * 1.0288
    else:
        sys.exit('Unrecognized filter in c3_calfactor')
        
    
    if not nosum:
        if (hdr['sumcol'] > 0): cal_factor = cal_factor / hdr['sumcol']
        if (hdr['sumrow'] > 0): cal_factor = cal_factor / hdr['sumrow']
        if (hdr['lebxsum'] > 1): cal_factor = cal_factor / hdr['lebxsum']
        if (hdr['lebysum'] > 1): cal_factor = cal_factor / hdr['lebysum']
        
    cal_factor = cal_factor*1.e-10
                     
    return cal_factor
    
#|------------------------------|
#|--- Do c2 Calibration Calc ---|
#|------------------------------|
def c2_calibrate(imIn, hdr, prepDir, noCalFac=False):
    im = np.copy(imIn)
    if hdr['detector'] != 'C2':
        sys.exit('Error in c2_calibrate, passed non C2 files')
    
    # Get the exp_factor
    efacDir = prepDir+'expfac_data/'
    expfac, bias = get_exp_factor(hdr, efacDir)
    hdr['EXPTIME'] = hdr['EXPTIME'] * expfac
    hdr['offset']  = bias
    if not noCalFac:
        calfac = c2_calfactor(hdr)
    else:
        calfac = 1

    # open the vignetting file
    c2vigFile = prepDir + 'c2vig_final.fts'
    with fits.open(c2vigFile) as hdulist:
        vig  = hdulist[0].data
        hdrV = hdulist[0].header
    # Checked the file used and dont need to mask hi/lo
     
    if (hdr['r1col'] != 20) or (hdr['r1row'] != 1) or (hdr['r2col'] != 1043) or (hdr['r2row'] != 1024):
        x1 = hdr['r1col'] - 20
        x2 = hdr['r2col'] - 20
        y1 = hdr['r1row'] - 1
        y2 = hdr['r2row'] - 1
        vig =  vig[y1:y2+1,x1:x2+1]
        print ('Hitting uncheck portion of vignetting in c2_calibrate, should double check')
        
    if (hdr['sumcol'] > 1) or (hdr['sumrow'] > 1):
        # lines 170 -178 in IDL
        sys.exit('Need to rebin vignetting and not done yet, byeeee')
        
    # Ignoring some header history updating
    if hdr['polar'] in ['PB', 'TI', 'UP', 'JY', 'JZ', 'Qs', 'Us', 'Qt', 'Jr', 'Jt']:
        im = im / hdr['exptime']
        im = im * calfac
        im = im * vig
    else:
        im = (im - bias) * calfac / hdr['exptime']
        im = im * vig
    
    return im, hdr
    
#|------------------------------|
#|--- Do c2 Calibration Calc ---|
#|------------------------------|
def c3_calibrate(imIn, hdr, prepDir, noCalFac=False, noMask=False):
    im = np.copy(imIn)
    if hdr['detector'] != 'C3':
        sys.exit('Error in c3_calibrate, passed non C3 files')
        
    # Get the exp_factor
    efacDir = prepDir+'expfac_data/'
    expfac, bias = get_exp_factor(hdr, efacDir)
    hdr['EXPTIME'] = hdr['EXPTIME'] * expfac
    hdr['offset']  = bias
    
    if not noCalFac:
        calfac = c3_calfactor(hdr)
    else:
        calfac = 1
        
    # Vignetting    
    mjd = hdr['mid_date']
    if mjd < 51000:
        vigFile = prepDir + 'c3vig_preint_final.fts' 
    else:
        vigFile = prepDir + 'c3vig_postint_final.fts'
    with fits.open(vigFile) as hdulist:
        vig  = hdulist[0].data
        hdrV = hdulist[0].header
    
    # Mask file    
    c3maskFile = prepDir + 'c3_cl_mask_lvl1.fts'
    with fits.open(c3maskFile) as hdulist:
        mask  = hdulist[0].data
        hdrM = hdulist[0].header
    
    # Ramp file    
    c3rampFile = prepDir + 'C3ramp.fts'
    with fits.open(c3rampFile) as hdulist:
        ramp = hdulist[0].data
        hdrR = hdulist[0].header
     
    # Bkg file for fuzziness?    
    c3bkgFile = prepDir + '3m_clcl_all.fts'
    with fits.open(c3bkgFile) as hdulist:
        bkg = hdulist[0].data
        hdrb = hdulist[0].header
    
    bkg = 0.8 * bkg / hdrb['exptime']

    if (hdr['r1col'] != 20) or (hdr['r1row'] != 1) or (hdr['r2col'] != 1043) or (hdr['r2row'] != 1024):
        x1 = hdr['r1col'] - 20
        x2 = hdr['r2col'] - 20
        y1 = hdr['r1row'] - 1
        y2 = hdr['r2row'] - 1
        vig =  vig[y1:y2+1,x1:x2+1]
        mask =  mask[y1:y2+1,x1:x2+1]
        ramp =  ramp[y1:y2+1,x1:x2+1]
        bkg =  bkg[y1:y2+1,x1:x2+1]
        
        print ('Hitting uncheck portion of vignetting in c2_calibrate, should double check')
    
    if (hdr['sumcol'] > 1) or (hdr['sumrow'] > 1):
        # lines 265 -289 in IDL
        sys.exit('Need to rebin vignetting and not done yet, byeeee')
    
    # check if monthly image    
    if hdr['fileorig'] == 0:
        print ('Untested monthly image portion of c3_calibrate, should doublecheck')
        im = im / hdr['exptime']
        im = im * calfac * vig - ramp
        if not noMask:
            im = im * mask
        return im, hdr
    
    # Rm the ramp for the colored filters. You know
    if hdr['FILTER'].upper() != 'CLEAR':
        ramp = 0
    
    # check if a polarization brightness image
    if hdr['polar'] in ['PB', 'TI', 'UP', 'JY', 'JZ', 'Qs', 'Us', 'Qt', 'Jr', 'Jt']:
        im = im / hdr['exptime']
        im = im * calfac * vig
        if not noMask:
            im = im * mask
        return im, hdr
    else:
        # Ignoring fuzzy image things
        im = (im - bias) / hdr['exptime']
        im = im * vig * calfac - ramp
        if not noMask:
            im = im * mask
        return im, hdr
                    
#|----------------------------|
#|--- c2 Main Prep Wrapper ---|
#|----------------------------|
def c2_prep(filesIn, prepDir):
    ims, hdrs = [], []
    for aFile in filesIn:
        with fits.open(aFile) as hdulist:
            im  = hdulist[0].data
            hdr = hdulist[0].header
        im, hdr = c2_calibrate(im, hdr, prepDir)
        rad, dist, lon = get_solar_ephem(hdr['date-obs'], isSOHO=True)
        # Add in the solar rad in arcsec
        hdr['rsun'] = rad * 3600
        ims.append(im)
        hdrs.append(hdr)
    
    return np.array(ims), hdrs
    
#|----------------------------|
#|--- c3 Main Prep Wrapper ---|
#|----------------------------|
def c3_prep(filesIn, prepDir):
     ims, hdrs = [], []
     for aFile in filesIn:
         with fits.open(aFile) as hdulist:
             im  = hdulist[0].data
             hdr = hdulist[0].header
         im, hdr = c3_calibrate(im, hdr, prepDir)
         rad, dist, lon = get_solar_ephem(hdr['date-obs'], isSOHO=True)
         # Add in the solar rad in arcsec
         hdr['rsun'] = rad * 3600
         ims.append(im)
         hdrs.append(hdr)
    
     return np.array(ims), hdrs
    


