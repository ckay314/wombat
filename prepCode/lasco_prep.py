"""
Module for functions related to LASCO C2/C3 processing.
Largely a port of the corresponding IDL routines and we 
have kept names matching and indicated what portions have
been left out to facilitate comparison to the other version. 

External Calls:
    scc_funs, cor_prep

"""
import numpy as np
import sunpy.map
import sys, os
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.time import parse_time
from scc_funs import rebinIDL
from cor_prep import warp_tri


#|---------------------------|
#|--- Get solar ephemeris ---|
#|---------------------------|
def get_solar_ephem(yymmdd, isSOHO=False):
    """
    Function to get properties of the sun for a given date. Port of
    solar_ephem.pro from IDL.

    Input:
        yymmdd: the date of interest in the form YYMMDD
               
    Optional Input:   
        isSOHO: a flag to calculated values for SOHO (defaults to false)
                sets distance to 0.99 au instead of 1 au
    
    Output:
        radius: the radius of the sun [in deg]
        
        dist: distance from sun to earth [in AU]
    
        lon: mean longitude of sun, correcteed for aberration [in deg]
    

    """
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
    
#|---------------------------|
#|--- Get exposure factor ---|
#|---------------------------|
def get_exp_factor(hdr, efacDir):
    """
    Function to get the exposure factor for LASCO observationvs

    Input:
        hdr: the header for the observation of interest
    
        efacDir: the directory where the supporting efac files are located
           
    Output:
        efac: the exposure factor
        
        bias: the bias value
    

    """
    #|-------------------------------|
    #|--- Build support file name ---|
    #|-------------------------------|
    tel = hdr['detector'].lower()
    mjd = hdr['mid_date']
    jd = mjd + 2400000.5
    t = Time(jd, format='jd')
    dt = t.to_datetime()
    yymm = dt.strftime('%y%m%d') # idl has days so we do too despite the name    
    fn = tel + '_expfactor_'+yymm+'.dat'
    
    #|-----------------------------|
    #|--- Pull values from file ---|
    #|-----------------------------|
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
    """
    Function to get the calibration factor for a LASCO C2 image

    Input:
        hdr: the header of the image we want to calbrate
           
    Optional Input:   
        noSum: flag not to account for summing in the calfactor (defaults to false)
            
    Output:
        cal_factor: the calibration factor

    """
    #|------------------------------|
    #|--- Pull header properties ---|
    #|------------------------------|
    filt = hdr['filter'].upper()
    polar = hdr['POLAR'].upper()
    mjd = hdr['mid_date']
    
    #|---------------------------------|
    #|--- Get factor based on props ---|
    #|---------------------------------|
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
        
    #|---------------------------|
    #|--- Summing adjustments ---|
    #|---------------------------|
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
    """
    Function to get the calibration factor for a LASCO C3 image

    Input:
        hdr: the header of the image we want to calbrate
           
    Optional Input:   
        noSum: flag not to account for summing in the calfactor (defaults to false)
            
    Output:
        cal_factor: the calibration factor

    """
    #|------------------------------|
    #|--- Pull header properties ---|
    #|------------------------------|
    filt = hdr['filter'].upper()
    polar = hdr['POLAR'].upper()
    mjd = hdr['mid_date']
    
    #|---------------------------------|
    #|--- Get factor based on props ---|
    #|---------------------------------|
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
        
    
    #|---------------------------|
    #|--- Summing adjustments ---|
    #|---------------------------|
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
    """
    Main processing wrapper for C2 calibration

    Input:
        imIn: the image we want to calbrate
        
        hdr: the header of the image we want to calbrate
    
        prepDir: the path where the extra files needed for prep are stored
       
    Optional Input:   
        noCalFac: flag to turn off calibration factor correction (defaults to False)
            
    Output:
        img: the calibrated image 
        
        hdr: the updated header for the calbrated image 
    

    """
    #|-------------------|
    #|--- Check is C2 ---|
    #|-------------------|
    im = np.copy(imIn)
    if hdr['detector'] != 'C2':
        sys.exit('Error in c2_calibrate, passed non C2 files')
    
    #|-----------------------------|
    #|--- Pull exp_factor value ---|
    #|-----------------------------|
    efacDir = prepDir+'expfac_data/'
    expfac, bias = get_exp_factor(hdr, efacDir)
    hdr['EXPTIME'] = hdr['EXPTIME'] * expfac
    hdr['offset']  = bias
    if not noCalFac:
        calfac = c2_calfactor(hdr)
    else:
        calfac = 1

    #|---------------------------|
    #|--- Get vignetting file ---|
    #|---------------------------|
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
        
    #|-----------------------------|
    #|--- Apply the corrections ---|
    #|-----------------------------|
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
    """
    Main processing wrapper for C3 calibration

    Input:
        imIn: the image we want to calbrate
        
        hdr: the header of the image we want to calbrate
    
        prepDir: the path where the extra files needed for prep are stored
       
    Optional Input:   
        noCalFac: flag to turn off calibration factor correction (defaults to False)
    
        noMask: flag to turn off masking (defaults to False)
            
    Output:
        img: the calibrated image 
        
        hdr: the updated header for the calbrated image 
    

    """
    
    #|-------------------|
    #|--- Check is C3 ---|
    #|-------------------|
    im = np.copy(imIn)
    if hdr['detector'] != 'C3':
        sys.exit('Error in c3_calibrate, passed non C3 files')
        
    #|-----------------------------|
    #|--- Pull exp_factor value ---|
    #|-----------------------------|
    efacDir = prepDir+'expfac_data/'
    expfac, bias = get_exp_factor(hdr, efacDir)
    hdr['EXPTIME'] = hdr['EXPTIME'] * expfac
    hdr['offset']  = bias
    
    if not noCalFac:
        calfac = c3_calfactor(hdr)
    else:
        calfac = 1
        
    #|---------------------------|
    #|--- Get vignetting file ---|
    #|---------------------------|
    mjd = hdr['mid_date']
    if mjd < 51000:
        vigFile = prepDir + 'c3vig_preint_final.fts' 
    else:
        vigFile = prepDir + 'c3vig_postint_final.fts'
    with fits.open(vigFile) as hdulist:
        vig  = hdulist[0].data
        hdrV = hdulist[0].header
    
    #|---------------------|
    #|--- Get mask file ---|
    #|---------------------|
    c3maskFile = prepDir + 'c3_cl_mask_lvl1.fts'
    with fits.open(c3maskFile) as hdulist:
        mask  = hdulist[0].data
        hdrM = hdulist[0].header
    
    #|---------------------|
    #|--- Get ramp file ---|
    #|---------------------|
    c3rampFile = prepDir + 'C3ramp.fts'
    with fits.open(c3rampFile) as hdulist:
        ramp = hdulist[0].data
        hdrR = hdulist[0].header
     
    #|---------------------------|
    #|--- Get background file ---|
    #|---------------------------|
    # Bkg file for fuzziness?    
    c3bkgFile = prepDir + '3m_clcl_all.fts'
    with fits.open(c3bkgFile) as hdulist:
        bkg = hdulist[0].data
        hdrb = hdulist[0].header
    
    bkg = 0.8 * bkg / hdrb['exptime']

    #|----------------------------|
    #|--- Subsections of files ---|
    #|----------------------------|
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
    
    #|-----------------------------|
    #|--- Apply the corrections ---|
    #|-----------------------------|
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
    """
    Main processing wrapper for LASCO C2 images

    Input:
        filesIn: a list of files we want to calbrate
            
        prepDir: the path where the extra files needed for prep are stored
           
    Output:
        ims: an array with image data for each of the files
        
        hdrs: the corresponding headers
    

    """
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
     """
     Main processing wrapper for LASCO C3 images

     Input:
         filesIn: a list of files we want to calbrate
            
         prepDir: the path where the extra files needed for prep are stored
           
     Output:
         ims: an array with image data for each of the files
        
         hdrs: the corresponding headers
    
     """
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
    

#|-------------------------|
#|---  C2 Warp Function ---|
#|-------------------------|
def c2_warp(im, hdr):
    """
    Function to correct for warping in C2 images

    Input:
        img: the image we want to correct
        
        hdr: the header of the image we want to correct
        
    Output:
        img: the corrected image 
        
        hdr: the updated header for the corrected image    

    """
    #|-------------------------------------|
    #|--- Set up grid of control points ---|
    #|-------------------------------------|
    # Establish control points every 32 pixels
    w = np.arange(33*33)
    y = (w / 33).astype(int)
    x = w - y * 33
    x = x *32
    y = y*32
    
    im, hdr = reduce_std_size(im, hdr, noRebin=True, noCal=True)

    #|----------------------------------|
    #|--- Get Occulter Center pixels ---|
    #|----------------------------------|
    # Use defaults for occulter center
    tel  = hdr['DETECTOR']
    filt = hdr['FILTER']
    
    if tel == 'C1':
        if filt == 'FE X': ctr = [511.029,494.521]
        elif filt == 'FE XIV': ctr = [510.400,495.478]
        else: ctr = [511,495]
    elif tel == 'C2':
        if filt == 'ORANGE': ctr = [512.634,505.293]
        else: ctr = [512.634,505.293]
    elif tel == 'C3':
        if filt == 'ORANGE': ctr = [516.284,529.489]
        elif filt == 'CLEAR': ctr = [516.284,529.489]
        else: ctr = [516.284,529.489]
    else:
        sys.exit('EIT not ported or other unknown instrument in c2_warp')
        
    xc, yc = ctr[0], ctr[1]
    
    # Might need to dumb down hdr again.. tbd
    sumx = hdr['lebxsum'] * np.max([hdr['sumcol'], 1])
    sumy = hdr['lebysum'] * np.max([hdr['sumrow'], 1])

    #|---------------------------|
    #|--- Correct for summing ---|
    #|---------------------------|
    if sumx > 0:
        x = x / sumx
        xc = xc / sumx
    if sumy > 0:
        y = y / sumy
        yc = yc / sumy
        
    fDict = {'C1':5.8, 'C2':11.9, 'C3':56.} # pulled from subtense, called by get_sec_pixel
    scalef = fDict[tel]    
    r = np.sqrt((sumx * (x-xc))**2 + (sumy * (y-yc))**2)
    
    #|------------------------------|
    #|--- Apply warping function ---|
    #|------------------------------|
    mm = r * 0.021
    cf =[0.0051344125,-0.00012233862,1.0978595e-7] # pulled from DISTORTION_COEFFS for C2
    f1 = mm * ( cf[0] + cf[1] * mm**2 + cf[2]*mm**4)
    f1 = (r + f1/0.021)*scalef
    r0 = f1 / (sumx * scalef)
    
    theta = np.arctan2(y-yc, x-xc)
    x0 = r0 * np.cos(theta) + xc
    y0 = r0 * np.sin(theta) + yc
    
    #|-------------------------------------------|
    #|--- Convert using the cor_prep function ---|
    #|-------------------------------------------|    
    im = warp_tri(x, y, x0, y0, im)
        
    return im, hdr

#|-------------------------|
#|---  C3 Warp Function ---|
#|-------------------------|
def c3_warp(im, hdr):
    """
    Function to correct for warping in C3 images

    Input:
        img: the image we want to correct
        
        hdr: the header of the image we want to correct
        
    Output:
        img: the corrected image 
        
        hdr: the updated header for the corrected image    

    """

    #|-------------------------------------|
    #|--- Set up grid of control points ---|
    #|-------------------------------------|
    # Establish control points every 32 pixels
    w = np.arange(33*33)
    y = (w / 33).astype(int)
    x = w - y * 33
    x = x *32
    y = y*32

    im, hdr = reduce_std_size(im, hdr, noRebin=True, noCal=True)

    # Use defaults for occulter center
    tel  = hdr['DETECTOR']
    filt = hdr['FILTER']
    
    #|----------------------------------|
    #|--- Get Occulter Center pixels ---|
    #|----------------------------------|
    # Using default occulter center, same for all filters in occltr_cntr.pro
    ctr = [516.284,529.489]
        
    xc, yc = ctr[0], ctr[1]
    
    x1 = hdr['r1col'] - 20
    x2 = hdr['r2col'] - 20
    y1 = hdr['r1row'] - 1
    y2 = hdr['r2row'] - 1

    #|---------------------------|
    #|--- Correct for summing ---|
    #|---------------------------|
    sumx = hdr['lebxsum'] * np.max([hdr['sumcol'], 1])
    sumy = hdr['lebysum'] * np.max([hdr['sumrow'], 1])

    if sumx > 1:
        x  =  x / sumx
        xc = xc / sumx
        x1 = x1 / sumx
        x2 = x2 / sumx
    if sumy > 1:
        y  =  y / sumy
        yc = yc / sumy
        y1 = y1 / sumy
        y2 = y2 / sumy
        
    scalef = 56. # pulled from subtense for C3, called by get_sec_pixel
    r = np.sqrt((sumx * (x-xc))**2 + (sumy * (y-yc))**2)
    
    #|------------------------------|
    #|--- Apply warping function ---|
    #|------------------------------|
    # apply c3_distortion
    mm = r * 0.021
    cf =[-0.0151657, 0.000165455, 0.0] # pulled from DISTORTION_COEFFS for C3
    f1 = mm * ( cf[0] + cf[1] * mm**2)
    f1 = (r + f1/0.021)*scalef
    r0 = f1 / (sumx * scalef)
    
    theta = np.arctan2(y-yc, x-xc)
    x0 = r0 * np.cos(theta) + xc
    y0 = r0 * np.sin(theta) + yc
    
    #|-------------------------------------------|
    #|--- Convert using the cor_prep function ---|
    #|-------------------------------------------|    
    im = warp_tri(x, y, x0, y0, im)
        
    return im, hdr

#|---------------------------------|
#|--- Make Standard Format Img  ---|
#|---------------------------------|
def reduce_std_size(im, hdr,bias=None, noRebin=False, noCal=False, full=False, saveHDR=False):
    """
    Function to reduce to a standard, processed 512x512 C2/C3 image

    Input:
        img: the image we want to calbrate
        
        hdr: the header of the image we want to calbrate
          
    Optional Input:   
        bias: a bias value (think this was supposed to be an IDL output.., ignored here anyway)
        
        noRebin: flag to keep the image full sized (1024 not 512) (defauls to False)
        
        noCal: flag to turn off the calibration (defaults false)
        
        noRebin: flag to keep the image full sized (1024 not 512, defaults to False)
                 *IDL has both of these... seems unnecessary
    
        saveHDR: option to save the original hdr (i.e. do not put new values in it, defaults to false)
    
    Output:
        img: the standard image 
        
        hdr: the corresponding header
    

    """
    #|--------------------------|
    #|--- Pull Header Values ---|
    #|--------------------------|
    sumrow = np.max([hdr['SUMROW'], 1])
    sumcol = np.max([hdr['SUMCOL'], 1])
    lebxsum = np.max([hdr['LEBXSUM'], 1])
    lebysum = np.max([hdr['LEBYSUM'], 1])
    naxis1 = hdr['NAXIS1']
    naxis2 = hdr['NAXIS2']
    polar = hdr['polar']
    tel = hdr['TELESCOP']
    
    if (naxis1 <=0) or (naxis2 <=0):
        sys.exit('Invaid image passed to reduce_std_size')
        
    r1col = hdr['R1COL']
    r1row = hdr['R1ROW']
    if r1col < 1: r1col = hdr['P1COL']
    if r1row < 1: r1row = hdr['P1ROW']
    
    #|------------------------------------------|
    #|--- Standard Corrections based on hdr  ---|
    #|------------------------------------------|
    if (type(bias) != type(None)) & (not noCal):
        print('Hitting unchecked portion of reduce_std_size')
        if sumcol > 1:
            im = (im - bias) / (sumcol * sumrow)
        else:
            if lebxsum > 1:
                print ('Correcting for chip summing')
                im = (im - bias)/(lebxsum*lebysum)
        hdr['OFFSET'] = 0
    
    nxfac = 2**(sumcol+lebxsum-2)
    nyfac = 2**(sumrow+lebysum-2)
    
    if tel == 'SOHO':
        if ((hdr['r2col'] - r1col) == 1023) & ((hdr['r2row'] - r1row) == 1023)  & (naxis1 ==512):
            nxfac, nyfac = 2, 2
            
    nx = nxfac*naxis1
    ny = nyfac*naxis2
    
    # Some C1 images have incorrect values for r row/col values
    if (hdr['r2col'] - r1col + 1) != (naxis1*lebxsum): r1col = r1col - 32
    if (hdr['r2row'] - r1row + 1) != (naxis2*lebxsum): r1row = r1row - 32
    
    # Assuming not subframes
    if ((nx < 1024) or (ny < 1024)) & (tel == 'SOHO'):
        sys.exit('Havent ported subframe portion of reduce_std_size')
    
    scale_to = 512
    if noRebin: scale_to = naxis1
    if full: scale_to = 1024
    
    #|---------------------------------|
    #|--- Update the header values  ---|
    #|---------------------------------|
    if not saveHDR:
        if noCal:
            hdr['crpix1'] = ((hdr['crpix1']*nxfac)+r1col-20)*scale_to/1024. 
            hdr['crpix2'] = ((hdr['crpix2']*nyfac)+r1row-1)*scale_to/1024.
        else:
            sys.exit('Hitting uncoded portion of reduce_std_size (not noCal)')
        
        if tel == 'SOHO':
            hdr['R1COL'] = 20
            hdr['R1ROW'] = 1
            hdr['R2COL'] = 1043
            hdr['R2ROW'] = 1024
        
            if (type(bias) != type(None)):
                hdr['lebxsum'] = 1
                hdr['lebysum'] = 1
                hdr['offset']  = 0
                
        hdr['NAXIS1'] = scale_to
        hdr['NAXIS2'] = scale_to
        hdr['cdelt1'] = hdr['cdelt1']*(1024/scale_to)
        hdr['cdelt2'] = hdr['cdelt2']*(1024/scale_to)
 
    #|-----------------------|
    #|--- Rebin the image ---|
    #|-----------------------|
    im = rebinIDL(im, np.array([scale_to, scale_to]))
    return im, hdr
            
#|--------------------------|
#|---  Main Prep Wrapper ---|
#|--------------------------|
# Converts from 0.5 to Lev 1 data
def reduce_level_1(filesIn, no_mask=False, noScale=False, prepDir='prepFiles/soho/lasco/'):
    """
    Main processing wrapper for LASCO images

    Input:
        filesIn: a list of files to process
       
    Optional Input:   
        no_mask: flag to not include mask in images (defaults to false)
        
        noScale: flag that does nothing, for unported IDL parts (defaults to false)
        
        prepDir: directory where the processing helper files live (defaults to prepFiles/soho/lasco/)
    
    Output:
        ims: an array of prepped images corresponding to the input files 
        
        hdrs: an array of headers matching the images
    

    """
    # Might need to findthings from common blocks 162-165
    
    #|---------------------------|
    #|---  Check array/string ---|
    #|---------------------------|
    if type(filesIn) == type('ImAString'):
        filesIn = [filesIn]
    
    #|-----------------|
    #|--- Main Loop ---|
    #|-----------------|
    ims, hdrs = [], []
    # Loop through the array
    for aFile in filesIn:
        # Exit if cannot find file
        if not os.path.exists(aFile):
            sys.exit('Cannot find '+aFile)
        #|-------------------|
        #|--- Open a file ---|
        #|-------------------|
        with fits.open(aFile) as hdulist:
            im  = hdulist[0].data
            hdr = hdulist[0].header
       
        camera = hdr['detector']
        hdrIn = hdr
        # Assume we need to read in the vig/mask/ramp files
        
        xsumming = np.max([hdr['sumcol'], 1]) * np.max([hdr['lebxsum'], 1])
        ysumming = np.max([hdr['sumrow'], 1]) * np.max([hdr['lebysum'], 1])
        summing = xsumming * ysumming
        
        if summing > 1:
            # Ignoring fix wrap, just removes overflows
            dofull = False
        else:
            dofull = True
        
        if (hdr['r2col']-hdr['r1col']+hdr['r2row']-hdr['r1row']-1023-1023) != 0:
            sys.exit('Hit uncoded part in reduce_level_1, need to port reduce std size')
        
        fname = hdr['filename']
        source = fname[1]
        dot = fname.find('.')
        root = fname[:dot]
        yymmdd = hdr['date-obs'].replace('/','')[2:]
        
        if source in ['1', '3']:
            root = root[0]+'4'+root[2:]
        elif source == '2':
            root = root[0]+'5'+root[2:]
        elif source in ['m', 'd']:
            sys.exit('Hit uncoded part in reduce_level_1, need to port monthly code portion')
        
        outname = root + '.fts' # suspect we want a diff name but keep the var
        
        # Skipping printing to logfile/screen 245-261
        
        # Take caldir as optional input, points to wombat files instead of ssw
        
        # Skipping more logfile things 267 - 307        
        
        
        # |------------------------------------|
        # |---- Actual Processing by camera ---|
        # |------------------------------------|
        if camera == 'C1':
            sys.exit('C1 not available in reduce_level_1')
        elif camera == 'C2':
            #|-----------------|
            #|--- Calibrate ---|
            #|-----------------|
            im, hdr = c2_calibrate(im, hdr, prepDir)
            
            #|---------------|
            #|--- Warping ---|
            #|---------------|
            im, hdr = c2_warp(im, hdr)
                        
            # No mask - assume no missing blocks for now (true on test case)
        elif camera == 'C3':
            #|-----------------|
            #|--- Calibrate ---|
            #|-----------------|
            im, hdr = c3_calibrate(im, hdr, prepDir)
            
            #|---------------|
            #|--- Warping ---|
            #|---------------|
            im, hdr = c3_warp(im, hdr)
        
        #|-----------------------|
        #|--- Roll correction ---|
        #|-----------------------|       
        # adjust_hdr_tcr was a rabbit hole not worth the effort
        # Running with unadjusted values that are within 1 pix/deg
        roll = hdr['crota1']
        cx   = hdr['crpix1'] - 1 #think adjusted to array units not fits
        cy   =  hdr['crpix2'] - 1 
        
        if np.abs(roll) > 170:
            rectify = 180 
            cntr = 511.5
            x = cx-cntr
            y = cy-cntr
            cx = cntr + x * np.cos(rectify * np.pi/180.) - y * np.sin(rectify * np.pi/180.)
            cy = cntr + x * np.sin(rectify * np.pi/180.) + y * np.cos(rectify * np.pi/180.)
            
            im = np.rot90(im, k=2)
            roll = roll - 180
        else:
            rectify = 0
        
        xc = (cx - hdr['r1col'] + 20) / xsumming
        yc = (cy - hdr['r1row'] + 1) / ysumming
        
        if roll < -180: roll += 360
        
        crpix_x = xc+1
        crpix_y = yc+1
        
        #|---------------------|
        #|--- Adjust header ---|
        #|---------------------|       
        hdr['CRPIX1'] = crpix_x
        hdr['CRPIX2'] = crpix_y
        hdr['CROTA']  = roll
        hdr['CROTA1']  = roll
        hdr['CROTA2']  = roll
        hdr['CRVAL1']  = 0
        hdr['CRVAL2']  = 0
        hdr['CTYPE1']  = 'HPLN-TAN'
        hdr['CTYPE2']  = 'HPLT-TAN'
        hdr['CUNIT1']  = 'arcsec'
        hdr['CUNIT2']  = 'arcsec'
        fDict = {'C1':5.8, 'C2':11.9, 'C3':56.} # pulled from subtense, called by get_sec_pixel
        platescl = fDict[camera]
        hdr['CDELT1']  = platescl
        hdr['CDELT2']  = platescl
        hdr['XCEN'] = 0 + platescl*((hdr['naxis1']+1)/2. - crpix_x)
        hdr['YCEN'] = 0 + platescl*((hdr['naxis2']+1)/2. - crpix_y)
        
        
        # Not doing scaling
        ims.append(im)
        hdrs.append(hdr)
 
    return np.array(ims), hdrs     
            
if __name__ == '__main__':
    prepDir = 'wbFits/SOHO/LASCO/C2/'
    filesIn = 'pullFolder/SOHO/LASCO/C3/20230924T2142_C3_32754722.fts'
    reduce_level_1(filesIn, prepDir)