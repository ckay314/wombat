"""
Module for preparing PSP WISPR data. Largely a port
of the IDL functions of the same names.

"""
import numpy as np
#import sunpy.map
import sys
from scc_funs import scc_make_array, scc_zelensky_array, rebinIDL
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.time import parse_time
from sunpy.coordinates import get_horizons_coord
from sunspyce import get_sunspyce_hpc_point, get_sunspyce_roll, get_sunspyce_coord, get_sunspyce_lonlat, get_sunspyce_p0_angle, get_sunspyce_carr_rot
from sunpy.coordinates import spice
import scipy.io
from wcs_funs import fitshead2wcs, wcs_get_coord, idlsav2wcs



#|----------------------------|
#|--- Supress Info Logging ---|
#|----------------------------|
# Make sunpy/astropy shut up about info/warning for missing metadata
import logging
logging.basicConfig(level='INFO')
slogger = logging.getLogger('sunpy')
slogger.setLevel(logging.ERROR)
alogger = logging.getLogger('astropy')
alogger.setLevel(logging.ERROR)
np.seterr(divide='ignore')

#|--------------------------|
#|--- Launch SPICE thing ---|
#|--------------------------|
spice.install_frame('IAU_SUN')

#|---------------------|
#|--- Useful Global ---|
#|---------------------|
global dtor 
dtor = np.pi / 180.


#|-------------------------|
#|--- Read in fits file ---|
#|-------------------------|
def wispr_readfits(fileIn, LASCO=False, isCal=False):
    """
    Function to read in a wispr fits file
    
    Input:
        fileIn: the path to the wispr fits file 
    
    Optional Input:
        LASCO: flag to indicate it is a LASCO file (defaults to False)
               *** key ported from IDL but not really implemented
    
        isCal: flag to indicate that it is a calibration file being opened
               (defaults to False)
    
    Output:
        im: the image data from the fits file
    
        hdr: the corresponding header     

    """
    # Assume we are passed a good file, open it up and skip
    # down to clean up stuff at 128
    with fits.open(fileIn) as hdulist:
        im  = hdulist[0].data
        hdr = hdulist[0].header
    
    
    # Fix type in early def_wispr_hdr.pro
    hdr['CTYPE1A'] = 'RA---ZPN'
    hdr['CTYPE2A'] = 'DEC--ZPN'
    
    # Fix DATE-* keywords from Bug 460 WISPR FITS DATE-OBS incorrect
    if not isCal: # Our cal files are missing some keywords but don't do this anyway
        if hdr['VERS_CAL'][:2] == '0x':
            sys.exit('Hitting unported code for bug 460. TBD')
        
        # Fill in DATE_OBS in headers
        if not LASCO:
            isOld = False
            if hdr['DATE-OBS'] < '2006-02-14T00:00:00.000':
                isOld = True
        
            if hdr['INSTRUME'] != 'WISPR':
                sys.exit('Hitting unported code for not a WISPR header')
            
    return im, hdr
        
#|----------------------|
#|--- Get Bias Value ---|
#|----------------------|
def wispr_bias_offset(im, hdr):
    """
    Function to get the bias offset for wispr data
    
    Input:
        im: a wispr image
    
        hdr: the corresponding header
    
    Output:
        offset: the median bias value  

    """
    if hdr['rectify'] == True:
        rectrota = hdr['rectrota']
        # not sure the extent of possible options for this
        # and IDL rotates by (4-rectrota) if it is 1 or 3 
        # test case has 6
        # leave other options uncoded for now
        if rectrota == 1: # -> IDL rotate (im,4-1)
            tempIm = np.rot90(im,k=1)
        elif rectrota == 3: # -> IDL rotate (im,4-3)
            tempIm = np.rot90(im,k=-1)
        elif rectrota == 6:
            tempIm = np.rot90(im[:,::-1], k=-1)
        
        rows = hdr['naxis2']
    else:
        imrot = im
        
    mask = np.array([4,8]) / hdr['nbin1'] - 1
    mask = mask.astype(int)
    subIm = tempIm[mask[0]:mask[1]+1,:]                    
    offset = np.median(subIm[np.isfinite(subIm)])
    
    return offset
    
#|------------------------------|
#|--- Get calibration factor ---|
#|------------------------------|
def wispr_get_calfac(hdr):
    """
    Function to get the calibration factor for wispr data
    based on the detector, gainmode, and gaincmd values 
    within a header
    
    Input:
        hdr: a wispr header
    
    Output:
        calfac: the calibration factor  

    """
    if (hdr['detector'] in [1, 2]) and  (hdr['gainmode'] in ['HIGH', 'LOW']):
        if hdr['detector'] == 1:
            if hdr['gainmode'] == 'LOW': calfac = 2.49e-13
            else:  calfac = 4.09e-14
        else:
            if hdr['gainmode'] == 'LOW': calfac = 3.43e-13
            else:  calfac = 7.28e-14
        if hdr['gaincmd'] == 12: calfac = calfac * 1.27
        hdr['history'] = 'Calibration factor ' + str(calfac) +' applied'                    
    else:
        sys.exit('Invalid WISPR header, issue in calfac')
    return calfac
    
#|-----------------------------|
#|--- Get calibration image ---|
#|-----------------------------|
def wispr_get_calimg(hdr, wcalpath):
    """
    Function to open and process a wispr calibration image
    
    Input:
        hdr: the header for the image for which we want the calibration img
    
        wcalpath: path to where the wispr calibration images live
    
    Output:
        calimg: the image data for the calibration image
    
        calhdr: the corresponding calibration image header

    """
    if hdr['detector'] not in [1, 2]:
        sys.exit('Invalid WISPR header, issue in calimg')
    if hdr['detector'] == 1:
        calFile = 'WISPR_FredVignettingFitted_inner_20180809_01.fits'
    else:
        calFile = 'WISPR_FredVignettingFitted_outer_20190813_01.fits'    
    hdr['history'] = 'Using '+ calFile
        
    calimg, calhdr = wispr_readfits(wcalpath+calFile, isCal=True)
    
    # Check for sub field
    if (hdr['pxend1'] - hdr['pxbeg1'] != 2047) or (hdr['pxend2'] - hdr['pxbeg2'] != 1919):
        print ('Hitting untested portion, should double check cal img in wispr_get_calimg')
        calimg =  calimg[hdr['pxbeg2']-1:hdr['pxend2'], hdr['pxbeg1']-1:hdr['pxend1']]
    
    # chec if input image has been rectified and roate vignetting to mach
    if hdr['rectify']:
        calimg = np.transpose(calimg)
    
    # If input is binned, bin cali image
    if hdr['nbin'] > 1:
        calimg = rebinIDL(calimg, np.array([hdr['naxis2'], hdr['naxis1']]))

    calimg = calimg ** 2
    w = np.where(calimg == 0)
    if len(w[0]) > 0:
        calimg[w] = np.nan
    return calimg, calhdr

    
#|-------------------------------|
#|--- Main Correction Wrapper ---|
#|-------------------------------|
def wispr_correction(im, hdr, wcalpath, calfacOff=False, calimgOff=False, exptimeOff=False, truncOff=False):
    """
    The main wrapper for wispr image correction.
    
    Input:
        im: a wispr image
    
        hdr: the corresponding header
    
    Optional Input:
        calfacOff: flag to turn off calibration factor correction (defaults to False)
    
        calimgOff: flag to turn off calibration image correction (defaults to False)
    
        exptimeOff: flag to turn off exposure time correction (defaults to False)
    
        truncOff: flag to turn off IP truncation correction (defaults to False)
    
    Output:
        im: the corrected wispr image
    
        hdr: the updated header

    """
    hdr['history'] = 'Applied wispr_correction (Python port)'
    
    calcnt = 0 
    
   #|-----------------------------|
   #|--- Truncation Correction ---|
   #|-----------------------------|
    if not truncOff and (hdr['IPTRUNC'] != 0):
        im = im * 2**hdr['IPTRUNC']
        hdr['history'] = 'Image multiplied by ' + str(2**hdr['IPTRUNC']) + ' due to truncation'
        
    #|--------------------------|
    #|--- Binning Correction ---|
    #|--------------------------|
    # Correct for DN/s for binned images
    divfactor = hdr['nbin']
    if divfactor > 1:
        im = im / divfactor
        hdr['history'] = 'Image divided by ' + str(divfactor) + ' for binning'
        hdr['dsatval'] = hdr['dsatval'] / divfactor
            
    #|--------------------------------|
    #|--- Exposure Time Correction ---|
    #|--------------------------------|
    # Normalize for exposure time
    if not exptimeOff:
        im = im / hdr['xposure']
        hdr['dsatval'] = hdr['dsatval'] / hdr['xposure']
        hdr['bunit'] = 'DN/s'
        calcnt = calcnt + 1
        hdr['history'] = 'Exposure normalized to 1 second'
    
    #|-------------------------------------|
    #|--- Calibration Factor Correction ---|
    #|-------------------------------------|
    calfac = 1.
    if not calfacOff:
        hdr['bunit'] = 'MSB'
        calfac = wispr_get_calfac(hdr)
        hdr['dsatval'] = hdr['dsatval'] * calfac
        calcnt = calcnt + 1
   
    #|------------------------------------|
    #|--- Calibration Image Correction ---|
    #|------------------------------------|
    # cal img
    calimg = 1.
    if not calimgOff:
       calimg, calhdr = wispr_get_calimg(hdr, wcalpath) 
       calcnt = calcnt+1
       
    im = im / calimg * calfac
    
    #|-----------------------|
    #|--- Set Level Label ---|
    #|-----------------------|
    if calcnt == 3:
        if hdr['level'] != 'L2':
            hdr['level'] = 'L2'
   
    return im, hdr
    
#|-------------------------|
#|--- Get Pointing Info ---|
#|-------------------------|
def get_wispr_pointing(shdr, wcalpath, doSpice=True, doCoords=False):
    """
    Function to get the pointing of wispr
    
    Input:
        shdr: the header of a wispr image
    
        wcalpath: the path to where the wispr calibration images live
    
    Optional Input:
        doSpice: use spice functions to get the pointing (defaults to True)
    
        doCoords: flag to update the header with additional coordinate info
    
    Output:
        shdr: an updated header

    """
    shdr['VERS_CAL'] = '2020915'
    detect = shdr['DETECTOR']
    instr = ['SPP_WISPR_INNER', 'SPP_WISPR_OUTER']
    
    # From WISPR_FITSHeaderDefinition_20170811_01.pdf
    #8 WCS Compliant FITS Keyword Definition
    #detector depndent information
    
    #|-------------------------|
    #|--- Detector 1 values ---|
    #|-------------------------|
    if detect == 1:
        shdr['CRPIX1'] =  (991.547 - (1920 - shdr['PXEND2'])) / np.sqrt(shdr['nbin'])
        shdr['CRPIX2'] =  (1015.11 - (2048 - shdr['PXEND1'])) / np.sqrt(shdr['nbin'])
        
        if ((shdr['pxend2'] - shdr['pxbeg2'] + 1) / shdr['nbin1'] - shdr['naxis1']) == 32:
            shdr['CRPIX1'] = shdr['CRPIX1'] - 32
        
        shdr['CDELT1']  =	 .0211525 * shdr['nbin1'] # deg
        shdr['CDELT2']  =	 .0211525 * shdr['nbin2'] # deg
        shdr['CDELT1A'] =	-.0211525 * shdr['nbin1'] # deg
        shdr['CDELT2A'] =	 .0211525 * shdr['nbin2'] # deg
        
        shdr['PV2_0'] = -1.9879e-8
        shdr['PV2_1'] = 1.00501
        shdr['PV2_2'] = .0729583
        shdr['PV2_3'] = .275292
        shdr['PV2_4'] = -.701881
        shdr['PV2_5'] = 1.97518

        instrument_roll = 0.417683

        skew = np.array([[.99606, .999191], [.998218,.995773]])

        xcor = 0.54789
        ycor = -.3501
    
    #|-------------------------|
    #|--- Detector 2 values ---|
    #|-------------------------|
    elif detect == 2:
        shdr['CRPIX1'] = (984.733 - (1920 - shdr['PXEND2'])) / shdr['nbin1']
        shdr['CRPIX2'] = (1026.37 - (2048 - shdr['PXEND1'])) / shdr['nbin2']
        
        if (shdr['pxend2'] - shdr['pxbeg2'] + 1) / shdr['nbin1'] - shdr['naxis1'] == 32:
             shdr['CRPIX1'] = shdr['CRPIX1'] - 32
             
        shdr['CDELT1']  =	0.0282376 * shdr['nbin1']  # deg
        shdr['CDELT2']  =	0.0282376 * shdr['nbin2']  # deg
        shdr['CDELT1A'] =  -0.0282376 * shdr['nbin1']  # deg
        shdr['CDELT2A'] =	0.0282376 * shdr['nbin2']  # deg
        
        shdr['PV2_0'] = .000168385
        shdr['PV2_1'] = .983801
        shdr['PV2_2'] = .0737626
        shdr['PV2_3'] = -.374471
        shdr['PV2_4'] = 0.585763
        shdr['PV2_5'] = -.410706

        instrument_roll = 0.40121182
        skew = [[1,1],[1,1]]

        xcor = 0
        ycor = 0
        
    #|------------------------|
    #|--- Unknown Detector ---|
    #|------------------------|
    else:
        sys.exit('Invalid detector given to get_wispr_pointing')
        
    shdr['CUNIT1']  =	'deg'
    shdr['CUNIT2']  =	'deg'
    shdr['CUNIT1A'] =	'deg'
    shdr['CUNIT2A'] =	'deg'
    shdr['PV2_0A'] = shdr['PV2_0']
    shdr['PV2_1A'] = shdr['PV2_1']
    shdr['PV2_2A'] = shdr['PV2_2']
    shdr['PV2_3A'] = shdr['PV2_3']
    shdr['PV2_4A'] = shdr['PV2_4']
    shdr['PV2_5A'] = shdr['PV2_5']
    
    #|--------------------|
    #|--- Get pointing ---|
    #|--------------------|
    point = np.zeros(3)   
    if doSpice:
        point = get_sunspyce_hpc_point(shdr['DATE-AVG'], 'psp', instrument=instr[detect-1], doDeg=True) 
    point[2] = point[2] + instrument_roll
    roll1 = point[2]
    shdr['CRVAL1'] = point[0] + xcor
    shdr['CRVAL2'] = point[1] + ycor
        
    pc = np.array([[np.cos(point[2]*dtor),  -np.sin(point[2]*dtor)], [np.sin(point[2]*dtor), np.cos(point[2]*dtor)]]) * skew
    shdr['PC1_1'] = pc[0,0]
    shdr['PC1_2'] = pc[0,1] # swapped index to match IDL 
    shdr['PC2_1'] = pc[1,0] # swapped index to match IDL 
    shdr['PC2_2'] = pc[1,1] 

    shdr['nbin1'] = np.sqrt(shdr['nbin'])
    shdr['nbin2'] = np.sqrt(shdr['nbin'])
        
    # Skipping get att file for now, IDL doesn't seem to pull anything
    #if ('ATT_FILE' in shdr) & doSpice:
        
    shdr['PV1_1']    = 0.0 	    # deg
    shdr['PV1_2']    = 90.0	    # deg
    shdr['PV1_3']    = 180.0	# deg
    shdr['PV1_1A']   = 0.0 	    # deg
    shdr['PV1_2A']   = 90.0	    # deg
    shdr['PV1_3A']   = 180.0	# deg
    shdr['LATPOLE']  = 0.0
    shdr['LONPOLE']  = 180.0
    shdr['LATPOLEA'] = 0.0
    shdr['LONPOLEA'] = 180.0
    roll = 0.
    ra = 0.
    dec = 90.
        
    #|-------------------------|
    #|---Get satellite roll ---|
    #|-------------------------|
    if doSpice:
        roll, dec, ra = get_sunspyce_roll(shdr['DATE-AVG'], 'psp', instrument=instr[detect-1], system='GEI')   
    roll = roll + instrument_roll
    if ra < 0: ra = ra +360   
    shdr['CRVAL1A'] = ra+xcor
    shdr['CRVAL2A'] = dec+ycor
    shdr['CRPIX1A'] = shdr['CRPIX1']
    shdr['CRPIX2A'] = shdr['CRPIX2']
    pc= np.array([[np.cos(roll*dtor), np.sin(roll*dtor)], [-np.sin(roll*dtor), np.cos(roll*dtor)]]) * skew
    shdr['PC1_1A'] = pc[0,0]
    shdr['PC1_2A'] = pc[1,0] # swapped index to match IDL 
    shdr['PC2_1A'] = pc[0,1] # swapped index to match IDL     
    shdr['PC2_2A'] = pc[1,1]

    #|----------------------------|
    #|--- Pull detector values ---|
    #|----------------------------|
    if shdr['detector'] == 1:
        # Load up save files from IDL, may want to replace eventually
        if np.abs(roll) <= 150.:
            idl_save = wcalpath + 'rollcomp.sav'
        elif roll > 150:
            idl_save = wcalpath + 'rollcompp180.sav'
        elif roll < -150:
            idl_save = wcalpath + 'rollcompm180.sav'
        idlRoll = scipy.io.readsav(idl_save, python_dict=True)
        
        
        
        wcso = idlsav2wcs(wcalpath+'wcso.sav')
        wcso['NAXIS'] = wcso['NAXIS'] / np.sqrt(shdr['nbin'])
        wcso['CRPIX'] = wcso['CRPIX'] / np.sqrt(shdr['nbin'])
        wcso['CDELT'] = wcso['CDELT'] * np.sqrt(shdr['nbin'])
        # Things saved as arrays are [array, dtype]
        wcso['PC'] = np.transpose([[np.cos(roll1 * dtor), np.sin(roll1 * dtor)], [-np.sin(roll1 * dtor), np.cos(roll1 * dtor)]])
        wcso['CRVAL'] = [shdr['crval1'] - xcor, shdr['crval2'] - ycor]
        
        # poly(val, coeffs) IDL -> polyval(coeffs[::-1], val)
        shdr['CRVAL1'] = shdr['CRVAL1'] + np.polyval(idlRoll['p1'][::-1], roll)[0]
        shdr['CRVAL2'] = shdr['CRVAL2'] + np.polyval(idlRoll['p2'][::-1], roll)[0]
        roll1 = roll1 + np.polyval(idlRoll['p3'][::-1], roll)[0]
        pc = np.array([[np.cos(roll1*dtor), -np.sin(roll1*dtor)], [np.sin(roll1*dtor), np.cos(roll1*dtor)]]) * skew
        shdr['PC1_1']=pc[0,0]
        shdr['PC1_2']=pc[0,1] # swapped index to match IDL 
        shdr['PC2_1']=pc[1,0]# swapped index to match IDL 
        shdr['PC2_2']=pc[1,1]
        
        wcs = fitshead2wcs(shdr)
        
        # duplicate all the keys in the sav file wcs to lower case
        # seems a little risky if things are changed but not easy to sort 
        # out a consisent set without dups
        copyKeys = []
        for key in wcso:
            copyKeys.append(key)
        for key in copyKeys:
            wcso[key.lower()] = wcso[key]
            
            
        pt1 = wcs_get_coord(wcs, np.array([wcs['naxis'][0], wcs['naxis'][1]]).astype(int)/2)
        pt2 = wcs_get_coord(wcso, np.array([wcs['naxis'][0], wcso['NAXIS'][1]]).astype(int)/2)
        diff = (pt1 - pt2).reshape([-1])
        
        shdr['CRVAL1'] = shdr['CRVAL1'] - diff[0]
        shdr['CRVAL2'] = shdr['CRVAL2'] - diff[1]

        shdr['CRVAL1A']=shdr['CRVAL1A']+ np.polyval(idlRoll['p1'][::-1], roll)[0]
        shdr['CRVAL2A']=shdr['CRVAL2A']+ np.polyval(idlRoll['p2'][::-1], roll)[0]
        roll=roll+np.polyval(idlRoll['p3'][::-1],roll)[0]
        pc = [[np.cos(roll*dtor), -np.sin(roll*dtor)], [np.sin(roll*dtor), np.cos(roll*dtor)]] * skew
        shdr['PC1_1A']=pc[0,0]
        shdr['PC1_2A']=pc[0,1] # swapped index to match IDL
        shdr['PC2_1A']=pc[1,0] # swapped index to match IDL
        shdr['PC2_2A']=pc[1,1]
        
    if not shdr['rectify']:
        print('Hitting ported but untested part in get_wispr_pointing')
        shdr0 = shdr
        shdr['crpix1']= shdr0['naxis1'] - shdr0['crpix2'] + 1
        shdr['crpix2']= shdr0['naxis2'] - shdr0['crpix1'] + 1
        shdr['crpix1A']= shdr['crpix1']
        shdr['crpix2A']= shdr['crpix2']
    
        shdr['cdelt1'] = shdr0['cdelt2']
        shdr['cdelt2'] = -shdr0['cdelt1']
        shdr['cdelt1A'] = shdr0['cdelt2A']
        shdr['cdelt2A'] = -shdr0['cdelt1A']

        shdr['pc1_2'] = shdr0['pc2_1']
        shdr['pc2_1'] = shdr0['pc1_2']
        shdr['pc1_2a']= shdr0['pc2_1a']
        shdr['pc2_1a']= shdr0['pc1_2a']

        shdr['LONPOLE'] = -90.0
        shdr['LONPOLEa'] = -90.0

        shdr['PV1_3'] = -90.0
        shdr['PV1_3a'] = -90.0
    
    #|------------------------|
    #|--- Add coord values ---|
    #|------------------------|
    if doCoords:
        point = get_sunspyce_hpc_point(shdr['DATE-AVG'],'psp', doDeg=True)
        shdr['SC_PITCH'] = point[1]
        shdr['SC_ROLL']  = point[2]
        shdr['SC_YAW']   = point[0]
        
        pointb =  get_sunspyce_hpc_point(shdr['DATE-BEG'],'psp', doDeg=True)
        pointe =  get_sunspyce_hpc_point(shdr['DATE-END'],'psp', doDeg=True)
        if np.max(np.abs(pointb-point, pointb-pointe)) > 0.1:
            shdr['OBS_MODE'] = 'MOVING'
            print('Attitude change during image detected!!!')
        if np.abs(shdr['SC_ROLL']) > 8:
            shdr['OBS_MODE'] = 'ROLLED'
        elif np.max(np.abs(point[:2])) > 1:
            shdr['OBS_MODE'] = 'OFFPOINT'
        
        
        aCo = get_sunspyce_coord(shdr['DATE-AVG'],'psp', system='HAE', doVelocity=False, doMeters=True)    
        shdr['HAEX_OBS'] = aCo[0]
        shdr['HAEY_OBS'] = aCo[1]
        shdr['HAEZ_OBS'] = aCo[2] 
        
        aCo = get_sunspyce_coord(shdr['DATE-AVG'],'psp', system='HCI', doMeters=True)
        shdr['HCIX_OBS'] = aCo[0]
        shdr['HCIY_OBS'] = aCo[1]
        shdr['HCIZ_OBS'] = aCo[2]
        shdr['HCIX_VOB'] = aCo[3]
        shdr['HCIY_VOB'] = aCo[4]
        shdr['HCIZ_VOB'] = aCo[5]    
        
        aCo = get_sunspyce_coord(shdr['DATE-AVG'],'psp', system='HEE', doVelocity=False, doMeters=True)
        shdr['HEEX_OBS'] = aCo[0]
        shdr['HEEY_OBS'] = aCo[1]
        shdr['HEEZ_OBS'] = aCo[2]
        
        aCo = get_sunspyce_coord(shdr['DATE-AVG'],'psp', system='HEQ', doVelocity=False, doMeters=True)
        shdr['HEQX_OBS'] = aCo[0]
        shdr['HEQY_OBS'] = aCo[1]
        shdr['HEQZ_OBS'] = aCo[2]
        
        aCo = get_sunspyce_lonlat(shdr['DATE-AVG'],'psp', system='HEEQ', doDegrees=True, doMeters=True)
        shdr['HGLT_OBS'] = aCo[2]
        shdr['HGLN_OBS'] = aCo[1]
        
        aCo = get_sunspyce_lonlat(shdr['DATE-AVG'],'psp', system='Carrington', doDegrees=True, doMeters=True)
        shdr['CRLN_OBS'] = aCo[1]
        shdr['CRLT_OBS'] = aCo[2]
        
        # Set solar radius
        shdr['RSUN_REF'] = 6.95508e8
        shdr['RSUN_ARC'] = np.arctan(shdr['RSUN_REF'] / aCo[0]) * 180 / np.pi * 3600
        
        # Calc time for light to reach sc
        shdr['SUN_TIME'] = aCo[0] / 299792458.
        earth_coord = get_sunspyce_coord(shdr['DATE-AVG'], 'EARTH', system='HEQ', doMeters=True, doVelocity=False)
        shdr['EAR_TIME'] = earth_coord[0] / 299892458 - shdr['SUN_TIME']
        shdr['SOLAR_EP'] = get_sunspyce_p0_angle(shdr['DATE-AVG'], 'psp', doDegrees=True)
        # this seems to be set to an int even though its not flagged as such in the spice call?
        shdr['CAR_ROT'] = int(get_sunspyce_carr_rot(shdr['DATE-AVG'], spacecraft='psp') )
    
    return shdr
    

#|----------------------------|
#|--- Get image statistics ---|
#|----------------------------|
def wispr_img_stats(hdr, im):
    """
    Function to get statistics about the image that we will
    save in the header
    
    Input:    
        hdr: the corresponding header

        im: a wispr image
    
    Output:
        hdr: the updated header

        im: the unchanged image

    """
    goodIm = im[np.isfinite(im)]
    goodIm = goodIm[np.where(goodIm !=0)]
    if len(goodIm) > 0:
        hdr['datamin'] = np.min(goodIm)
        hdr['datamax'] = np.max(goodIm)
        hdr['dataavg'] = np.mean(goodIm)
        hdr['datasig'] = np.std(goodIm)
        hdr['datasat'] = 1 # not sure on this
        hdr['dsatval'] = hdr['datamax'] 

        pvals = [1,10,25,50, 75,90,95,98,99]
        pnames = ['datap01', 'datap10', 'datap25', 'datap50', 'datap75', 'datap90', 'datap95', 'datap98', 'datap99']
        for i in range(9):
            hdr[pnames[i]] = np.percentile(goodIm, pvals[i])

    return hdr, im


#|---------------------------|
#|--- Get straylight info ---|
#|---------------------------|
def wispr_straylight(hdr, im, dn=False):
    """
    Function to perform a simple straylight correction
    for wispr observations based on dsun_obs from
    the image header
    
    Input:    
        hdr: a wispr header

        im: a wispr image
    
    Optional Input:
        dn: a flag to use the calfac version of the correction
            (defaults to false)
    
    Output:
        hdr: the unchanged header

        im: the corrected image

    """
    au = hdr['dsun_obs'] / 1.49578707e11 
    if au <= 0.15:
        sl = 0.75e-14 * au**-3
    else: 
        sl = 0.5e-13 * au **-2
    
    if dn:
        print('Calfac version of wispr straylight untested')
        cf = wispr_get_calfac(hdr)
        sl = sl /cd
    
    im = im - sl
    return hdr, im

#|----------------------------|
#|--- Main wrapper fuction ---|
#|----------------------------|
def wispr_prep(filesIn, wcalpath, outSize=None, silent=False, biasOff=False, biasOffsetOff=False, lin_correct=False, straylightOff=False, pointingOff=False):
    """
    Main wrapper function for preparing wispr data
    
    Input:
        filesIn: a list of fits files for wispr images
    
        wcalpath: path to where the wispr calibration images live
    
    Optional Input:
        outSize: size of the output image (not implemented but keyword retained from IDL)
    
        silent: flag to not print unnecessary information to screen (defaults to False)
    
        biasOff: flag to turn off bias correction (defaults to False)

        biasOffsetOff: flag to turn off bias offset correction (defaults to False)
    
        lin_correct: flag from IDL that has not been ported
    
        straylightOff: flag to turn off straylight correction (defaults to False)
    
        pointingOff: flag to turn off pointing correction (defaults to False)
    
    Output:
        images_out: a list of processed wispr images 
    
        headers_out: a list of corresponding headers

    """
    # Port of basic functionality of IDL version
    
    # Want filesIn as a list, even if single
    if isinstance(filesIn, str):
        filesIn = [filesIn]
        
    # Ignore notupdated, write_flag keywords for now (113-6)
            
    # dont need to pre make arrays bc not idl
    
    num = len(filesIn)
    
    # Ignoreing save path
    
    images_out = []
    headers_out = []
    for i in range(num):
        if not silent:
            print ('Processing image ', i+1, ' out of ', num)
                
        #|---------------------|
        #|--- Read in files ---|
        #|---------------------|
        im, hdr = wispr_readfits(filesIn[i])

        #|-----------------------|
        #|--- Bias sutraction ---|
        #|-----------------------|
        if ~biasOff & hdr['ipbias'] ==0:
            sys.exit(' wispr_ bias needs to be ported')
            # add in bias comment to history
            
        #|------------------------------|
        #|--- Bias offset sutraction ---|
        #|------------------------------|
        # Remove bias offset
        if ~biasOffsetOff & hdr['ipbias'] !=0:
            offset = wispr_bias_offset(im, hdr)
            im = im - offset
            hdr['history'] = 'Subtracted ' + str(offset) +' for image for bias offest'
        # lin_correct not coded bc not called
        
        #|---------------------------|
        #|--- General Correction  ---|
        #|---------------------------|
        im, hdr = wispr_correction(im, hdr, wcalpath)
        
        #|------------------------------|
        #|--- Straylight Correction  ---|
        #|------------------------------|
        if (hdr['detector'] == 2) & (not straylightOff):
            hdr, im = wispr_straylight(hdr, im)
            
        if int(hdr['level'][-1:]) > 1:
            notupdated = False
        
        #|------------------------------|
        #|--- Resizing (not ported)  ---|
        #|------------------------------|
        full = False    
        if hdr['nbin'] == 1:
            full = True
        if (hdr['detector'] == 1 ) & ((hdr['naxis1'] != 1920/hdr['nbin1']) or (hdr['naxis2'] != 2048/hdr['nbin2'])):
            sys.exit('Need to port putin/zelensky array for wispr 1')
        elif (hdr['detector'] == 2 ) & (hdr['ipmask'] ==0) & ((hdr['naxis1'] != 1920/hdr['nbin1']) or (hdr['naxis2'] != 2048/hdr['nbin2'])):
            sys.exit('Need to port putin/zelensky array for wispr 2')
            
        #|-----------------------------|
        #|--- Pointing Information  ---|
        #|-----------------------------|
        if not pointingOff:
            hdr = get_wispr_pointing(hdr, wcalpath, doCoords=True)
        
        im[np.where(im < 0)] = 0
        
        #|--------------------------|
        #|--- Convert DN to MSB  ---|
        #|--------------------------|
        if (hdr['detector'] == 1 ):
            im = im / 3.93e-14 
        elif (hdr['detector'] == 2 ):
            im = im / 5.78e-14

        #|-------------------------|
        #|--- Image Statistics  ---|
        #|-------------------------|
        hdr, im = wispr_img_stats(hdr, im)

        
        #|--------------------------|
        #|--- Package to output  ---|
        #|--------------------------|
        images_out.append(im)
        headers_out.append(hdr)
        
    return images_out, headers_out        
       
