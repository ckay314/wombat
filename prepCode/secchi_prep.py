"""
Module for processing SECCHI observations. Largely a wrapper that
opens up the fits files and passes to the appropriate prep
routines for each instrument. We have kept names matching 
and indicated what portions have been left out to facilitate 
comparison to the other version. Kept at a near match to the
IDL routines on a pixel by pixel basis.

"""

import numpy as np
import sunpy.map
import sys
from scc_funs import scc_make_array, scc_zelensky_array, scc_img_trim
from euvi_prep import euvi_prep
from cor_prep import cor_prep, cor_polarize
from hi_prep import hi_prep
from astropy.io import fits


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


#|----------------------------|
#|--- Main Wrapper Function ---|
#|-----------------------------|
def secchi_prep(filesIn, outSize=None, silent=False, polarizeOn=False, prepDir=None):
    # Port of the basic functionality of IDL version
    # For polarized images need to just pass three at a time
    
    # Want filesIn as a list, even if single
    if isinstance(filesIn, str):
        filesIn = [filesIn]
    
    # Actual code starts at 256
    
    # common block loads comments (256)
    # prints info if not silent (259-261)
    # check if given inputs in right form (263-275)
    
    # |------------------------------------------------------|
    # |---------- Set up the run based on keywords ----------|
    # |------------------------------------------------------|
    # Create the keyword structure (277) that is used to send keywords
    # to the sub routines. Not really using yet bc ignoring most keywords
    ex = {}
    ex['blank'] = -1
    
    # Check the write_on keyword for type, ignoring for now
    # but might want later (279 - 288)
    
    # Check for the trim_off keyword, ignoring for now
    # but might want later (288-291)
    
    # Mega chunk of more keywords not using now (293 - 315)
 
    # 326 check for polarization just sets up memory things so skip
    
    # |------------------------------------------------------|
    # |------- Create array to return images to memory ------|
    # |------------- (no clue what that means) --------------|
    # Might have outsize keyword passed, don't have to explicitly check in Python
    # This sets up empty arrays and an empty header dict
    # -> probably don't need this bc python doesn't allot memory like IDL
    images, headers, outout, outsize, out = scc_make_array(filesIn, outSize=outSize)
    ex['outsize'] = outsize

    # |------------------------------------------------------|
    # |------------- Loop to process each image -------------|
    # |------------------------------------------------------|
    num = len(filesIn)
    images_out = []
    headers_out = []
    
    for i in range(num):
        
        # Skipping error catching in 344-359
        if not silent:
            print ('Processing image '+ str(i+1) +' out of ' + str(num))
            
        # im = SCCREADFITS(filenames[i],hdr,silent=silent,header=str_hdr)
        # Going to attempt using astropy fits instead of full port, tbd on success
        with fits.open(filesIn[i]) as hdulist:
            im  = hdulist[0].data.astype(np.int64)
            hdr = hdulist[0].header
                
        # Random cases that have been ported but not tested
        if 'calfac' in hdr: #COR2B test case doesn't have       
            if hdr['calfac'] == 0.:
                hdr['calfac'] = 1.
        if not hdr['rectify']:
            hdr['r1col']=hdr['p1col']
            hdr['r2col']=hdr['p2col']
            hdr['r1row']=hdr['p1row']
            hdr['r2row']=hdr['p2row']
        # Not including keyword rectify for now

        # Skipping pre commissioning since we don't seem to hit (383 - 404)
        det = hdr['DETECTOR']
        if det == 'EUVI':
            # Only one of our cases that hits this is EUVI
            gtFile = prepDir + 'secchi_gtdbase.geny'
            im, hdr = scc_img_trim(im, hdr, gtFile=gtFile)
        # Not hitting trimming (405 - 408)
                
        # Rescale image to fit output size
        #im = SCC_PUTIN_ARRAY(im,hdr,outout,_extra=ex)
        im, hdr = scc_zelensky_array(im, hdr, outout, out)

        # Not appliying discri_pobk (413-425)
        # Check if level 1 image
        let = hdr['filename'][16]
        encoded_bytes = let.encode(encoding='utf-8')
        levelb = encoded_bytes[0]
        if (levelb <= 57) & (levelb >= 48):
            print ('Calibration already applied')
            ex['calibrate_off'] = True
            ex['calfac_off'] = True
        else:
            ex['calibrate_off'] = False
            ex['calfac_off'] = False
                    
                
        # |------------------------------------------------------|
        # |----------------- Begin Calibration ------------------|
        # |------------------------------------------------------|
        det = hdr['DETECTOR']
                
        if det == 'EUVI':
            im, hdr = euvi_prep(im, hdr, prepDir)
        elif det == 'COR1':
            im, hdr = cor_prep(im, hdr, prepDir, outSize)        
            # Do polarization below after looped
            
        elif det == 'COR2':
            im, hdr = cor_prep(im, hdr, prepDir, outSize)
             
            # No polarization for now
        elif det == 'HI1':
            im, hdr = hi_prep(im, hdr, prepDir)
        elif det == 'HI2':
            im, hdr = hi_prep(im, hdr, prepDir)
            
        # IP summing already done it seems
                
        # Return the things
        images_out.append(im)
        headers_out.append(hdr)
        
    if polarizeOn:
        seq = images_out[::-1] # is reversed in IDL so do it here too
        seq_hdr = headers_out[::-1]
        im, hdr = cor_polarize(seq, seq_hdr)
        images_out = [im]
        headers_out = [hdr]
        
    return images_out, headers_out
                
  
