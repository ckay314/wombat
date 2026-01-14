"""
Module for functions related to COR1/COR2 processing that are 
called by secchi_prep. Largely a port of the corresponding
IDL routines and we have kept names matching and indicated
what portions have been left out to facilitate comparison to
the other version. The whole routine tends to be within 0.1% 
accuracy for each pixel with only minor differences introduce
by the difference in IDL and python built in interpretation
routines.

External Calls:
    scc_funs, wcs_funs

"""

import numpy as np
import os
import sys
from astropy.io import fits
from scc_funs import secchi_rectify, fill_from_defhdr, rebinIDL, scc_getbkgimg, scc_sebip
from wcs_funs import get_Suncent, fitshead2wcs
import datetime
from scipy.interpolate import griddata

#|-------------------------|
#|--- Set a few globals ---|
#|-------------------------|

c = np.pi / 180.
cunit2rad = {'arcmin': c / 60.,   'arcsec': c / 3600.,  'mas': c / 3600.e3,  'rad':  1.}

#|---------------------------|
#|-- Get Calibration Image --|
#|---------------------------|
def get_calimg(hdr, calPath, calimg_filename=None, outSize=None):
    """
    Function that pulls the appropriate calibration images for a
    cor image.

    Input:
        hdr: the header of the image for which we want the cal img
    
        calPath: the path to where the calibration images are stored
    
    Optional Input:
        calimg_filename: the name of the calibration image file
    
        outSize: size of the output image (not currently implemented)

    Output:
        cal: the image to use for calibration
    
        hdr: the original header (with modifications?)


    """     
    # Assuming proper header passed. Starting at line 131
    new_flag = True
    HIsum_flag = False
    
    # Create calibration image filename
    #path = calpath
    
    # |--------------------------------------|
    # |--- Use detector to make file name ---| 
    # |--------------------------------------|
    det = hdr['DETECTOR']
    # |--- COR1 ---|
    if det == 'COR1':
        cal_version = '20090723_flatfd'
        obs = hdr['OBSRVTRY']
        obsLet = obs[7].upper()
        tail = '_fCc1'+obsLet+'.fts'
        
    # |--- COR2 ---|
    elif det == 'COR2':
        obs = hdr['OBSRVTRY']
        if obs == 'STEREO_A':
            cal_version = '20060929_vignet'
        else:
            cal_version = '20140723_vignet'
        obsLet = obs[7].upper()
        tail = '_vCc2'+obsLet+'.fts'
        
    # |--- EUVI ---|
    elif det == 'EUVI':
        cal_version = '20060823_wav'
        wave = str(hdr['WAVELNTH']).strip()
        obs = hdr['OBSRVTRY']
        obsLet = obs[7].upper()
        tail = wave+'_fCeu'+obsLet+'.fts'
        
    # |--- HI1 ---|
    elif det == 'HI1':
        obs = hdr['OBSRVTRY']
        obsLet = obs[7].upper()
        if hdr['summed'] == 1:
            cal_version = '20061129_flatfld'
            tail = '_raw_h1'+obsLet+'.fts'
        else:
            cal_version = '20100421_flatfld'    
            tail = '_sum_h1'+obsLet+'.fts'
            HIsum_flag = True
            
    # |--- HI2 ---|
    elif det == 'HI2':
        obs = hdr['OBSRVTRY']
        obsLet = obs[7].upper()
        if hdr['summed'] == 1:
            cal_version = '20150701_flatfld'
            tail = '_raw_h2'+obsLet+'.fts'  
        else:
            cal_version = '20150701_flatfld'
            tail = '_sum_h2'+obsLet+'.fts'
            HIsum_flag = True
            
    # |--- Unknown, must exit ---|
    else:
        print ('DETECTOR could not be found')
        sys.exit()
    
        
    # |---------------------------------------|
    # |--- Build the file name and load it ---| 
    # |---------------------------------------|
    filename = calPath+cal_version+tail
    
    # Check if we were passed a file name
    if calimg_filename:
        filename = calimg_filename
        
    # IDL checks if this filename is same as cal_img from common block
    # so it doesn't redo. We're just gonna load it as new
    if new_flag:
        if os.path.exists(filename):
            with fits.open(filename) as hdulist:
                cal_image =  hdulist[0].data
                cal_hdr   =  hdulist[0].header
            cal_filename = filename
        else:
            sys.exit('Cannot locate calibration image '+filename)
        # Make sure the calibration header has all the keywords. Looks like
        # it's all the default values but some tags missing which breaks things
        cal_hdr = fill_from_defhdr(cal_hdr)
        
        
    # |--------------------------------------|
    # |--- Trim and correct the cal image ---| 
    # |--------------------------------------|
    # |--- Trim calibration image to CCD coordinates ---|
    if cal_hdr['P1COL'] <= 1:
        if HIsum_flag:
            x1 = 25    #(hdr.P1COL-1)/2 ********HACK******** (copied IDL comments)
            x2 = 1048  #(hdr.P2COL-1)/2  data problem in pipeline
            y1 = 0     #(hdr.P1ROW-1)/2
            y2 = 1023  #hdr.P2ROW-1)/2
        else:
            x1 = 50
            x2 = 2047+50
            y1 = 0
            y2 = 2047
        cal = cal_image[y1:y2+1,x1:x2+1] # think need to account for noninclusive pythong
    else:
        cal = cal_image
    
    # |--- Correct calibrage image for rectification ---|
    calRect = True
    if cal_hdr['RECTIFY'] in [False, 'F']: False
    if (hdr['RECTIFY']) and calRect:
        cal, cal_hdr = secchi_rectify(cal, cal_hdr)
    
    # |--- Correct callibration image for rescale -> HI ---|
    if HIsum_flag:
        if hdr['summed'] < 2:
            ssum = 1
        else:
            ssum = 2**(hdr['summed']-2)
    else:
        ssum = 2**(hdr['summed']-1)
     
    # |--- Rebin if cal isn't same shape as source im. ---|
    s = cal.shape
    if ssum != 1:
        cal = rebinIDL(cal, np.array([int(s[0]/ssum), int(s[1]/ssum)]))
    
    # Add in returning filename?
    
    # |--- Rotate if in certain time range ---| 
    dobs = None
    post_conj = False
    if 'date-obs' in hdr:
        dobs = hdr['date-obs']
    elif 'date_obs' in hdr:
        dobs = hdr['date_obs']
    dtobs = datetime.datetime.strptime(dobs,"%Y-%m-%dT%H:%M:%S.%f")
    cut1 = datetime.datetime(2015,5,19)
    cut2 = datetime.datetime(2023,8,12)
    if (dtobs > cut1) & (dtobs < cut2):
       cal = np.rot90(cal, k=2)
                 
    return cal, hdr
    
#|------------------------------|
#|--- Get Calibration Factor ---|
#|------------------------------|
def get_calfac(hdr):
    """
    Function that determines a float value calibration factor based on
    the instrument and date (pulled from a header)

    Input:
        hdr: the header of the image for which we want the calfac


    Output:
        calfac: the calibration factor 


    """
    # Assuming passed proper header
    
    # |----------------------------------------|
    # |--- Get init value based on detector ---| 
    # |----------------------------------------|
    det = hdr['detector']
    
    # |--- COR1 ---|
    if det == 'COR1':
        if hdr['obsrvtry'] == 'STEREO_A':
            calfac = 6.578e-11
            t0 = datetime.datetime.strptime('2007-12-01T03:41:48.174', "%Y-%m-%dT%H:%M:%S.%f")
            rate = 0.00648
        if hdr['obsrvtry'] == 'STEREO_B':
            calfac = 7.080e-11
            t0 = datetime.datetime.strptime('2008-01-17T02:20:15.717', "%Y-%m-%dT%H:%M:%S.%f")
            rate = 0.00258 
        myDT = datetime.datetime.strptime(hdr['date-avg'], "%Y-%m-%dT%H:%M:%S.%f")    
        years = (myDT-t0).total_seconds() / (3600.*24*365.25)
        calfac = calfac / (1-rate*years)
        
    # |--- COR2 ---|    
    elif det == 'COR2':
        if hdr['obsrvtry'] == 'STEREO_A':
            calfac = 2.7e-12*0.5
        if hdr['obsrvtry'] == 'STEREO_B':
            calfac = 2.8e-12*0.5
    
    # |--- EUVI ---|        
    elif det == 'EUVI':
        gain = 15.
        wave = hdr['wavelnth']
        calfac = gain * (3.65 * wave) / (13.6 * 911)
    
    # |--- HI1 ---|    
    elif det == 'HI1':
        # |--- STA ---|
        if hdr['OBSRVTRY'] == 'STEREO_A':
            t0 = datetime.datetime.strptime('2011-06-27T00:00:00.000', "%Y-%m-%dT%H:%M:%S.%f")
            myDT = datetime.datetime.strptime(hdr['date-avg'], "%Y-%m-%dT%H:%M:%S.%f") 
            years = (myDT-t0).total_seconds() / (3600.*24*365.25)
            if years < 0: years = 0
            calfac = 3.453e-13 + 5.914e-16*years # Bsun/DN        
        # |--- STB ---|
        if hdr['OBSRVTRY'] == 'STEREO_B':
            t0 = datetime.datetime.strptime('2007-01-01T00:00:00.000', "%Y-%m-%dT%H:%M:%S.%f")
            myDT = datetime.datetime.strptime(hdr['date-avg'], "%Y-%m-%dT%H:%M:%S.%f") 
            years = (myDT-t0).total_seconds() / (3600.*24*365.25)
            if years < 0: years = 0
            calfac = 3.55e-13
            annualchange = 0.001503
            calfac = calfac / (1-annualchange*years)
    
    # |--- HI2 ---|
    elif det == 'HI2':
        # |--- STA ---|
        if hdr['OBSRVTRY'] == 'STEREO_A':
            t0 = datetime.datetime.strptime('2015-01-01T00:00:00.000', "%Y-%m-%dT%H:%M:%S.%f")
            myDT = datetime.datetime.strptime(hdr['date-avg'], "%Y-%m-%dT%H:%M:%S.%f")
            years = (myDT-t0).total_seconds() / (3600.*24*365.25)
            if years < 0:
                calfac = 4.476e-14 + 5.511e-17*years
            else:
                calfac = 4.512e-14 + 7.107e-17*years
        # |--- STB ---|
        if hdr['OBSRVTRY'] == 'STEREO_B':
            t0 = datetime.datetime.strptime('2000-12-31T00:00:00.000', "%Y-%m-%dT%H:%M:%S.%f")
            myDT = datetime.datetime.strptime(hdr['date-avg'], "%Y-%m-%dT%H:%M:%S.%f")
            years = (myDT-t0).total_seconds() / (3600.*24*365.25)
            calfac = 4.293e-14 + 3.014e-17 * years                        
    else:
        sys.exit('Unknown detected in get_calfac')

    # |--- Set in header ---|
    hdr['calfac'] = calfac
    
    # |-----------------------------|
    # |--- IP Summing Correction ---| 
    # |-----------------------------|
    sumcount = 0
    # Correct for IP summing scale factor
    if (hdr['ipsum'] > 1 ) & (calfac != 1):
        divfactor = (2**(hdr['ipsum']-1))**2
        sumcount = hdr['ipsum'] - 1
        hdr['ipsum'] = 1
        calfac = calfac / divfactor
        hdr['history'] = 'get_calfac Divided calfac by '+str(divfactor)+' to account for IPSUM'
        
    # |-----------------------------------|
    # |--- Total Brightness Correction ---| 
    # |-----------------------------------|
    # Apply factor of two for total brightness images that are not double exposures
    if (hdr['polar'] == 1001) & (hdr['seb_prog'] != 'DOUBLE'):
        calfac = 2*calfac
        hdr['history'] = 'get_calfac Applied factor of 2 for total brightness'
    return calfac    

#|--------------------------------|
#|--- (Quick) 2D Interpolation ---|
#|--------------------------------|
def interp2d(xgrid, ygrid, gridvals, x_in, y_in):
    """
    Function to perform 2D interpretation (quickly) from a regular grid bc the built in
    versions are shockingly slow. Based upon interp2d from the googles/aneeshnaik

    Input:
        xgrid: the original x values of the regular grid (1D array)
    
        ygrid: the original y values of the regular grid (1D array)
    
        gridvals: the values on the grid defined by xgrid & ygrid (2D array)
    
        x_in: the x values of the desired output points (1D or scalar)
    
        y_in: the y values of the desired output points (1D or scalar)
    

    Output:
        z: the interpolated image

    """
    
    # |--- Check if want values for scalar or array ---|
    isScalar = False
    if not isinstance(x_in, (list, np.ndarray)):
        isScalar = True
        x_in = np.array([x_in])
        y_in = np.array([y_in])
    
    #|---  Get spacing ---|
    # Assume xgrid/ygrid are evenly spaced
    nx, ny = len(xgrid), len(ygrid)
    dx, dy = xgrid[1] - xgrid[0], ygrid[1] - ygrid[0]
    
    #|--- Take care of edges ---|
    #  Set any points beyond the boundary to the boundary value
    x_in[x_in < xgrid[0]] = xgrid[0]
    y_in[y_in < ygrid[0]] = ygrid[0]
    x_in[x_in > xgrid[-1]] = xgrid[-1]
    y_in[y_in > ygrid[-1]] = ygrid[-1]
    
    #|---  Find indices of neighbor to desired points ---|
    i1 = np.floor((x_in - xgrid[0])/dx).astype(int)
    i1[i1 == (nx-1)] = nx - 2
    i2 = i1 + 1
    j1 = np.floor((y_in - ygrid[0])/dy).astype(int)
    j1[j1 == (ny-1)] = ny - 2
    j2 = j1 + 1
    
    #|---  Get neighbor xy and vals ---|
    x1, x2 = xgrid[i1], xgrid[i2]
    y1, y2 = ygrid[j1], ygrid[j2]
    z11, z21, z12, z22 = gridvals[i1,j1], gridvals[i2,j1], gridvals[i1,j2], gridvals[i2,j2]
    
    #|---  Interpolate ---|
    t11 = z11 * (x2 - x_in) * (y2 - y_in)
    t21 = z21 * (x_in - x1) * (y2 - y_in)
    t12 = z12 * (x2 - x_in) * (y_in - y1)
    t22 = z22 * (x_in - x1) * (y_in - y1)
    z = (t11 + t21 + t12 + t22) / (dx * dy)
    
    #|--- Return as scalar if gave scalar ---|
    if isScalar:
        z = z[0]
    return z

#|----------------------------|
#|--- Warped Triangulation ---|
#|----------------------------|
def warp_tri(xo,yo,xi,yi,img):
    """
    Function that takes an image defined on a regular grid and 
    warps it onto an irregular grid. The regularness is in terms
    of pixel location (not real world coordinates) and can use to
    warp pixels to account for the real world coords warp

    Input:
        xo: x values of output points (irregular grid)  
        yo: y values of output points (irregular grid)
    
        xi: x values of input points (regular grid)
        yi: y values of input points (regular grid)
    
        img: the image/z-values on the regular grid
    

    Output:
        imgOut: the image warped to the points defined by xo/xo
    

    """
    #|--- Define image shape ---|
    nx, ny = img.shape
    
    #|--- Not including TPS option ---|
    
    # |--- Make input mesh grids ---|
    grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))
    
    #|--- Pair output points ---|
    pointsO = [[xo[i], yo[i]] for i in range(len(xo))]
    
    #|---------------------------|
    #|--- Actual warping part ---|
    #|---------------------------|
    # This is same as trigrid calls. Get the coordinates of the 
    # irregular points wrt to the regular system then just 
    # interpolate in the next step
    xt = griddata(pointsO, xi, (grid_x, grid_y), method='linear')
    yt = griddata(pointsO, yi, (grid_x, grid_y), method='linear')
    
    # Use our own interp func bc scipy is weirdly slow for this
    imgOut = interp2d(np.arange(nx), np.arange(nx), np.transpose(img), xt, yt)
    imgOut = imgOut.reshape([nx,nx]) # is match to RegularGridInterpolator
    
    return imgOut
        
#|-------------------------------------|
#|--- COR2 Calibration Main Routine ---|
#|-------------------------------------|
def cor_calibrate(img, hdr,prepDir, outSize=None, sebip_off=False, exptime_off=False, bias_off=False, calimg_off=False, calfac_off=False):
    """
    Main wrapper script for calibrating COR2 images.

    Input:
        img: the image we want to calbrate
        
        hdr: the header of the image we want to calbrate
    
        prepDir: the path where the extra files needed for prep are stored
       
    Optional Input:   
        outSize: size of the output image (not currently implemented)
        
        sebip_off: flag to turn off the seb ip correction (defaults false)
        
        exptime_off: flag to turn off the exposure time correction (defaults false)
    
        bias_off: flag to turn off the bias correction (defaults false)
    
        calfac_off: flag to turn off the calibration factor correction (defaults false)

    Output:
        img: the calibrated image 
        
        hdr: the updated header for the calbrated image 

    """
    #|--- Update hdr with processing info ---|
    newStuff = 'Applied python port of cor_calibrate.pro CK 2025'
    hdr['history'] = newStuff
    
    #|-------------------------|
    #|--- SEB IP correction ---|
    #|-------------------------|
    if not sebip_off:
        img, hdr, sebipFlag = scc_sebip(img, hdr)
    
    #|--------------------------------|
    #|--- Exposure time correction ---|
    #|--------------------------------|
    # Get the exposure time, needed to converts to DN/S
    # Actual exptime adjustment done below
    if not exptime_off:
        exptime = float(hdr['exptime'])
        if exptime != 1.:
            hdr['history'] = 'Exposure Normalized to 1 Second from ' + str(exptime)
    
    #|------------------------|
    #|--- Bias subtraction ---|
    #|------------------------|
    # Just gets the bias, actual adjustment done below
    if bias_off:
        biasmean = 0.
    else:
        biasmean = float(hdr['biasmean'])
        if hdr['ipsum'] > 1:
            biasmean = biasmean * (2** (hdr['ipsum']-1))**2
        if biasmean != 0.:
            hdr['history'] = 'Bias subtracted '+ str(biasmean)
            hdr['OFFSETCR'] = biasmean
            
    
    #|----------------------------------------|
    #|--- Vignetting/flat field correction ---|
    #|----------------------------------------|
    # Again, just grabbing values to use below
    if calimg_off:
        calimg = 1.0
    else:
        calimg, hdr = get_calimg(hdr, prepDir+'calimg/', outSize=outSize)
        hdr['history'] = 'Applied vignetting '
        
    if calfac_off:
        calfac = 1.
    else:    
        calfac = get_calfac(hdr)
 
    #|-----------------------------|
    #|--- Apply all corrections ---|
    #|-----------------------------|
    # This will give div 0 issues. Just zero out the infs for now
    img = ((img - biasmean) * calfac / exptime) / calimg
    img[np.where(calimg == 0)] = 0.
        
    return img, hdr

#|-------------------------------------|
#|--- COR1 Calibration Main Routine ---|
#|-------------------------------------|
def cor1_calibrate(img, hdr, prepDir, outSize=None, sebip_off=False, exptime_off=False, bias_off=False, calimg_off=False, calfac_off=False, bkgimg_off=False):  
    """
    Main wrapper script for calibrating COR1 images.

    Input:
        img: the image we want to calbrate
        
        hdr: the header of the image we want to calbrate
    
        prepDir: the path where the extra files needed for prep are stored
       
    Optional Input:   
        outSize: size of the output image (not currently implemented)
        
        sebip_off: flag to turn off the seb ip correction (defaults false)
        
        exptime_off: flag to turn off the exposure time correction (defaults false)
    
        bias_off: flag to turn off the bias correction (defaults false)
    
        calfac_off: flag to turn off the calibration factor correction (defaults false)
        
        bkgimg_off: flag to turn off background image subtraction (defaults false) 

    Output:
        img: the calibrated image 
        
        hdr: the updated header for the calbrated image 

    """
    #|--- Update hdr with processing info ---|
    newStuff = 'Applied python port of cor_calibrate.pro CK 2025'
    hdr['history'] = newStuff
    
    # assuming no missing
    
    #|-------------------------|
    #|--- SEB IP correction ---|
    #|-------------------------|
    if not sebip_off:
        img, hdr, sebipFlag = scc_sebip(img, hdr)
            
    #|--------------------------------|
    #|--- Exposure time correction ---|
    #|--------------------------------|
    # Get the exposure time, needed to converts to DN/S
    # Actual exptime adjustment done below
    if exptime_off:
        exptime = 1.
    else:
        exptime = float(hdr['exptime'])
        if exptime != 1.:
            hdr['history'] = 'Exposure Normalized to 1 Second from ' + str(exptime)
        
    #|------------------------|
    #|--- Bias subtraction ---|
    #|------------------------|
    # Just gets the bias, actual adjustment done below
    if bias_off:
        biasmean = 0.
    else:
        biasmean = float(hdr['biasmean'])
        if hdr['ipsum'] > 1:
            biasmean = biasmean * (2** (hdr['ipsum']-1))**2
        if biasmean != 0.:
            hdr['history'] = 'Bias subtracted '+ str(biasmean)
            hdr['OFFSETCR'] = biasmean
    
    #|----------------------------------------|
    #|--- Vignetting/flat field correction ---|
    #|----------------------------------------|
    # Again, just grabbing values to use below
    if calimg_off:
        calimg = None
    else:
        calimg, hdr = get_calimg(hdr, prepDir+'calimg/')
        hdr['history'] = 'Applied vignetting '

    #|------------------------------|
    #|--- Background subtraction ---|
    #|------------------------------|
    if bkgimg_off:
        bkgimg = None
    else:
        bkgimg, bhdr = scc_getbkgimg(hdr)
        if hdr['ccdsum'] != bhdr['ccdsum']:
            bkgimg = False
    if bkgimg is not None:
        sumdif = hdr['ipsum'] - bhdr['ipsum']
        if sumdif != 0:
            bkgimg = bkgimg * (4**sumdif)
        if exptime_off:
            bkgimg = bkgimg * hdr['exptime']
        hdr['history'] = 'Background subtracted'    
            
    #|-------------------------------|
    #|--- Photometric calibration ---|
    #|-------------------------------|
    if calfac_off:
        calfac = 1
    else:
        calfac = get_calfac(hdr)
        if calfac != 1:
            hdr['history'] = 'Applied calibration fact'
    
    #|-----------------------------|
    #|--- Apply all corrections ---|
    #|-----------------------------|
    if biasmean != 0:
        img = img - biasmean
    
    if exptime != 1:
        img = img / exptime
    
    if bkgimg is not None:
        img = img - bkgimg
    if calimg is not None:
        img = img / calimg
    # No cosmic correction
    if calfac !=1:
        img = img * calfac
    
    
    return img, hdr
    
#|-------------------------------|
#|--- COR2 Warping Correction ---|
#|-------------------------------|
def cor2_warp(im,hdr):
    """
    Function that corrects for warping in a cor2 image.

    Input:
        im:  the image to warp
    
        hdr: the header of the image 

    Output:
        img: the altered image 
    
        hdr: the modified header 
    
    Notes:
        Presumably this is taking an img from COR2 and correcting
        by warping instrumental effects back to a uniform spacing.
        The IDL comments are confusing, likely bc of differences on
        whether spacing is regular/irregular in terms of pixels vs
        arcsec. We would expect img in to be reg in pixels but irreg
        in arcsec and this func to warp the orig pixel to make reg
        arcsec spacing. Regardless, we match the IDL code in terms 
        of algorithm 


    """
    #|-------------------------------------|
    #|--- Set up grid of control points ---|
    #|-------------------------------------|
    # Get x and y points at every 32 pixels
    gridsize = 32
    w = np.arange(((2048/gridsize)+1)**2)
    y =  (w / ((2048/gridsize) + 1)).astype(int)
    x = w - y*((2048/gridsize)+1)
    x = x * gridsize
    y = y * gridsize
    
    # |--- Only process if we have an obs ---|
    if 'OBSRVTRY' in hdr: 
        #|--- Get sun center in pixels ---|
        my_wcs = fitshead2wcs(hdr) 
        sc = hdr['OBSRVTRY']
        sc = sc[-1]
        scnt = get_Suncent(my_wcs)
        
        # |--- Account for binning/scalefactor ---|
        sumxy = 2**(hdr['summed']-1)
        
        # |--- Get r of regular grid points ---|
        r = np.sqrt((x - sumxy*scnt[0])**2  + (y - sumxy*scnt[1])**2)
        
        # |--- Apply warping functions to r ---|
        if sc == 'A':
            cf = [1.04872e-05, -0.00801293, -0.243670]
        else:
            cf = [1.96029e-05, -0.0201616,   4.68841] 
        r0 = (r + (cf[2]+(cf[1]*r)+(cf[0]*(r*r)))) / sumxy

        # |--- Undo binning correction ---|
        x = x / sumxy
        y = y / sumxy
        
        # |--- Get the xi/yi from warped r ---|
        theta = np.arctan2((y - scnt[1]), (x-scnt[0]))
        xi = r0 * np.cos(theta) + scnt[0]
        yi = r0 * np.sin(theta) + scnt[1]
        
        # |--- Convert from img xi/yi to x/y ---|
        im = warp_tri(x,y,xi,yi,im)
        
        hdr['distcorr'] = True
        hdr['history'] = 'Applied distortion correction'
    
    return im, hdr

#|-------------------------|
#|--- Main Prep Routine ---|
#|-------------------------|
def cor_prep(im, hdr, prepDir, outSize=None, calibrate_off=False, warp_off=False):
    """
    Main processing wrapper for all COR1/COR2 images

    Input:
        img: the image we want to calbrate
        
        hdr: the header of the image we want to calbrate
    
        prepDir: the path where the extra files needed for prep are stored
       
    Optional Input:   
        outSize: size of the output image (not currently implemented)
        
        calibrate_off: flag to turn off the calibration (defaults false)
        
        warp_off: flag to turn off the COR2 warping correction (defaults false)
    
    Output:
        img: the calibrated image 
        
        hdr: the updated header for the calbrated image 
    

    """
    # Assuming passed a nice header 
    
    # Skipping to 174
    # Not hitting cosmic ray for now (174-178)
    
    # Not hitting missing blocks for now (180-183)
    
    #|--------------------------------|
    #|--- Call calibration routine ---|
    #|--------------------------------|
    if not calibrate_off:
        if hdr['detector'] == 'COR1':
            im, hdr = cor1_calibrate(im, hdr, prepDir, outSize=outSize)
        else:
            im, hdr = cor_calibrate(im, hdr, prepDir, outSize=outSize)
    
    # Not hitting Missing block mask (193-199)
    missing = 0
    
    #|----------------------------|
    #|--- Call warping routine ---|
    #|----------------------------|
    nowarp = False
    if warp_off or calibrate_off:
        nowarp = True
    if (hdr['detector'] == 'COR2') & ~nowarp:
        aFile = hdr['att_file']
        gterr = aFile[-3:-2]
        if (gterr != '2') & (gterr != '+'):
            print ('Havent implemented cor2_point yet')
            print (Quit)
        im, hdr = cor2_warp(im,hdr)        
    
    # Skipping rotate for now
    # Skipping color table
    # Skipping updating header
    
    #|------------------------|
    #|--- Bonus Correction ---|
    #|------------------------|
    # Correct for in-flight calbration Bug 232
    if (hdr['detector'] == 'COR2'):
        hdr['CDELT1'] = 14.7*2**(hdr['SUMMED']-1)
        hdr['CDELT2'] = 14.7*2**(hdr['SUMMED']-1)
        
    # No date/logo adding
    
    return im, hdr
           
#|-------------------------|
#|--- Process Polarized ---|
#|-------------------------|
def cor_polarize(seq, seq_hdr, doPB=False, doPolAng=False):
    """
    Function to process a series of polarized COR images
    
    Input:
        seq: a sequence of 3 polarized images at different angles
    
        hdr: the corresponding image headers 
    
    Optional Input:
        doPB: flag to calc polarization brightness (defaults false)
    
        doPolAng: flag to calc polarization angle (defaults false)

    Output:
        im: the total brightness calculated from the pol images 
            if doPB is flagged it returns the polarization brightness
            if doPolAng is flagged it returns the polarized angle
    
        hdr: the corresponding header 
    
    Notes:
        The code assumes an appropriate set of images are passed


    """
    # Skipping the checks on polar angles and times
    
    #|--------------------------|
    #|--- Get Mueller Matrix ---|
    #|--------------------------|
    angle1 = seq_hdr[0]['polar'] * np.pi/180.
    angle2 = seq_hdr[1]['polar'] * np.pi/180.
    angle3 = seq_hdr[2]['polar'] * np.pi/180.
    s, d   = 0.5, 0.5 
    x = [[s, d*np.cos(2*angle1), d*np.sin(2*angle1)], [s, d*np.cos(2*angle2), d*np.sin(2*angle2)], [s, d*np.cos(2*angle3), d*np.sin(2*angle3)]]
    # Invert it
    mueller = np.linalg.inv(x)
    
    #|------------------------------|
    #|--- Calc Stokes Parameters ---|
    #|------------------------------|
    im1 = seq[0]
    im2 = seq[1]
    im3 = seq[2]
    
    I = im1*mueller[0,0] + im2*mueller[0,1] + im3*mueller[0,2] # total brigtness
    Q = im1*mueller[1,0] + im2*mueller[1,1] + im3*mueller[1,2] 
    U = im1*mueller[2,0] + im2*mueller[2,1] + im3*mueller[2,2]
    
    pbim = np.sqrt(Q**2+U**2)
    
    #|-------------------------------|
    #|--- Pick output img by type ---|
    #|-------------------------------|
    if doPB:
        im = pbim
        polval = 1002
        fnstr = 'P'
    elif doPolAng:
        im = 0.5 * np.arctan2(U, Q)
        polval = 1004
        fnstr = 'A'
        unit = 'radians'
    else:
        im = I
        polval = 1001
        fnstr = 'B'
        
    # Skipping percent polarization
    
    #|-----------------------------|
    #|--- Updating header party ---|
    #|-----------------------------|
    # Gonna assume 0 polar angle is earliest time
    angs = np.array([angle1, angle2, angle3])
    idx0 = np.where(angs == 0)[0][0]
    idxf = np.where(angs == 240*np.pi/180)[0][0]
    idxm = np.where(angs == 120*np.pi/180)[0][0]
    hdr = seq_hdr[idx0]

    # Do all the averaging, which is probably mostly unneccesary
    # but following IDL
    hdr['expcmd']	= (seq_hdr[0]['expcmd'] + seq_hdr[1]['expcmd'] + seq_hdr[2]['expcmd']) / 3.
    hdr['exptime']	= -1.			# N/A for polarized product images
    hdr['biasmean']	= (seq_hdr[0]['biasmean'] + seq_hdr[1]['biasmean'] + seq_hdr[2]['biasmean']) / 3.
    hdr['biassdev'] = np.max(np.array([seq_hdr[0]['biassdev'], seq_hdr[1]['biassdev'], seq_hdr[2]['biassdev']]))
    hdr['ceb_t']	= (seq_hdr[0]['ceb_t'] + seq_hdr[1]['ceb_t'] + seq_hdr[2]['ceb_t']) / 3.
    hdr['temp_ccd']	= (seq_hdr[0]['temp_ccd'] + seq_hdr[1]['temp_ccd'] + seq_hdr[2]['temp_ccd']) / 3.
    hdr['readtime']	= (seq_hdr[0]['readtime'] + seq_hdr[1]['readtime'] + seq_hdr[2]['readtime']) / 3.
    hdr['cleartim']	= (seq_hdr[0]['cleartim'] + seq_hdr[1]['cleartim'] + seq_hdr[2]['cleartim']) / 3.
    hdr['ip_time']	= np.sum(seq_hdr[0]['ip_time'] + seq_hdr[1]['ip_time'] + seq_hdr[2]['ip_time']) / 3.
    hdr['compfact']	= (seq_hdr[0]['compfact'] + seq_hdr[1]['compfact'] + seq_hdr[2]['compfact']) / 3.
    hdr['nmissing']	= np.sum(seq_hdr[0]['nmissing'] + seq_hdr[1]['nmissing'] + seq_hdr[2]['nmissing']) / 3.
    hdr['tempaft1']	= (seq_hdr[0]['tempaft1'] + seq_hdr[1]['tempaft1'] + seq_hdr[2]['tempaft1']) / 3.
    hdr['tempaft2']	= (seq_hdr[0]['tempaft2'] + seq_hdr[1]['tempaft2'] + seq_hdr[2]['tempaft2']) / 3.
    hdr['tempmid1']	= (seq_hdr[0]['tempmid1'] + seq_hdr[1]['tempmid1'] + seq_hdr[2]['tempmid1']) / 3.
    hdr['tempmid2']	= (seq_hdr[0]['tempmid2'] + seq_hdr[1]['tempmid2'] + seq_hdr[2]['tempmid2']) / 3.
    hdr['tempfwd1']	= (seq_hdr[0]['tempfwd1'] + seq_hdr[1]['tempfwd1'] + seq_hdr[2]['tempfwd1']) / 3.
    hdr['tempfwd2']	= (seq_hdr[0]['tempfwd2'] + seq_hdr[1]['tempfwd2'] + seq_hdr[2]['tempfwd2']) / 3.
    hdr['tempthrm']	= (seq_hdr[0]['tempthrm'] + seq_hdr[1]['tempthrm'] + seq_hdr[2]['tempthrm']) / 3.
    hdr['temp_ceb']	= (seq_hdr[0]['temp_ceb'] + seq_hdr[1]['temp_ceb'] + seq_hdr[2]['temp_ceb']) / 3.
    hdr['date-end'] = seq_hdr[idxf]['date-avg'] # Not exact to IDL but not likely relevant
    hdr['date-avg'] = seq_hdr[idxm]['date-avg'] # Not exact to IDL but not likely relevant
    hdr['N_IMAGES'] = 3
    hdr['ENCODERP'] = -1
    hdr['CROTA']	= (seq_hdr[0]['CROTA'] + seq_hdr[1]['CROTA'] + seq_hdr[2]['CROTA']) / 3.
    hdr['PC1_1']	= np.cos(hdr['CROTA']*np.pi/180)
    hdr['PC1_2']	= -np.sin(hdr['CROTA']*np.pi/180) 
    hdr['PC2_1']	= np.sin(hdr['CROTA']*np.pi/180)
    hdr['PC2_2']	= np.cos(hdr['CROTA']*np.pi/180)
    
    # Skipping some comments in the header
    hdr['POLAR'] = polval
    if doPolAng: hdr['bunit'] = unit
    
    hdr['filename'] = hdr['filename'].replace('n4c', '1'+fnstr+'4c')
    
    return im, hdr