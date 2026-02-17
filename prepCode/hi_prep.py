"""
Module for functions related to HI processing that are 
called by secchi_prep. Largely a port of the corresponding
IDL routines and we have kept names matching and indicated
what portions have been left out to facilitate comparison to
the other version. 

External Calls:
    scc_funs, cor_prep

"""
import numpy as np
import os
import sys
from astropy.io import fits
import datetime
import scipy
from scipy.interpolate import griddata
from scc_funs import scc_sebip, scc_hi_diffuse, scc_getbkgimg
from cor_prep import get_calfac, get_calimg

import matplotlib.pyplot as plt

#|-----------------------------|
#|--- Get date for pointing ---|
#|-----------------------------|
def hi_read_pointing(fle):
    """
    Function that takes a pointing file and returns a list of 
    headers in date order

    Input:
        fle: a pointing file

    Output:
        outs: a list of headers in date order 


    """
    #|--- Open the pointing file ---|
    with fits.open(fle) as hdulist:
        #|--- Loop through list and grab each date ---|
        nxt = len(hdulist) 
        allDates = []
        for i in range(nxt-1):
            hdr = hdulist[i+1].header 
            if 'date-avg' in hdr: hdr['DATE_AVG'] = hdr['date-avg']
            if 'DATE-AVG' in hdr: hdr['DATE_AVG'] = hdr['date-avg']
            allDates.append(hdr['date-avg'])
        
        #|--- Sort by dates and return ---|
        ordIdx = np.argsort(allDates)
        ordIdx = [int(i) for i in ordIdx]
        outs = [hdulist[i+1].header for i in ordIdx]
        return outs

#|----------------------------|
#|--- Correct the pointing ---|
#|----------------------------|
def hi_fix_pointing(hdr, prepDir, hipointfile=None, ravg=None, tvary=False):
    """
    Function to correct the pointing in an HI header

    Input:
        hdr: a header 
    
        prepDir: the path to where the prep files are stored
    
    Optional Input:
        hipointfile: the name of the pointing file (not implemented)
    
        ravg: sets rtmp value
    
        tvary: flag to not use the fix_mu_fov correction file (defaults to false)

    Output:
        hdr: the modified header 
    
    Notes:
        Only base version ported to suit needs of wombat processing

    """     
    # |------------------------------------|
    # |--- Assume not given hipointfile ---| 
    # |------------------------------------|
    if hipointfile == None:
        myDir = prepDir + 'hi/'
        
        # |--- Set the rtmp criteria ---|
        rtmp = 5
        if ravg != None: rtmp = ravg
        
        # |--- Get pointing file name ---|
        yymmdd = hdr['date-avg'][:10]
        if tvary:
            fle = myDir + 'pnt_' + hdr['detector'] + hdr['obsrvtry'][7] + '_' + yymmdd + '.fts'
        else:
            fle = myDir + 'pnt_' + hdr['detector'] + hdr['obsrvtry'][7] + '_' + yymmdd + '_fix_mu_fov.fts' 
        
        # |-------------------------------|
        # |--- Process the hipointfile ---| 
        # |-------------------------------|
        if os.path.exists(fle):
            hipoint = hi_read_pointing(fle)
            ec = -1
            for i in range(len(hipoint)):
                aHdr = hipoint[i]
                if aHdr['extname'] == hdr['date-avg']:
                    ec = i
            # |-----------------------------------------|
            # |--- Pick the appropriate pointing hdr ---| 
            # |-----------------------------------------|        
            if ec != -1:
                pHdr = hipoint[ec] 
                stcravg = pHdr['ravg']
                stcnst1 = pHdr['nst1']
                
                if hdr['naxis1'] != 0:
                    sumdif = round(hdr['cdelt1'] / pHdr['cdelt1'])
                else:
                    sumdif = 1
                    
                if stcnst1 < 20:
                    print ('Assuming subfield in hi_fix_pointing, but havent ported this')
                    print (Quit)
                # |--------------------------------------------|
                # |--- Replace hdr values using poining hdr ---| 
                # |--------------------------------------------|
                else:
                    if (pHdr['ravg'] < rtmp) & (pHdr['ravg'] >= 0.):
                        hdr['crval1a'] = pHdr['crval1a']
                        hdr['crval2a'] = pHdr['crval2a']
                        hdr['pc1_1a'] = pHdr['pc1_1a']
                        hdr['pc1_2a'] = pHdr['pc1_2a']
                        hdr['pc2_1a'] = pHdr['pc2_1a']
                        hdr['pc2_2a'] = pHdr['pc2_2a']
                        hdr['cdelt1a'] = pHdr['cdelt1a']*sumdif
                        hdr['cdelt2a'] = pHdr['cdelt2a']*sumdif
                        hdr['pv2_1a'] = pHdr['pv2_1a']
                        hdr['crval1'] = pHdr['crval1']
                        hdr['crval2'] = pHdr['crval2']
                        hdr['pc1_1'] = pHdr['pc1_1']
                        hdr['pc1_2'] = pHdr['pc1_2']
                        hdr['pc2_1'] = pHdr['pc2_1']
                        hdr['pc2_2'] = pHdr['pc2_2']
                        hdr['cdelt1'] = pHdr['cdelt1']*sumdif
                        hdr['cdelt2'] = pHdr['cdelt2']*sumdif
                        hdr['pv2_1'] = pHdr['pv2_1']
                        hdr['xcen'] = pHdr['xcen']
                        hdr['ycen'] = pHdr['ycen']
                        hdr['crota'] = pHdr['crota']
                        hdr['ins_x0'] = pHdr['ins_x0']
                        hdr['ins_y0'] = pHdr['ins_y0']
                        hdr['ins_r0'] = pHdr['ins_r0']
                        hdr['ravg'] = pHdr['ravg']
                    else:
                        print('r_avg criteria not met in hi_fix_pointing, but havent ported this')
                        print (Quit)
            else:
                print('No pointing calibration data found in hi_fix_pointing, but havent ported this')
                print (Quit)        
    return hdr

#|------------------------|
#|--- Invert something ---|
#|------------------------|
def sc_inverse(n,diag, below, above):
    """
    Function to inverse smearing matrix (?)

    Input:
        n: the size of the inverted axis
    
        diag: exp_eff as calculated in hi_desmear
    
        below: ?
    
        above: ?

    Output:
        p: a matrix used by hi_desmear
    
    Notes:
        Ported and matches IDL, but not particularly well-understood 

    """
    wt_above = above / diag
    wt_below = below / diag
    
    wt_above1 = wt_above - 1
    wt_below1 = wt_below - 1
    
    ints = np.array(range(n-2))+1
    power_above = np.zeros(n-1)
    power_below = np.zeros(n-1)
    
    power_above[0] = 1
    power_below[0] = 1
    
    # Can do w/o for loop in python
    power_above[1:] = wt_above1 ** ints
    power_below[1:] = wt_below1 ** ints
    
    v, u = np.zeros(n), np.zeros(n)
    v[1:] = wt_below * (power_below * power_above[::-1])
    u[1:] = wt_above * (power_above * power_below[::-1])

    d = -u[1] / wt_above - (np.sum(v)-v[n-1])
    f = 1. / (diag * (d + wt_above*np.sum(v)))
    
    u[0], v[0] = d, d
    u = u*f
    v = v[::-1] * f
    p = np.empty([n,n])
    # set up p the same as IDL, transverse at tend
    # IDL indexing is very non-pythonic, think just give it starting index?
    p[:,0] = u[0] 
    for row in np.array(range(n-3))+1:
        p[:row,row] = v[n-row-1:n-1]
        p[row:,row] = u[0:n-row]
    p[:,-1] = v

    # now in python indexing which is oppo of IDL
    p = np.transpose(p)

    return p

#|-------------------------|
#|--- Remove saturation ---|
#|-------------------------|
def hi_remove_saturation(im, hdr, saturation_limit=None, nsaturated=None):
    """
    Function to remove saturated points

    Input:
        im: an HI image
    
        hdr: the header of the image
    
    Optional Input:
        saturation_limit: the limit for saturation
    
        nsaturated: the number of saturated points

    Output:
        p: a matrix used by hi_desmear
    
    Notes:
        Ported and matches IDL, but not particularly well-understood 

    """
    # ignoring header check
    if saturation_limit == None: saturation_limit = 14000
    if saturation_limit < 0:
        return im, hdr
    # IDL has but not needed with how we use colmask
    #if nsaturated == None: nsaturated = 5
    
    #|-------------------------------------|
    #|--- Calc corrected saturation val ---|
    #|-------------------------------------|
    n_im = hdr['imgseq'] + 1
    ssum = hdr['summed']
    dsatval = saturation_limit * n_im*(2.**(ssum-1))**2
    
    #|-----------------------|
    #|--- Find saturation ---|
    #|-----------------------|
    ii = np.where(im > dsatval)
    nii = len(ii)
    
    #|------------------------------|
    #|--- Mask out saturated val ---|
    #|------------------------------|
    if nii > 0:
        mask = np.copy(im) * 0
        mask[ii] = 1
        colmask = np.sum(mask, axis=0)
        cols = np.unique(ii[1])
        satCols = []
        for col in cols:
            if colmask[col] > dsatval:
                im[:,ii] = np.nan
        return im, hdr
    else:
        return im, hdr

#|-----------------------|
#|--- Get the cosmics ---|
#|-----------------------|
def hi_cosmics(im, hdr):
    """
    Function that gets the cosmic ray scrub report 

    Input:
        im: a HI image
        
        hdr: the header of the image
         
    Output:
        im: the image with the cosmic report filled by
            the adjacent row values
    
        hdr: the corresponding header 
    
        cosmics: the cosmic ray info


    """     
    #|---------------------------------|
    #|--- Pull cosmics based on hdr ---|
    #|---------------------------------|
    if ('s4h' not in hdr['filename']):
        cosmics = hdr['cosmics']
    elif (hdr['n_images'] < 1) & (hdr['imgseq'] < 1):
        cosmics = hdr['cosmics']
    else:
        count = hdr['imgseq'] + 1
    
        #|-----------------------------|
        #|--- Determine if inverted ---|
        #|-----------------------------|
        inverted = False
        if hdr['rectify']:
            if (hdr['date-obs'] > '2015-07-01T00:00:00') & (hdr['date-obs'] < '2023-08-12T00:00:00'):
                inverted = hdr['OBSRVTRY'] == 'STEREO_A'
            else:
                inverted = hdr['OBSRVTRY'] == 'STEREO_B'
                
        #|------------------------|
        #|--- Process Inverted ---|
        #|------------------------|
        if inverted:
            cosmic_counter = im[0,count]
            if cosmic_counter == count:
                cosmics = im[0,:count][::-1]
            else:
                print ('hit un ported part in hi_cosmics, need to do')
                print (Quit)
        
        #|----------------------------|
        #|--- Process Non-Inverted ---|
        #|----------------------------|
        else:
            naxis1 = hdr['naxis1']
            naxis2 = hdr['naxis2']
            # gotta switch axes for python
            cosmic_counter = im[naxis2-1, naxis1-count-1]
            if cosmic_counter == count:
                cosmics = np.copy(im[naxis2-1,naxis1-count:naxis1])
                # fill image from row below
                im[naxis2-1,naxis1-count-1:naxis1]=im[naxis2-2, naxis1-count-1:naxis1]
            else:
                print ('hit un ported part in hi_cosmics, need to do')
                print (Quit)
    return im, hdr, cosmics

#|---------------------------|
#|--- Desmear Observation ---|
#|---------------------------|
def hi_desmear(im,hdr):
    """
    Function corrects for smearing

    Input:
        im: the image to correct (in DN units)
    
        hdr: the corresponding header
    
    Output:
        im: the corrected image (in DN/s)
    
        hdr: the updated header

    """     
    #|---------------------------|
    #|--- Check header values ---|
    #|---------------------------|
    if hdr['CLEARTIM'] < 0:
        sys.exit('CLEARTIM invalid in header. Cannot desmear')
    if hdr['RO_DELAY'] < 0:
        sys.exit('RO_DELAY invalid in header. Cannot desmear')
    if hdr['LINE_CLR'] < 0:
        sys.exit('LINE_CLR invalid in header. Cannot desmear')
    if hdr['LINE_RO'] < 0:
        sys.exit('LINE_RO invalid in header. Cannot desmear')
        
    #|------------------------------|
    #|--- Post-Conjunction Check ---|
    #|------------------------------|
    date_obs = hdr['date-obs']    
    post_conj = False
    if (date_obs > '2015-07-01T00:00:00') & (date_obs < '2023-08-12T00:00:00'):
        post_conj = True
    
    #|---------------------------|
    #|--- Check for underscan ---|
    #|---------------------------|
    # Extract image array if underscan present.
    if (hdr['dstart1'] < 1) or (hdr['naxis1'] == hdr['naxis2']):
        image = im
    else:
        print ('Unchecked line in hi_desmear AAA')
        image = im[hdr['dstart2']-1:hdr['dstop2'],hdr['dstart1']-1:hdr['dstop1']]
    
    #|-------------------------------|
    #|--- Compute inv corr matrix ---|
    #|-------------------------------|
    clearest=0.70
    exp_eff = hdr['EXPTIME'] + hdr['n_images'] * (clearest-hdr['CLEARTIM'] + hdr['RO_DELAY'])
    
    # Weight correction by number of images and some other words
    dataWeight = hdr['n_images'] * ((2**(hdr['ipsum']-1)))

    #|---------------------------|
    #|--- Determine inversion ---|
    #|---------------------------|
    inverted = False
    if hdr['rectify']:
        if hdr['OBSRVTRY'] == 'STEREO_B': inverted = True
        if (hdr['OBSRVTRY'] == 'STEREO_A') & (post_conj): inverted = True
    
    #|-------------------------|
    #|--- Invert the matrix ---|
    #|-------------------------|
    if inverted:
        fixup = sc_inverse(hdr['naxis2'], exp_eff, dataWeight*hdr['line_clr'], dataWeight*hdr['line_ro'])
    else:
        fixup = sc_inverse(hdr['naxis2'], exp_eff, dataWeight*hdr['line_ro'], dataWeight*hdr['line_clr'])
    
    #|-------------------------|
    #|--- Correct the image ---|
    #|-------------------------|    
    image = np.matmul(fixup, image)
    # patch the repaired image back in if needed
    if (hdr['dstart1'] < 1) or (hdr['naxis1'] == hdr['naxis2']):
        im = image
    else:
        im[hdr['dstart2']-1:hdr['dstop2'],hdr['dstart1']-1:hdr['dstop1']] = image
        
    return im, hdr

#|----------------------|
#|--- Correct Things ---|
#|----------------------|
def hi_correction(im, hdr, prepDir, sebip_off=False, bias_off=False, exptime_off=False, desmear_off=False, calfac_off=False, calimg_off=False):
    """
    Main wrapper for HI image correction

    Input:
        img: the image we want to correct
        
        hdr: the header of the image we want to correct
    
        prepDir: the path where the extra files needed for prep are stored
       
    Optional Input:   
        sebip_off: flag to turn off the seb ip correction (defaults false)

        bias_off: flag to turn off the bias correction (defaults false)
        
        exptime_off: flag to turn off the exposure time correction (defaults false)
    
        desmear_off: flag to turn off the smearing correction (defaults false)
    
        calfac_off: flag to turn off the calibration factor correction (defaults false)

        calimg_off: flag to turn off the calibration image correction (defaults false)

    Output:
        img: the calibrated image 
        
        hdr: the updated header for the calbrated image 

    """     
    #|-------------------------------|
    #|--- Assume passed valid hdr ---|
    #|-------------------------------|
    
    #|-------------------------|
    #|--- SEB IP Correction ---|
    #|-------------------------|
    if not sebip_off:
        im, hdr, sebipFlag  = scc_sebip(im, hdr)


    #|------------------------|
    #|--- Bias subtraction ---|
    #|------------------------|
    if bias_off:
        biasmean = 0.
    else:
        biasmean = float(hdr['biasmean'])
        # Check if done onboard
        ip19str =hdr['IP_00_19']
        if ('103' in ip19str) or ('37' in ip19str) or ('38' in ip19str):
            biasmean = 0
        if hdr['ipsum'] > 1:
            biasmean = biasmean * (2** (hdr['ipsum']-1))**2
        if biasmean != 0.:
            hdr['history'] = 'Bias subtracted '+ str(biasmean)
            hdr['OFFSETCR'] = biasmean
            im = im - biasmean
            
    #|--------------------------------|
    #|--- Cosmics check/correction ---|
    #|--------------------------------|
    im, hdr, cosmics = hi_cosmics(im, hdr)    
    im, hdr = hi_remove_saturation(im, hdr)

    #|--------------------------------|
    #|--- Exposure Time Correction ---|
    #|--------------------------------|
    if not exptime_off:
        if desmear_off:
            print('Need to code desmear off in hi_correction in hi_prep')
            print (Quit)
        else:
            im, hdr = hi_desmear(im, hdr)
            if hdr['nmissing'] > 0:
                print ('Need to code hi_fill_missing')
                print (Quit)
            hdr['bunit'] = 'DN/s'
    
    # capture ipsum
    ipkeep = hdr['ipsum']
    
    #|-------------------------------------|
    #|--- Calibration Factor Correction ---|
    #|-------------------------------------|
    if calfac_off:
        calfac = 1.
    else:
        calfac = get_calfac(hdr)
    
    diffuse = 1.0
    if (calfac != 1):
        if not calimg_off:
            hdr['history'] = 'Applied calibration factor'
            diffuse = scc_hi_diffuse(hdr, ipsum=ipkeep)
            hdr['history'] = 'Applied diffuse source correction'
    else:
        calfac_off = True
    
    #|------------------------------------|
    #|--- Calibration Image Correction ---|
    #|------------------------------------|    
    # Correction for flat field and vignetting
    if calimg_off:
        calimg = 1.0
    else:
        calimg, chdr = get_calimg(hdr, prepDir+'calimg/')
        hdr['history'] = 'Applied flat field'
        
    #|------------------------|
    #|--- Apply Correction ---|
    #|------------------------|
    im = (im * calimg * calfac * diffuse)

    return im, hdr

#|-------------------------|
#|--- Main Prep Routine ---|
#|-------------------------|
def hi_prep(im, hdr, prepDir, calfac_off=False, calimg_off=False):
    """
    Main wrapper for HI image preparation

    Input:
        img: the image we want to correct
        
        hdr: the header of the image we want to correct
    
        prepDir: the path where the extra files needed for prep are stored
    
    Optional Input:   
        calfac_off: flag to turn off the calibration factor correction (defaults false)

        calimg_off: flag to turn off the calibration image correction (defaults false)
    
       
    Output:
        img: the calibrated image 
        
        hdr: the updated header for the calbrated image 

    """     
    #|------------------------------|
    #|--- Check passed valid hdr ---|
    #|------------------------------|
    det = hdr['DETECTOR']
    if det not in ['HI1', 'HI2']:
        sys.exit('hi_prep only for HI detector')
    
    #|----------------------|
    #|--- 'Bugzilla 332' ---|
    #|----------------------|
    if (hdr['naxis1'] > 1024) & (hdr['imgseq'] !=0) & (hdr['n_images'] == 1):
        hdr['imgseq'] = 0
    
    #|----------------------------------|
    #|--- Pass to correction routine ---|
    #|----------------------------------|
    # cosmic seems undefined on first pass then saved after for future calls?
    im, hdr = hi_correction(im, hdr, prepDir, calfac_off=calfac_off, calimg_off=calimg_off)
    
    #|------------------------------|
    #|--- Fix pointing in header ---|
    #|------------------------------|
    # Update header with the best calibrated pointing
    hdr = hi_fix_pointing(hdr, prepDir)
    
    # Smooth mask - HI2 only and/or off
    
    # No color table
    
    # Skipping updating header
    
    # No date/logo adding
    
    return im, hdr
    
def srem(files, side='a', tel='hi_2'):
    # will need monthly min, can pull from https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi_backgrounds/a/monthly_min/201207/
    # using the correct YYYYMM format
    
    # skipping keywords stuff 134 - 150
    ndy = 1
    # Check keywords
    side = side.lower()
    if side not in ['a', 'ahead', 'sta', 'b', 'behind', 'stb']:
        sys.exit('Unknown side passed to srem, pick a or b')
    
    nim = len(files)
    # Skipping part that sorts a/b
    
    # Load the files
    ims, hdrs = [], []
    for i in range(nim):
        with fits.open(files[i]) as hdulist:
            ims.append(hdulist[0].data.astype(np.int64))
            hdrs.append(hdulist[0].header)
        #im, hdr = hi_prep(ims[i], hdrs[i], '/Users/kaycd1/wombat/prepFiles/stereo/')
        #ims[i], hdrs[i] = im, hdr
    
    # Starting at 189
    # Pull first header and check roll
    i0 = 0
    with fits.open(files[i0]) as hdulist:
        #im  = hdulist[0].data#.astype(np.int64)
        hdrs0 = hdulist[0].header
    maxroll = 10
    if hdrs0['crota'] > 180:
        rota = 360 - hdrs0['crota']
    else:
        rota = np.abs(hdrs0['crota'])
    if rota > maxroll:
        sys.exit('First image in srem has crota above maxval. Check next not ported yet')
    
    # Unclear what 204-209 doing, loading things piece by piece for IDL happiness?
    
    modelim, mhdr = scc_getbkgimg(hdrs0, secchi_bkg ='/Users/kaycd1/wombat/STEREObackgrounds/', match=True)
    
    # Dont hit expcorr keyword
    
    # no bmin/max keywords, just take defaults
    bmin = -1000. * (mhdr['exptime'] / hdrs0['exptime'])
    bmax =  1000. * (mhdr['exptime'] / hdrs0['exptime'])
    
    if (tel == 'hi_2'): mfilt = 1 
    
    date = hdrs[0]['date-obs']
    
    if tel in ['hi_1', 'hi_2']:
        t = tel.replace('_', '')
    else:
        t = tel
    
    fid = date[:10].replace('-','') + '_'+t
    sdir = fid + side + '/' # default save directory
    path = 'tmp/' + fid + side

    # Back out corners to mitigate flickering
    if (tel == 'hi_2') and (side == 'b'):
        print ("haven't checked this ")
        mask = np.zeros([hdrs[0]['naxis1'], hdrs[0]['naxis2']])
        s = mask.shape()
        xcen = hdrs[0]['naxis1']/2
        ycen = hdrs[0]['naxis2']/2
        dist = 1.2 * (hdrs[0]['naxis1']/2)**2
        for i in range (s[0]):
            for j in range(s[1]):
                if ((i - xcen)**2 + (j -ycen)**2) < dist:
                    mask[i,j] = 1
                    
    # not doing the usebase loop
    
    # also not doing base skip
    
    # starting at 389
    j = 0
    i = i0
    
    # Might skip some files in the list so need to have separate indices with 
    # i for list subscript and j for loop-action 
    
    # need to loop this... tbd
    # not sure about mod switching indices between 1/0 come back to this
    allIms = []
    allHdrs = []
    for j in range(nim-1):
        print('Making frame', j+1, 'of ', nim)
        hdr1 = hdrs[j]
        hdr2 = hdrs[j+1]
        im1  = ims[j]
        im2  = ims[j+1]
        hdr  = hdr2 # why IDL...
        if np.abs(hdr['crota'] - hdrs0['crota'] > maxroll):
            print ('Skipping ', hdr['filename'], ' because exceeds max roll')
            sys.exit('Need to make it skip to next file, tbd') 
    
        # not doing pos keyword
    
        srem_img, shifts = rdif_hi(im1, im2, hdr, side=side, tel=tel, model=modelim)
    
        # hdr filename not actually being changed...
        hdr['polar'] = 0
    
        # skipping crop
    
        # do B mask, not using pos or deproj keys so will hit
        if (tel == 'hi_2') and (side == 'b'):
            srem_img = mask * srem_img
        
        if 'hi' in tel:
            hdr = hi_fix_pointing(hdr, '/Users/kaycd1/wombat/prepFiles/')
        
        allIms.append(srem_img)
        allHdrs.append(hdr)
    # skipping nosnow keyword 431-439
    
    fig = plt.figure()
    for j in range(nim-1):
        #aIm, hdr = hi_prep(allIms[j], allHdrs[j], '/Users/kaycd1/wombat/prepFiles/stereo/')
        calfac = get_calfac(allHdrs[i])
        plt.imshow(allIms[j]*calfac, vmin=-5e-11, vmax=5e-11, origin='lower', cmap='Greys_r')
        plt.show()


def rdif_hi(im1, im2, hdr, side='a', tel='hi_1', model=None, min=False, shift=True, mfilt=True):
    nreg = 1
    
    side = side.lower()
    tel  = tel.lower()
    if side not in ['a', 'b']:
        sys.exit('Exiting rdif_hi, side must be either a or b')
    if tel == 'hi1':
        tel = 'hi_1'
    elif tel == 'hi2':
        tel = 'hi_2'
    elif tel not in ['hi_1', 'hi_2']:
        sys.exit('Exiting rdif_hi, side must be either hi_1 or hi_2') 
        
    # Doesn't hit secchi prep, just assigns A/B to img1/img2
    
    if type(model) == type(None):
        sys.exit('Need to port secchi background part in rdif_hi')
        
    s = im1.shape
    nx1, nx2 = s[1], s[0] # shape is [nrows, ncols]
    
    # not using pos, model2 = model
    med = np.median(model)
    med1 = np.median(im1)
    med2 = np.median(im2)
    med12 = med1 / med2
    if med12 >= 1: med12 = 1./med12
    if med12 >= 0.97:
        med1, med2 = 1., 1.
    else:
        print ('Doing median normalization')
        med1 = med / med1
        med2 = med / med2
    
    if tel == 'hi_2':
        # skipping deproj
        frame1 = im1 * med1 - model
        frame2 = im2 * med2 - model
    
        szf = frame1.shape
        nx1, nx2 = szf[1], szf[0]
        dum1, dum2 = frame1, frame2
        dum_ant = frame1
        dum_act = frame2
    else:
        print ('hitting non hi2 untested part of rdif_hi')
        szf = frame1.shape
        nx1, nx2 = szf[1], szf[0]
        frame1 = img1*med1
        dum1 = np.log10(frame1)
        frame2 = img2*med2
        dum2 = np.log10(frame2)
        dum_ant = frame1 - model
        dum_act = frame2 - model
        
    if shift:
        ant = dum1 - np.roll(dum1, 3, axis=0)
        act = dum2 - np.roll(dum2, 3, axis=0)

    rawdif = dum_act - dum_ant
    
    # not doing debug
    scale, scl, split = 1, 1, 1
    x, y = 0., 0.
    regx, regy = 0, 0
    x1,y1 = 0, 0
    a2 = np.zeros([nx2, nx1]) # nrows, ncols
    shifts = np.zeros([nreg,nreg,2]) # assuming nreg = 1
    
    if shift:
        # ignoring for loops bc nreg = 1
        x2 = int(float(nx1) / nreg - 1)
        y2 = int(float(nx2) / nreg - 1)
        
        reg1 = ant[y1:y2+1, x1:x2+1] # switching indexing
        reg2 = act[y1:y2+1, x1:x2+1]        
        regx = x2 - x1 + 1
        regy = y2 - y1 + 1 
        
        # skipping scale 207-210
        # skipping box
        
        if side == 'a':
            if tel == 'hi_2':
                r1=230*2 # 512->1024 means x2 (sure)
                r2=450*2
                s1=130*2
                s2=400*2
            else:
                r1=0
                r2=500
                s1=700
                s2=1000
        else:
            if tel == 'hi_2':
                r1=139*2 # 512->1024 means x2 (sure)
                r2=311*2
                s1= 44*2
                s2=473*2
            else:
                r1=500
                r2=1000
                s1=700
                s2=1000
        
        # get cross correlation region
        if (nreg > 1) or (nx1 < 1024) or (nx2 < 1024):
            ccreg1 = reg1
            ccreg2 = reg2
        else:
            ccreg1 = reg1[s1:s2+1, r1:r2+1]
            ccreg2 = reg2[s1:s2+1, r1:r2+1]
        
        w1 = ~ np.isfinite(ccreg1)
        ccreg1[w1] = 0
        w2 = ~ np.isfinite(ccreg2)
        ccreg2[w2] = 0
        h, xmax, ymax = test_crosscorr(ccreg1,ccreg2,x,y, doGauss=True)
        
        x = xmax * scl/scale
        y = ymax * scl/scale
        
        reg1 = dum_ant[y1:y2+1, x1:x2+1]
        reg2 = dum_act[y1:y2+1, x1:x2+1]

        if scl != 1:
            sys.exit('Scl != 1 portion of rdif_hi not ported')
        
        # shift isnt exact but order=1 is closest match to idl
        temp = scipy.ndimage.shift(reg2, shift=(y, x), order=1, mode='constant', cval=0.0)
        d = temp - reg1
        
        if mfilt:
            d = scipy.ndimage.median_filter(d, size=7, mode='constant', cval=0.0)
            
        a2 = d # haven't done any regions so assume is fine? without indexing
        
        xout = x
        yout = y  
        shifts[0,0,0] = x
        shifts[0,0,1] = y
    
    # Might want to pass more things, not certain how much of the passed variables
    # in IDL are used later on...
    return a2, shifts
    
def test_crosscorr(arr1, arr2, xmax, ymax, doGauss=False):
    # Trying to keep var names same as idl and switching indexing as needed
    if arr1.shape != arr2.shape:
        sys.exit('Shape mismatch in test_crosscorr')
    
    array1 = arr1
    array2 = arr2
    
    sz = array1.shape
    xdim, ydim = sz[1], sz[0]    
    
    # skipping kmask portion 110 - 129
    # These lines are the ''equivalent'' of 134 in IDL
    # according to google. It is definitely not an exact match
    # (some normalization differences?) but they do seem to 
    # have the maximum in the same spot, which is all we need
    fft_array1 = np.fft.fft2(array1)
    fft_array2 = np.fft.fft2(array2)
    conj_fft_array2 = np.conj(fft_array2)
    multiplied_result = fft_array1 * conj_fft_array2
    result = np.real(np.fft.ifft2(multiplied_result))
    
    whereMax = np.where(result == np.max(result))
    xmax = whereMax[1][0] % xdim
    ymax = int(whereMax[1][0] / xdim) # IDL seems to round down so take int
    
    xshift = int(xdim / 2) - xmax
    yshift = int(ydim / 2) - ymax
    
    if doGauss:
        # on gauss2dfit
        res_shift = np.roll(result, (yshift, xshift), axis=(0, 1))
        xx = range(xdim)
        yy = range(ydim)
        x,y = np.meshgrid(xx,yy)
        
        # Since this is a fit it isn't 100% match to IDL but it is minor diff
        p0 = [np.max(res_shift), np.mean(x), np.mean(y), 10, 10, 0, np.min(res_shift)]
        popt, pcov = scipy.optimize.curve_fit(gaussian_2d, (x.ravel(), y.ravel()), res_shift.ravel(), p0=p0)
        amplitude_fit, x_mean_fit, y_mean_fit, x_stddev_fit, y_stddev_fit, theta_fit, offset_fit = popt
        xmax = x_mean_fit - xshift
        ymax = y_mean_fit - yshift
    else:
        sys.exit('Non Gauss part of test_crosscorr not ported yet')
    
    if (xmax > xdim/2): xmax = xmax - xdim
    if (ymax > ydim/2): ymax = ymax - ydim

    return result, xmax, ymax
    
def gaussian_2d(xy, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta, offset):
    """
    Returns a 2D Gaussian function (elliptical, with rotation).
    xy is a tuple (x, y) where x and y are 1D arrays or flattened 2D arrays.
    
    From the googles
    """
    x, y = xy
    # Rotate coordinates
    x_prime = np.cos(theta) * (x - x_mean) - np.sin(theta) * (y - y_mean)
    y_prime = np.sin(theta) * (x - x_mean) + np.cos(theta) * (y - y_mean)
    # Calculate the Gaussian
    g = offset + amplitude * np.exp(
        -(((x_prime) / x_stddev)**2 + ((y_prime) / y_stddev)**2) / 2
    )
    return g.ravel() # curve_fit expects a 1D array of data
    
files = ['/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_020921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_040921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_060921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_080921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_100921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_120921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_140921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_160921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_180921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_200921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_220921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120713_020921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120713_040921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120713_060921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120713_080921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120713_100921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120713_120921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120713_140921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120713_180921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120713_220921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120714_020921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120714_040921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120714_060921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120714_080921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120714_100921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120714_120921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120714_140921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120714_160921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120714_180921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120714_200921_s4h2A.fts']

lessfiles = ['/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_020921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_040921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_060921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_080921_s4h2A.fts','/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_100921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_120921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_140921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_160921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_180921_s4h2A.fts',	'/Users/kaycd1/wombat/pullFolder/STEREO/HI2A/HI2A_20120712_200921_s4h2A.fts']

profiles = ['../wbFits/STEREO/HI2A/wbpro_stahi2_20120712T020921.fits', '../wbFits/STEREO/HI2A/wbpro_stahi2_20120712T040921.fits',  '../wbFits/STEREO/HI2A/wbpro_stahi2_20120712T060921.fits',  '../wbFits/STEREO/HI2A/wbpro_stahi2_20120712T080921.fits',  '../wbFits/STEREO/HI2A/wbpro_stahi2_20120712T100921.fits', '../wbFits/STEREO/HI2A/wbpro_stahi2_20120712T120921.fits', '../wbFits/STEREO/HI2A/wbpro_stahi2_20120712T140921.fits', '../wbFits/STEREO/HI2A/wbpro_stahi2_20120712T160921.fits', '../wbFits/STEREO/HI2A/wbpro_stahi2_20120712T180921.fits', '../wbFits/STEREO/HI2A/wbpro_stahi2_20120712T200921.fits']

if __name__ == '__main__':
    srem(files, side='a')
    