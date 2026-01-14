"""
Module for functions related to various generic secchi 
processing. Called by secchi_prep and friends. Largely a port 
of the corresponding IDL routines and we have kept names (mostly) 
matching and indicated what portions have been left out to facilitate 
comparison to the other version. Kept at a near match to the
IDL routines on a pixel by pixel basis. We try to keep anything
generic or named scc_ here as opposed to the individual 
instrument python files

External Calls:
    sunspyce

"""
from astropy.io import fits
import numpy as np
import datetime
import platform
from astropy import wcs
import sys, os
from scipy.io import readsav
from sunspyce import get_sunspyce_roll, get_sunspyce_hpc_point

#|--------------------|
#|--- Date Globals ---|
#|--------------------|
global mjd_epoch, idl_base_date
mjd_epoch = datetime.datetime(1858, 11, 17, 0, 0, 0)
idl_base_date = datetime.datetime(1979, 1, 1, 0, 0, 0) # needed for anytim matching

#|----------------------------------|
#|--- Make empty secchi-like hdr ---|
#|----------------------------------|
def def_secchi_hdr():
    """
    Function that generates and empty secci style header

    Input:
       None
    
    Output:
        hdr: a dictionary set up with typical secchi style keywords
             set to default/null values
    
    """    
    hdr = {}
    hdr['EXTEND'] = 'F'
    hdr['BITPIX'] = 0
    hdr['NAXIS'] = 0
    hdr['NAXIS1'] = 0
    hdr['NAXIS2'] = 0
    hdr['DATE_OBS'] = ''
    hdr['TIME_OBS'] = ''
    hdr['FILEORIG'] = ''
    hdr['SEB_PROG'] = ''
    hdr['SYNC'] = ''
    hdr['SPWX'] = 'F'
    hdr['EXPCMD'] = -1
    hdr['EXPTIME'] = -1
    hdr['DSTART1'] = 0
    hdr['DSTOP1'] = 0
    hdr['DSTART2'] = 0
    hdr['DSTOP2'] = 0
    hdr['P1COL'] = 0
    hdr['P2COL'] = 0
    hdr['P1ROW'] = 0
    hdr['P2ROW'] = 0
    hdr['R1COL'] = 0
    hdr['R2COL'] = 0
    hdr['R1ROW'] = 0
    hdr['R2ROW'] = 0
    hdr['RECTIFY'] = 'F'
    hdr['RECTROTA'] = 0
    hdr['RECTROTA'] = ''
    hdr['LEDPULSE'] = 0
    hdr['OFFSET'] = 9999
    hdr['BIASMEAN'] = 0.
    hdr['BIASSDEV'] = -1.
    hdr['GAINCMD'] = -1
    hdr['GAINMODE'] = '' 
    hdr['SUMMED'] = 0.
    hdr['SUMROW'] = 1
    hdr['SUMCOL'] = 1
    hdr['CEB_T'] = 999
    hdr['TEMP_CCD'] = 9999.
    hdr['POLAR'] = -1
    hdr['ENCODERP'] = -1
    hdr['WAVELNTH'] = 0
    hdr['FILTER'] = ''
    hdr['ENCODERQ'] = -1
    hdr['FPS_ON'] = ''
    hdr['OBS_PROG'] = 'schedule'
    hdr['DOORSTAT'] = -1
    hdr['SHUTTDIR'] = '' 
    hdr['READ_TBL'] = -1
    hdr['CLR_TBL'] = -1
    hdr['READFILE'] = ''
    hdr['DATE_CLR'] = ''
    hdr['DATE_RO'] = ''
    hdr['READTIME'] = -1.
    hdr['CLEARTIM'] = 0.
    hdr['IP_TIME'] = -1
    hdr['COMPRSSN'] = 0
    hdr['COMPFACT'] = 0
    hdr['NMISSING'] = -1.
    hdr['MISSLIST'] = ''
    hdr['SETUPTBL'] = ''
    hdr['EXPOSTBL'] = ''
    hdr['MASK_TBL'] = ''
    hdr['IP_TBL'] = ''
    hdr['COMMENT'] = ''
    hdr['HISTORY'] = ''
    hdr['DIV2CORR'] = 'F'
    hdr['DISTCORR'] = 'F'
    hdr['TEMPAFT1'] = 9999.
    hdr['TEMPAFT2'] = 9999.
    hdr['TEMPMID1'] = 9999.
    hdr['TEMPMID2'] = 9999.
    hdr['TEMPFWD1'] = 9999.
    hdr['TEMPFWD2'] = 9999.
    hdr['TEMPTHRM'] = 9999.
    hdr['TEMP_CEB'] = 9999.
    hdr['ORIGIN'] = ''
    hdr['DETECTOR'] = ''
    hdr['IMGCTR'] = 0
    hdr['TIMGCTR'] = 0
    hdr['OBJECT'] = ''
    hdr['FILENAME'] = ''
    hdr['DATE'] = ''
    hdr['INSTRUME'] = 'SECCHI'
    hdr['OBSRVTRY'] = ''
    hdr['TELESCOP'] = 'STEREO'
    hdr['WAVEFILE'] = ''
    hdr['CCDSUM'] = 0.
    hdr['IPSUM'] = 0.
    hdr['DATE_CMD'] = '' 
    hdr['DATE_AVG'] = ''
    hdr['DATE_END'] = ''
    hdr['OBT_TIME'] = 0.
    hdr['APID'] = 0
    hdr['OBS_ID'] = 0
    hdr['OBSSETID'] = 0
    hdr['IP_PROG0'] = 0
    hdr['IP_PROG1'] = 0
    hdr['IP_PROG2'] = 0
    hdr['IP_PROG3'] = 0
    hdr['IP_PROG4'] = 0
    hdr['IP_PROG5'] = 0
    hdr['IP_PROG6'] = 0
    hdr['IP_PROG7'] = 0
    hdr['IP_PROG8'] = 0
    hdr['IP_PROG8'] = 0
    hdr['IP_00_19'] = ''
    hdr['OBSERVER'] = ''
    hdr['BUNIT'] = ''
    hdr['BLANK'] = 0
    hdr['FPS_CMD'] = ''
    hdr['VERSION'] = ''
    hdr['CEB_STAT'] = -1
    hdr['CAM_STAT'] = -1
    hdr['READPORT'] = ''
    hdr['CMDOFFSE'] = 0.
    hdr['RO_DELAY'] = -1.
    hdr['LINE_CLR'] = -1.
    hdr['RAVG'] = -999.
    hdr['BSCALE'] = 1.0
    hdr['BZERO'] = 0.
    hdr['SCSTATUS'] = -1
    hdr['SCANT_ON'] = ''
    hdr['SCFP_ON'] = ''
    hdr['CADENCE'] = 0
    hdr['CRITEVT'] = ''
    hdr['EVENT'] = 'F'
    hdr['EVCOUNT'] = ''
    hdr['EVROW'] = 0
    hdr['EVCOL'] = 0
    hdr['COSMICS'] = 0
    hdr['N_IMAGES'] = 0
    hdr['VCHANNEL'] = 0
    hdr['OFFSETCR'] = 0.
    hdr['DOWNLINK'] = ''
    hdr['DATAMIN'] = -1.0
    hdr['DATAZER'] = -1.0
    hdr['DATASAT'] = -1.0
    hdr['DSATVAL'] = -1.0
    hdr['DATAAVG'] = -1.0
    hdr['DATASIG'] = -1.0
    hdr['DATAP01'] = -1.0
    hdr['DATAP10'] = -1.0
    hdr['DATAP25'] = -1.0
    hdr['DATAP50'] = -1.0
    hdr['DATAP75'] = -1.0
    hdr['DATAP90'] = -1.0
    hdr['DATAP95'] = -1.0
    hdr['DATAP98'] = -1.0
    hdr['DATAP99'] = -1.0
    hdr['CALFAC'] = 0.
    hdr['CRPIX1'] = 0
    hdr['CRPIX2'] = 0
    hdr['CRPIX1A'] = 0
    hdr['CRPIX2A'] = 0
    hdr['RSUN'] =  0.
    hdr['CTYPE1'] = 'HPLN-TAN'
    hdr['CTYPE2'] = 'HPLN-TAN'
    hdr['CRVAL1'] = 0.
    hdr['CRVAL2'] = 0.
    hdr['CROTA'] = 0.
    hdr['PC1_1'] = 1.
    hdr['PC1_2'] = 0.
    hdr['PC2_1'] = 0.
    hdr['PC2_2'] = 1.
    hdr['CUNIT1'] = ''
    hdr['CUNIT2'] = ''
    hdr['CDELT1'] = 0.
    hdr['CDELT2'] = 0.
    hdr['PV2_1'] = 0.
    hdr['PV2_1A'] = 0.
    hdr['SC_ROLL'] = 9999.
    hdr['SC_PITCH'] = 9999.
    hdr['SC_YAW'] = 9999.
    hdr['SC_PITA'] = 9999.
    hdr['SC_YAWA'] = 9999.
    hdr['INS_R0'] = 0.
    hdr['INS_Y0'] = 0.
    hdr['INS_X0'] = 0.
    hdr['CTYPE1A'] = 'RA---TAN'
    hdr['CTYPE2A'] = 'RA---TAN'
    hdr['CUNIT1A'] = 'deg'
    hdr['CUNIT1A'] = 'deg'
    hdr['CRVAL1A'] = 0.
    hdr['CRVAL2A'] = 0.
    hdr['PC1_1A'] = 1.
    hdr['PC1_2A'] = 0.
    hdr['PC2_1A'] = 0.
    hdr['PC2_2A'] = 1.
    hdr['CDELT1A'] = 0.
    hdr['CDELT2A'] = 0.
    hdr['CRLN_OBS'] = 0.
    hdr['CRLT_OBS'] = 0.
    hdr['XCEN'] = 9999.
    hdr['YCEN'] = 9999.
    hdr['EPHEMFIL'] = ''
    hdr['ATT_FILE'] = ''
    hdr['DSUN_OBS'] = 0.
    hdr['HCIX_OBS'] = 0.
    hdr['HCIY_OBS'] = 0.
    hdr['HCIZ_OBS'] = 0.
    hdr['HAEX_OBS'] = 0.
    hdr['HAEY_OBS'] = 0.
    hdr['HAEZ_OBS'] = 0.
    hdr['HEEX_OBS'] = 0.
    hdr['HEEY_OBS'] = 0.
    hdr['HEEZ_OBS'] = 0.
    hdr['HEQX_OBS'] = 0.
    hdr['HEQY_OBS'] = 0.
    hdr['HEQZ_OBS'] = 0.
    hdr['LONPOLE'] = 180
    hdr['HGLN_OBS'] = 0.
    hdr['HGLT_OBS'] = 0.
    hdr['EAR_TIME'] = 0.
    hdr['SUN_TIME'] = 0.
    #Skipping the EUV only keywords for now 
    
    return hdr

#|---------------------------------|
#|--- Add missing keys to a hdr ---|
#|---------------------------------|
def fill_from_defhdr(hdr):
    """
    Function that adds any missing secchi-like keywords to an 
    existing header (using default/null values for missing)

    Input:
        hdr: the input header
    
    Output:
        hdr: the header with added keys/values

    """    
    mthdr   = def_secchi_hdr()
    allKeys = np.array(mthdr.keys())
    keys    = np.array(hdr.keys())
    for key in mthdr.keys():
        if key not in hdr.keys():
            hdr[key] = mthdr[key]
    return hdr

#|--------------------------------|
#|--- Get origin based on inst ---|
#|--------------------------------|
def sccrorigin(hdr):
    """
    Function that returns the origin for the data portion
    of a secchi instrument fits image

    Input:
        hdr: a header for a secchi image
    
    Output:
        coord:  an array of [column, row] corresponding to the pixels
                of the origin


    """    
    # Full port of sccrorigin
    # Could probably just make a dictionary
    p1col=51 # why are these in IDL? seem unused
    p1row=1 
    if hdr['rectify']:
        if hdr['OBSRVTRY'] == 'STEREO_A':
            det = hdr['detector']
            if det == 'EUVI':
                r1col, r1row = 129, 79
            elif det == 'COR1':
                r1col, r1row = 1, 79
            elif det == 'COR2':
                r1col, r1row = 129, 51
            elif det == 'HI1':
                r1col, r1row = 51, 1
            elif det == 'HI2':
                r1col, r1row = 51, 1
        elif hdr['OBSRVTRY'] == 'STEREO_B':
            det = hdr['detector']
            if det == 'EUVI':
                r1col, r1row = 1, 79
            elif det == 'COR1':
                r1col, r1row = 129, 51
            elif det == 'COR2':
                r1col, r1row = 1, 79
            elif det == 'HI1':
                r1col, r1row = 79, 129
            elif det == 'HI2':
                r1col, r1row = 79, 129
        else:
            # asuuming LASCO/EIT
            r1col, r1row = 20, 1
    else:
        r1col, r1row = 51, 1
    return [r1col, r1row]

#|-------------------------------------|
#|--- Get headers + initiate arrays ---|
#|-------------------------------------|
def scc_make_array(filesIn, outSize=None, trim_off=False):
    """
    Function that makes empty arrays to hold the fits image data. This 
    could probably be tossed bc python doesn't require pre-defined variables

    Input:
        filesIn: a list of fits files path+names
    
    Optional Input:
        outSize: size of the output image. Gets overwritten for all cases?
    
        trim_off: flag to turn off trimming the image down to only data (defaults to false)

    Output:
        imgs: an array of empty arrays sized to hold the image data
    
        hdrs: the hdrs from each file
    
        outout: the maximum dimension of outSize
        
        outSize: an array with the integer sizes of the dimensionts
    
        out: a dictionary of useful things
             {'outsize':outSize,'offset':offset,'readsize':readsize,'binned':summed}
    
        *** this seems like the outputs could be cleaned up but not a priority fix


    """    
    # Port of IDL code
    # Starting at line 50
    
    #|-----------------------------|
    #|--- Define out dictionary ---|
    #|-----------------------------|
    # Cannot find where output_array common is defined but seems 
    # to always have the following values    
    out =  {'outsize':[2048.00, 2048.00], 'offset':[129, 2176, 51, 2098], 'readsize':[2048, 2048], 'binned':1.00000}
    
    num = len(filesIn)
    
    # IDL used readfits to just get the header and passes -1 to void (55)
    # void = sccreadfits(filenames,mhdr,/nodata)
    mhdr = []
    for i in range(num):
        with fits.open(filesIn[i]) as hdulist:
               mhdr.append(hdulist[0].header)
    
    #|----------------------------|
    #|--- Untrimmed properties ---|
    #|----------------------------|
    if trim_off:
        # These seem to be the contents of the out common block...
        outsize  = [2176,2176]
        readsize = [2176,2176]
        offset   = [1,2176,1,2176]
        summed   = 1.
    
    else:
        #|-------------------------------|
        #|--- Calculate values w/trim ---|
        #|-------------------------------|
        
        # call sccrorigin
        # gets the 'rectified lower left (origin) value of full im area
        start = sccrorigin(mhdr[0])
        offset = [start[0],start[0]+2047,start[1],start[1]+2047]
            
        # Exclude over/underscan
        for i in range(num):
            r = mhdr[i]['R2COL']-mhdr[i]['R1COL']+mhdr[i]['R2ROW']-mhdr[i]['R1ROW']
            w = r < 2047*2
            
            # Getting max extent if there are sub_fields, doesn't seem to be triggered
            # so ignore for now but set flag for later (74-84)
            if w:
                print ('Have sub fields in ssc_make_array, need to add code')
                print (quit) # force a break
            
        readsize = np.array([offset[1]-offset[0]+1,offset[3]-offset[2]+1])
            
 
    #|--------------------------|
    #|--- Summing Adjustment ---|
    #|--------------------------|
    if outSize:
        summed = np.max(readsize) / outSize[0]
        outSize = readsize / summed
    else:
        summeds = [mhdr[i]['SUMMED'] for i in range(num)]
        if len(np.unique(summeds)) != 1:
            print ('Have different summed values in scc_make_array, need to add code')
            print (quit)
        summed = 2 ** (summeds[0] -1)
        outSize = readsize / summed
            
            
    # Skipping polarized for now (102 - 108)
    
    #|----------------------------|
    #|--- Reset out dictionary ---|
    #|----------------------------|
    out = {'outsize':outSize,'offset':offset,'readsize':readsize,'binned':summed}
    outout = np.max(outSize)
    
    #|-------------------------|
    #|--- Make empty arrays ---|
    #|-------------------------|
    imgs = np.empty([int(outSize[0]), int(outSize[1]), num], dtype=float)
    # makes an empty header
    headers = []
    for i in range(num):
        aHdr = def_secchi_hdr()
        headers.append(aHdr)
        
    return imgs,  headers, int(outout), outSize.astype(int), out
        
#|---------------------------------|
#|--- Fill array with fits data ---|
#|---------------------------------|
def scc_zelensky_array(im, hdr, outsize, out):
    """
    Function that puts data into the empty arrays. This is based on
    scc_putin_array in IDL but CK got cute/political. Probably 
    unnecessary anyway bc python can just open fits files directly
    to image variables

    Input:
        im: image data from a fits file
    
        hdr: header data from a fits file
    
        outsize: the desired size of the output image
    
        out: a dictionary of useful things
             {'outsize':outSize,'offset':offset,'readsize':readsize,'binned':summed}
    
    Output:
        output_img: the image, which is either the same or size changed according to outSize
    
        hdr: the updated header

    """        
    # Assume we have been given proper header and out is defined from make_array
    
    # Set up some things
    ccdfac = 1.
    ccdsizeh = 2048
    if hdr['INSTRUME'] != 'SECCHI':
        print ('Cannot guarantee scc_zelelnsky_array works for not secchi cases')
        print (Quit)
        
    # Skipping new/full keyword    

    # Skipping to 141

    #|------------------------------|
    #|--- Check if changing size ---|
    #|------------------------------|    
    output_img = im
    if (hdr['naxis1'] != out['outsize'][0]) or (hdr['naxis2'] != out['outsize'][1]):
        if (out['readsize'][1] - 1 != (hdr['r2col'] - hdr['r1col'])) or (out['readsize'][0] - 1 != (hdr['r2row'] - hdr['r1row'])):
            print ('hit uncoded part of scc_zelensky_array')
            print(Quit)
            
        #|--------------------------------|
        #|--- Resize and update header ---|
        #|--------------------------------|
        if out['binned'] != 2 ** (hdr['summed'] -1):
            outshape = np.array(output_img.shape).astype(float)
            bindif = np.max(outshape /np.max(out['outsize']))
            output_img = rebinIDL(im, out['outsize'].astype(int))
            
            hdr['summed'] = np.log(out['binned']) / np.log(2) + 1
            hdr['dstop1'] = hdr['dstop1'] / bindif
            hdr['dstop2'] = hdr['dstop2'] / bindif
            hdr['CRPIX1'] = 0.5+(hdr['crpix1']-0.5)/bindif
            hdr['CRPIX1A']= 0.5+(hdr['CRPIX1A']-0.5)/bindif
            hdr['CRPIX2'] = 0.5+(hdr['crpix2']-0.5)/bindif
            hdr['CRPIX2A']= 0.5+(hdr['CRPIX2A']-0.5)/bindif
            hdr['CDELT1'] =  hdr['CDELT1']*bindif
            hdr['CDELT2'] =  hdr['CDELT2']*(bindif)
            hdr['CDELT1A'] =  hdr['CDELT1A']*(bindif)
            hdr['CDELT2A'] =  hdr['CDELT2A']*(bindif)
            
            s = output_img.shape
            hdr['naxis1'] = s[0]
            hdr['naxis2'] = s[1]
            hdr['dstop1'] = np.min([hdr['dstop1'], s[0]]) # think this equiv of IDL
            hdr['dstop2'] = np.min([hdr['dstop2'], s[1]])
            
            # Don't think we use xcen/ycen so ignoring this
                                    
    return output_img, hdr
    
#|---------------------|
#|--- Rectify it!!! ---|
#|---------------------|
def secchi_rectify(a, scch, norotrate=False, silent=True):
    """
    Function that rectifies/rotates an image based on the spacecraft
    information. A little math and a lot of tedious header edits
    
    Input:
        a: an image
    
        scch: the corresponding header
    
        ** CK did not pick these var names, matches IDL...
    
    Optional Input:
        norotrate: flag to not rotate the image, only update rot keyword in hdr (defaults to False)
                *** should prob be norotate not norotrate but keep as is to avoid breaking anything
    
        silent: flag to not print non-critical info to screen (defaults to True)

    Output:
        b: a rotated image
    
        scch: the corresponding header

    """    
    #|----------------------------------|
    #|--- Check if already rectified ---|
    #|----------------------------------|    
    if scch['rectify'] not in ['F', False, 'False', '0', 0]:
        if not silent:
            print('We already done did the rectifying. Returning original img in secchi_rectify')
        return a, scch
    
    #|---------------------------------|
    #|--- Check if post conjunction ---|
    #|---------------------------------|
    post_conj = False 
    # Have to check if key actually exist bc stripped from some calibration files
    # Haven't tested a case that hits this
    if 'date_obs' in np.array(scch.keys):
        date_obs = scch['date_obs']
        try:
            doDT = datetime.strptime(date_obs, "%Y-%m-%dT%H:%M" )
            cut1 = datetime.datetime(2015,7,1,0,0,0)
            cut2 = datetime.datetime(2023,8,12,0,0,0)
            if (doDT > cut1) & (doDT < cut2):
                post_conj = True
        except:
            print('No date in header to use in rectify. Assuming not post conjunction')
            
    # Copy the header
    stch = scch
    
    #|--------------------------------|
    #|--- Do full rotation version ---|
    #|--------------------------------|    
    if ~norotrate:
        scch['rectify'] = True
        obs = scch['obsrvtry']
        
        #|---------------------------------|
        #|--- STA, not post-conjunction ---|
        #|---------------------------------|
        if (obs == 'STEREO_A') & ~post_conj:
            det = scch['detector']  
            
            #|------------|
            #|--- EUVI ---|
            #|------------|
            if det == 'EUVI':
                b = np.rot90(a[:,::-1], k=-1)
                stch['r1row']=2176-scch['p2col']+1
                stch['r2row']=2176-scch['p1col']+1
                stch['r1col']=2176-scch['p2row']+1
                stch['r2col']=2176-scch['p1row']+1
                stch['crpix1']=scch['naxis2']-scch['crpix2']+1
                stch['crpix2']=scch['naxis1']-scch['crpix1']+1
                stch['naxis1']=scch['naxis2']
                stch['naxis2']=scch['naxis1']
                stch['sumrow']=scch['sumcol']
                stch['sumcol']=scch['sumrow']
                stch['rectrota']=6
                rotcmt='transpose and rotate 180 deg CCW'
                #--indicate imaging area - rotate 6
                #  naxis1
                stch['dstart1']	=(129-stch['r1col']+1)>1
                stch['dstop1']	=stch['dstart1']-1 + ((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(79-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)
                

            #|------------|
            #|--- COR1 ---|
            #|------------|
            if det == 'COR1':
                b=np.rot90(a,k=1)  
                stch['r1row']=2176-scch['p2col']+1
                stch['r2row']=2176-scch['p1col']+1
                stch['r1col']=scch['p1row']
                stch['r2col']=scch['p2row']
                stch['crpix1']=scch['crpix2']
                stch['crpix2']=scch['naxis1']-scch['crpix1']+1
                stch['naxis1']=scch['naxis2']  
                stch['naxis2']=scch['naxis1']
                stch['sumrow']=scch['sumcol']
                stch['sumcol']=scch['sumrow']
                stch['rectrota']=3
                rotcmt='rotate 270 deg CCW'
                #--indicate imaging area - rotate 3
                #  naxis1
                stch['dstart1']	=1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(79-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)
            
            #|------------|
            #|--- COR2 ---|
            #|------------|
            if det == 'COR2':
                # IDL -> b = rotate(a,1) same as np.rot90(k=3)
                b = np.rot90(a,k=3)
                stch['r1row'] = scch['p1col']
                stch['r1row']=scch['p1col']
                stch['r2row']=scch['p2col']
                stch['r1col']=2176-scch['p2row']+1
                stch['r2col']=2176-scch['p1row']+1
                stch['crpix1']=scch['naxis2']-scch['crpix2']+1
                stch['crpix2']=scch['crpix1']
                stch['naxis1']=scch['naxis2'] 
                stch['naxis2']=scch['naxis1']
                stch['sumrow']=scch['sumcol']
                stch['sumcol']=scch['sumrow']
                stch['rectrota']=1
                rotcmt='rotate 90 deg CCW'
                #--indicate imaging area - rotate 1
                #  naxis1
                stch['dstart1']	=(129-stch['r1col']+1)>1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(51-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)
                
            #|-----------|
            #|--- HI1 ---|
            #|-----------|
            if det == 'HI1':
                b=a 	    	# no change
                stch['r1row']=scch['p1row']
                stch['r2row']=scch['p2row']
                stch['r1col']=scch['p1col']
                stch['r2col']=scch['p2col']
                #stch['naxis1']=scch['naxis1'] & stch['naxis2']=scch['naxis2']
                stch['rectrota']=0
                rotcmt='no rotation necessary'
            
            #|-----------|
            #|--- HI2 ---|
            #|-----------|
            if det == 'HI2':
                b=a 	    	# no change
                stch['r1row']=scch['p1row']
                stch['r2row']=scch['p2row']
                stch['r1col']=scch['p1col']
                stch['r2col']=scch['p2col']
                #stch['naxis1']=scch['naxis1'] & stch['naxis2']=scch['naxis2']
                stch['rectrota']=0
                rotcmt='no rotation necessary'
                            
        #|---------------------------|
        #|--- STB, whenever (RIP) ---|
        #|---------------------------|
        if (obs == 'STEREO_B'):
            det = scch['detector']  
            
            #|------------|
            #|--- EUVI ---|
            #|------------|
            if det == 'EUVI':
                b=np.rot90(a,k=1)
                stch['r1row']=2176-scch['p2col']+1
                stch['r2row']=2176-scch['p1col']+1
                stch['r1col']=scch['p1row']
                stch['r2col']=scch['p2row']
                stch['crpix1']=scch['crpix2']
                stch['crpix2']=scch['naxis1']-scch['crpix1']+1
                stch['naxis1']=scch['naxis2']
                stch['naxis2']=scch['naxis1']
                stch['sumrow']=scch['sumcol']
                stch['sumcol']=scch['sumrow']
                stch['rectrota']=3
                rotcmt='rotate 270 deg CCW'
                #--indicate imaging area - rotate 3
                #  naxis1
                stch['dstart1']	=1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(79-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)

            #|------------|
            #|--- COR1 ---|
            #|------------|
            if det == 'COR1':
                b=np.rot90(a,k=-1)      # 90 CCW =[x0,y0]=[y0,x1], [x1,y1]=[y1,x0] 
                stch['r1row']=scch['p1col']
                stch['r2row']=scch['p2col']
                stch['r1col']=2176-scch['p2row']+1
                stch['r2col']=2176-scch['p1row']+1
                stch['crpix1']=scch['naxis2']-scch['crpix2']+1
                stch['crpix2']=scch['crpix1']
                stch['naxis1']=scch['naxis2']
                stch['naxis2']=scch['naxis1']
                stch['sumrow']=scch['sumcol']
                stch['sumcol']=scch['sumrow']
                stch['rectrota']=1
                rotcmt='rotate 90 deg CCW'
                #--indicate imaging area - rotate 1
                #  naxis1
                stch['dstart1']	=(129-stch['r1col']+1)>1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(51-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)
            
            #|------------|
            #|--- COR2 ---|
            #|------------|
            if det == 'COR2':
                # IDL -> b = rotate(a,3) same as np.rot90(,k=1)
                b = np.rot90(a)
                stch['r1row']=2176-scch['p2col']+1
                stch['r2row']=2176-scch['p1col']+1
                stch['r1col']=scch['p1row']
                stch['r2col']=scch['p2row']
                stch['crpix1']=scch['crpix2']
                stch['crpix2']=scch['naxis1']-scch['crpix1']+1
                stch['naxis1']=scch['naxis2']
                stch['naxis2']=scch['naxis1']
                stch['sumrow']=scch['sumcol']
                stch['sumcol']=scch['sumrow']
                stch['rectrota']=3
                rotcmt='rotate 270 deg CCW'
                #--indicate imaging area - rotate 3
                #  naxis1
                stch['dstart1']	=1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(79-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)
            
            #|-----------|
            #|--- HI1 ---|
            #|-----------|
            if det == 'HI1':
                b= a[::-1,::-1]
                stch['r1row']=2176-scch['p2row']+1
                stch['r2row']=2176-scch['p1row']+1
                stch['r1col']=2176-scch['p2col']+1
                stch['r2col']=2176-scch['p1col']+1
                stch['crpix1']=scch['naxis1']-scch['crpix1']+1
                stch['crpix2']=scch['naxis2']-scch['crpix2']+1
                stch['naxis1']=scch['naxis1']
                stch['naxis2']=scch['naxis2']
                stch['rectrota']=2
                rotcmt='rotate 180 deg CCW'
                #--indicate imaging area - rotate 2
                #  naxis1
                stch['dstart1']	=(79-stch['r1col']+1)>1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(129-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)
            
            #|-----------|
            #|--- HI2 ---|
            #|-----------|
            if det == 'HI2':
                b= a[::-1,::-1]
                stch['r1row']=2176-scch['p2row']+1
                stch['r2row']=2176-scch['p1row']+1
                stch['r1col']=2176-scch['p2col']+1
                stch['r2col']=2176-scch['p1col']+1
                stch['crpix1']=scch['naxis1']-scch['crpix1']+1
                stch['crpix2']=scch['naxis2']-scch['crpix2']+1
                stch['naxis1']=scch['naxis1']
                stch['naxis2']=scch['naxis2']
                stch['rectrota']=2
                rotcmt='rotate 180 deg CCW'
                #--indicate imaging area - rotate 2
                #  naxis1
                stch['dstart1']	=(79-stch['r1col']+1)>1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(129-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)
            
                
        #|-----------------------------|
        #|--- STA, post-conjunction ---|
        #|-----------------------------|    
        if (obs == 'STEREO_A') & post_conj:
            det = scch['detector']
            
            #|------------|
            #|--- EUVI ---|
            #|------------|
            if det == 'EUVI':
                b=np.transpose(a)
                stch['r1row']=scch['p1col']
                stch['r2row']=scch['p2col']
                stch['r1col']=scch['p1row']
                stch['r2col']=scch['p2row']
                stch['crpix1']=scch['naxis1']-scch['crpix2']+1
                stch['crpix2']=scch['naxis2']-scch['crpix1']+1
                stch['naxis1']=scch['naxis2']
                stch['naxis2']=scch['naxis1']
                stch['sumrow']=scch['sumcol']
                stch['sumcol']=scch['sumrow']
                stch['rectrota']=4
                rotcmt='transpose'
                #--indicate imaging area - rotate 4
                #  naxis1
                stch['dstart1']	=(129-stch['r1col']+1)>1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(79-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)

            #|------------|
            #|--- COR1 ---|
            #|------------|
            if det == 'COR1':
                b=np.rot90(a, k=-1) #stch['r1row'] 90 CCW =[x0,y0]=[y0,x1], [x1,y1]=[y1,x0] 
                stch['r1row']=scch['p1col']
                stch['r2row']=scch['p2col']
                stch['r1col']=2176-scch['p2row']+1
                stch['r2col']=2176-scch['p1row']+1
                stch['crpix1']=scch['naxis2']-scch['crpix2']+1
                stch['crpix2']=scch['crpix1']
                stch['naxis1']=scch['naxis2'] 
                stch['naxis2']=scch['naxis1']
                stch['sumrow']=scch['sumcol']
                stch['sumcol']=scch['sumrow']
                stch['rectrota']=1
                rotcmt='rotate 90 deg CCW'
                #--indicate imaging area - rotate 1
                #  naxis1
                stch['dstart1']	=(129-stch['r1col']+1)>1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(51-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)
            
            #|------------|
            #|--- COR2 ---|
            #|------------|
            if det == 'COR2':
                # IDL -> b = rotate(a,3) same as np.rot90(,k=1)
                b = np.rot90(a)
                stch['r1row']=2176-scch['p2col']+1
                stch['r2row']=2176-scch['p1col']+1
                stch['r1col']=scch['p1row']
                stch['r2col']=scch['p2row']
                stch['crpix1']=scch['crpix2']
                stch['crpix2']=scch['naxis1']-scch['crpix1']+1
                stch['naxis1']=scch['naxis2'] 
                stch['naxis2']=scch['naxis1']
                stch['sumrow']=scch['sumcol']
                stch['sumcol']=scch['sumrow']
                stch['rectrota']=3
                rotcmt='rotate 270 deg CCW'
                #--indicate imaging area - rotate 3
                #  naxis1
                stch['dstart1']	=1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(79-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)

            #|-----------|
            #|--- HI1 ---|
            #|-----------|
            if det == 'HI1':
                b=a[::-1,::-1]
                stch['r1row']=2176-scch['p2row']+1
                stch['r2row']=2176-scch['p1row']+1
                stch['r1col']=2176-scch['p2col']+1
                stch['r2col']=2176-scch['p1col']+1
                stch['crpix1']=scch['naxis1']-scch['crpix1']+1
                stch['crpix2']=scch['naxis2']-scch['crpix2']+1
                stch['naxis1']=scch['naxis1']
                stch['naxis2']=scch['naxis2']
                stch['rectrota']=2
                rotcmt='rotate 180 deg CCW'
                #--indicate imaging area - rotate 2
                #  naxis1
                stch['dstart1']	=(79-stch['r1col']+1)>1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(129-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)
            
            #|-----------|
            #|--- HI2 ---|
            #|-----------|
            if det == 'HI2':
                b=a[::-1,::-1]
                stch['r1row']=2176-scch['p2row']+1
                stch['r2row']=2176-scch['p1row']+1
                stch['r1col']=2176-scch['p2col']+1
                stch['r2col']=2176-scch['p1col']+1
                stch['crpix1']=scch['naxis1']-scch['crpix1']+1
                stch['crpix2']=scch['naxis2']-scch['crpix2']+1
                stch['naxis1']=scch['naxis1']
                stch['naxis2']=scch['naxis2']
                stch['rectrota']=2
                rotcmt='rotate 180 deg CCW'
                #--indicate imaging area - rotate 2
                #  naxis1
                stch['dstart1']	=(79-stch['r1col']+1)>1
                stch['dstop1']	=stch['dstart1']-1+((stch['r2col']-stch['r1col']+1)<2048)
                #  naxis2
                stch['dstart2']	=(129-stch['r1row']+1)>1
                stch['dstop2']	=stch['dstart2']-1+((stch['r2row']-stch['r1row']+1)<2048)
            
                
        
    #|-------------------------------|
    #|--- Do keyword only version ---|
    #|-------------------------------|    
    else:
        stch.rectify = 'F'
        b = a 	    	# no change
        stch['r1row']=scch['p1row']
        stch['r2row']=scch['p2row']
        stch['r1col']=scch['p1col']
        stch['r2col']=scch['p2col']
        #stch['naxis1']=scch['naxis1'] & stch['naxis2']=scch['naxis2']
        stch['rectrota']=0
        rotcmt='no rotation'
        
    if stch['r1col'] < 1:
        stch['r2col'] = stch['r2col']+np.abs(stch['r1col'])+1
        stch['r1col'] = 1
    if stch['r1row'] < 1:
        stch['r2row'] = stch['r2row']+np.abs(stch['r1row'])+1
        stch['r1row'] = 1            
    
    # Don't have to explicitly add to header in python (skipping 485-509)
    
    # Reset the hdr to new values
    scch = stch
        
    return b, scch
    
#|-----------------------------|
#|--- Get pix of sun center ---|
#|-----------------------------|
def scc_sun_center(hdr):
    """
    Function that gets the pixel location of the sun center. Mostly
    a wrapper of the astropy wcs function, which is slower than the 
    routine in wcs_funs, but slightly more accurate

    Input:
        hdr: a header for the time/instrument
    
    Optional Input:
        downSize: size of the output image (in pixels)
                  *** assuming a square output

    Output:
        scen: center of the sun in pixels [x,y]


    """    
    # assuming proper header, doing single not array of headers
    
    sunc = {'xcen':0, 'ycen':0.}
    my_wcs = wcs.WCS(hdr)
    
    # assuming scale is 1 bc not setting keywords right now
    scale = 1
    
    scen = wcs_get_pixel(my_wcs, [0,0])
    
    return scen
    
#|---------------------------|
#|--- Mimic IDL rebinning ---|
#|---------------------------|
# Not 100% exact same but as close as possible
def rebinIDL(arr, new_shape):
    """
    Function that rebin things an array in the most similar way
    possible to the standard IDL rebinning. This is not 100 percent
    exact match but it is pretty close

    Input:
        arr: an input array
    
    Output:
        new_shape: the desired shape of the output

    """    
    factors = arr.shape // new_shape
    outarr = arr.reshape(new_shape[0], factors[0], new_shape[1], factors[1]).mean(3).mean(1)
    return outarr
    
#|------------------------|
#|--- Adjust for sebip ---|
#|------------------------|
def scc_sebip(img, hdr):
    """
    Function that performs a SEB IP correction

    Input:
        img: an image
    
        hdr: its header
    
    Output:
        im: the corrected image
        
        hdr: the update header
    
        flag: a flag indicating the correction was applied (?)

    """    
    # Assuming everything is ok as usual
    im = img
    flag = 0

    #|------------------------------|
    #|--- Pull/process IP values ---|
    #|------------------------------|
    ip = hdr['ip_00_19']
    # Make sure IP is 60 char long, could be as low as 58
    # (just porting these very non python lines for now)
    if len(ip) < 60:
        ip = ' ' + ip
    if len(ip) < 60:
        ip = ' ' + ip
    # This is a string of 20 up to 3 digit numbers. Most are 2 digits but it gets squished
    # together during the rare 3 digit ones so have to separate. Copied IDL method (ish) but  
    # could probably simplify   
    ipEnc = ip.encode(encoding='utf-8')
    byteIt = np.array([ipEnc[i] for i in range(60)]) 
    seb_ip = [chr(byteIt[i*3])+chr(byteIt[i*3+1])+chr(byteIt[i*3+2]) for i in range(20)]

    # Trim SW images
    if '117' in seb_ip:
    #x = np.where(seb_ip == '117')[0]
    #if len(x) != 0:
        print ('Need to port this when hit proper test case (in scc_sebip)')
        print (Quit)

    # Don't need the Vin Diesel chunk (108 - 121) from IDL, just check the cases 
    # for corrections on the fly below. (its a bunch of xxx = lines)
    flag = False

    #|-----------------------------------------------|
    #|--- Check for each tag and adjust as needed ---|
    #|-----------------------------------------------|
    seb_ip = np.array(seb_ip)
    # |--- Case of 1 ---|
    if '  1' in seb_ip:
        count = len(np.where(seb_ip == '  1')[0])
        if hdr['DIV2CORR']: 
            count = count  - 1
        im = im * (2 ** count)
        hdr['history'] = 'seb_ip Corrected for Divide by 2 x '+str(count)
        flag = True
        
    # |--- Case of 2 ---|
    if '  2' in seb_ip:
        count = len(np.where(seb_ip == '  2')[0])
        im = im**(2**count)
        hdr['history'] = 'seb_ip Corrected for Square Root x '+str(count)
        flag = True
        
    # |--- Case of 16 or 17 ---|
    if (' 16' in seb_ip) or (' 17' in seb_ip):
        count = len(np.where(seb_ip == ' 16')[0]) + len(np.where(seb_ip == ' 17')[0])
        im = im * (64**count)
        hdr['history'] = 'seb_ip Corrected for HI?SPW Divide by 64 x '+str(count)
        flag = True

    # |--- Case of 50 ---|
    if ' 50' in seb_ip:
        count = len(np.where(seb_ip == ' 50')[0])
        im = im * (4**count)
        hdr['history'] = 'seb_ip Corrected for for Divide by 4 x '+str(count)
        flag = True
        
    # |--- Case of 53 ---|
    if ' 53' in seb_ip:
        count = len(np.where(seb_ip == ' 53')[0])
        im = im * (4**count)
        hdr['history'] = 'seb_ip Corrected for for Divide by 4 x '+str(count)
        flag = True
        
    # |--- Case of 82 ---|
    if ' 82' in seb_ip:
        count = len(np.where(seb_ip == ' 82')[0])
        im = im * (2**count)
        hdr['history'] = 'seb_ip Corrected for Divide by 2 x '+str(count)
        flag = True

    # |--- Case of 83 ---|
    if ' 83' in seb_ip:
        count = len(np.where(seb_ip == ' 83')[0])
        im = im * (4**count)
        hdr['history'] = 'seb_ip Corrected for for Divide by 4 x '+str(count)
        flag = True

    # |--- Case of 84 ---|
    if ' 84' in seb_ip:
        count = len(np.where(seb_ip == ' 84')[0])
        im = im * (8**count)
        hdr['history'] = 'seb_ip Corrected for for Divide by 8 x '+str(count)
        flag = True

    # |--- Case of 85 ---|
    if ' 85' in seb_ip:
        count = len(np.where(seb_ip == ' 85')[0])
        im = im * (16**count)
        hdr['history'] = 'seb_ip Corrected for for Divide by 16 x '+str(count)
        flag = True

    # |--- Case of 86 ---|
    if ' 86' in seb_ip:
        count = len(np.where(seb_ip == ' 86')[0])
        im = im * (32**count)
        hdr['history'] = 'seb_ip Corrected for for Divide by 32 x '+str(count)
        flag = True

    # |--- (Crosby) Case of 87 (booooo) ---|
    if ' 87' in seb_ip:
        count = len(np.where(seb_ip == ' 87')[0])
        im = im * (64**count)
        hdr['history'] = 'seb_ip Corrected for for Divide by 64 x '+str(count)
        flag = True

    # |--- (Pastrnak) Case of 88 (yay) ---|
    if ' 88' in seb_ip:
        count = len(np.where(seb_ip == ' 88')[0])
        im = im * (128**count)
        hdr['history'] = 'seb_ip Corrected for for Divide by 128 x '+str(count)
        flag = True

    # |--- Case of 118 ---|
    if '118' in seb_ip:
        count = len(np.where(seb_ip == '118')[0])
        im = im * (3**count)
        hdr['history'] = 'seb_ip Corrected for for Divide by 3 x '+str(count)
        flag = True
    return im.astype(int), hdr, flag

#|----------------------------|
#|--- Get background image ---|
#|----------------------------|
def scc_getbkgimg(hdr, secchi_bkg ='STEREObackgrounds/', doRot=False):
    """
    Function that grabs the appropriate background file for a given header.
    It matches the image size of the input but does not currently correct
    for any (minor) difference in the between their rolls.

    Input:
        hdr: a list of fits files path+names
    
    Optional Input:
        secchi_bkg: folder where the background files are stored
                    (defaults to STEREObackgrounds/)
    
        doRot: rotate to correct for a roll diff between input and 
               background file (not implemented, defaults false)
    
    Output:
        bim: the processed background image
    
        bhdr: the corresponding header


    """    
    # Assume no interp for now
    
    # port of get_delim, untested on windows
    myos = platform.system()
    if myos == 'Windows':
        delim = '\\'
    else:
        delim = '/'
        
    # Assume proper header
    
    #|--------------------|
    #|--- Get detector ---|
    #|--------------------|
    tel = hdr['DETECTOR'].upper().strip()
    if tel == 'EUVI':
        cam = 'eu'
    else:
        cam = tel[0].lower()+tel[-1]
    isHI = (cam == 'h1') or (cam == 'h2')
    
    doshift = False # is this still used? commented out below
    
    #|----------------------|
    #|--- Get spacecraft ---|
    #|----------------------|
    # Get sc name
    tags = np.array([key for key in hdr])
    filename = hdr['FILENAME']
    if'OBSRVTRY' not in tags:
        #MVI header
        sc = filename[-4].lower()
        print ('Not certain on pulling from MVI check below is expected for sc')
        print (sc)
    else:
        sc = hdr['OBSRVTRY'][7].lower()

    # Check scname
    if sc not in ['a', 'b']:
        sys.exit('Error in scc_getbkgim. Invalid spacecraft name.')
        
    #|------------------------|
    #|--- Check processing ---|
    #|------------------------|
    # Check processing level
    isL1 = False
    levchar = filename[16]
    if (levchar ==1) or ((levchar == 0) & isHI):
        isL1 = True
        match = False

    #|----------------------|
    #|--- Build filename ---|
    #|----------------------|    
    # Not setting up keywords raw_calroll, calroll, daily so skipping (401-413)
    ndays = 30
    rootdir = secchi_bkg + sc + delim + 'monthly_min' + delim
    fchar = 'm'

    # Form the polar search string
    polstr = '_pTBr_'
    if 'polar' in hdr:
        polar = int(hdr['polar'])
        if (polar <  361) & (polar >= 0):
            polstr = '_p'+str(polar).zfill(3)+'_'

    # not using the totalb, double_totalB keys for now so skipping (423-427)
    
    #|-----------------------|
    #|--- Format the date ---|
    #|-----------------------|    
    # Get the date
    if 'DATE-AVG' in tags:
        dtin = hdr['DATE-AVG']
        daTag = 'DATE-AVG'
    elif 'DATE_AVG' in tags:
        dtin = hdr['DATE_AVG']
        daTag = 'DATE_AVG'
    cal = dtin[:4]+dtin[5:7]+dtin[8:10] # should pull out YYYYMMDD for reasonable strings
    sdir = cal[0:6]
    sfil = cal[2:8]    
    # assume we can do what we need with DT obj
    avgDT = datetime.datetime.strptime(dtin, "%Y-%m-%dT%H:%M:%S.%f" )
    mjdin = (avgDT - mjd_epoch).total_seconds()/(24*3600)
    
    #|-------------------------|
    #|--- Check repointings ---|
    #|-------------------------|    
    # Check against major sc repointings
    if sc == 'a':
        repoint = ['2006-12-21T13:15', '2007-02-03T13:15','2015-05-19T00:00', '2023-08-12T00:00']
        if tel == 'COR1':
            repoint.extend(['2010-01-27T16:49', '2010-11-19T16:00', '2011-01-12T12:23', '2011-02-11T04:23', '2011-03-08T17:00', '2011-12-05T12:03', '2012-02-19T02:33', '2012-03-16T00:00', '2014-07-08T00:00', '2014-08-23T17:00', '2015-11-16T00:00', '2016-03-15T00:00', '2016-04-13T01:08', '2016-05-25T05:40', '2016-06-01T12:00', '2016-06-08T00:00', '2016-09-02T16:19', '2018-09-27T16:44', '2019-02-05T21:07', '2019-02-17T05:57', '2020-01-14T23:52', '2021-05-07T23:53', '2023-05-15T23:00'])
        else:
            repoint.extend(['2014-08-19T00:00'])
        nomrollmjd = 54160
    if sc == 'b':
        repoint = ['2007-02-03T18:20', '2007-02-21T20:00']
        if tel == 'COR1':
            repoint.extend(['2009-01-30T16:20', '2010-03-24T01:17', '2011-03-11T00:00'])
        if tel == 'COR2':
            repoint.extend(['2010-02-23T08:12', '2011-01-27T03:47','2011-04-25T18:30'])
        nomrollmjd = 54313
    repoint = np.array(sorted(repoint))        
    # add in check for no exist
    i1 = np.where(repoint < dtin)[0]
    i2 = np.where(repoint > dtin)[0]
    if len(i1) > 0:
        minDT = datetime.datetime.strptime(repoint[np.max(i1)], "%Y-%m-%dT%H:%M")
        mjdmin = (minDT - mjd_epoch).total_seconds()/(24*3600)
    else:
        mjdmin = 0
    if len(i2) > 0:        
        maxDT = datetime.datetime.strptime(repoint[np.min(i2)], "%Y-%m-%dT%H:%M")
        mjdmax = (maxDT - mjd_epoch).total_seconds()/(24*3600)
    else:
        mjdmax = 99999

    pntroll = hdr['sc_roll']
    if (mjdin > 57160) & (mjdin < 60168):
        pntroll = hdr['crota']
    if pntroll < 0:
        titleangle = 360 + pntroll
    else:
        titleangle = pntroll

    if (np.abs(pntroll) < 10) or (tel == 'COR1') or (mjdin < nomrollmjd):
            postd = ''
    else:
        print ("Reached uncoded part, need to test with example")
        print (QUit)
    
    if isHI:
        roll = 0
    
    #|---------------------------|
    #|--- Find closest friend ---|
    #|---------------------------|    
    # Look for the correct or closest file. IDL makes this a headache
    # This is equiv to 531-662 but ignoring interp and other stuff
    exactFile = rootdir + sdir + delim + fchar + cam + sc.upper() + polstr + sfil + postd + '.fts'
    #print (exactFile)
    if os.path.exists(exactFile):
        bkgFile = exactFile
    else:
        # loop through n days before and after to find closest friend
        dday = 0
        while dday <= ndays:
            dday += 1
            plusDT = avgDT + datetime.timedelta(days=dday)
            pstr = plusDT.strftime('%Y%m%d')  
            psdir, psfil =  pstr[:-2], pstr[2:]
            pPath = rootdir + psdir + delim + fchar + cam + sc.upper() + polstr + psfil + postd + '.fts'
            hasP =  os.path.exists(pPath)
            minusDT = avgDT + datetime.timedelta(days=-dday)
            mstr = minusDT.strftime('%Y%m%d')  
            msdir, msfil =  mstr[:-2], mstr[2:]
            mPath = rootdir + msdir + delim + fchar + cam + sc.upper() + polstr + msfil + postd + '.fts'
            hasM = os.path.exists(mPath)
            if hasP and hasM:
                dday = 9999
                delP = (plusDT - avgDT).total_seconds()
                delM = (avgDT - minusDT).total_seconds
                if delM <= delP:
                    bkgFile = mPath
                else:
                    bkgFile = pPath
            elif hasP:
                dday = 9999
                bkgFile = pPath
            elif hasM:
                dday = 9999
                bkgFile = mPath
                
            else:
                if dday > ndays:
                    sys.exit('Cannot find appropriate background file')
                    
    # Skipping more for interp, postd, doubles 
    
    #|---------------------|
    #|--- Open bkg file ---|
    #|---------------------|    
    # Read in file, skipping 666-711
    if isL1 and not isHI:
        print ('Need to add secchi prep the background, not coded')
        print (Quit) 
    else:
        with fits.open(bkgFile) as hdulist:
            bim  = hdulist[0].data
            bhdr = hdulist[0].header
    # Skipping median keyword
    btags = np.array([key for key in bhdr])
    date_avg = ''
    if 'DATE-AVG' in btags:
        date_avg = bhdr['DATE-AVG']
    if 'date-avg' in btags:
        date_avg = bhdr['date-avg']
    elif 'DATE_AVG' in btags:
        date_avg = bhdr['DATE_AVG']
    elif 'date_avg' in btags:
        date_avg = bhdr['date_avg']
    if date_avg == '':
        if 'DATE_OBS' in btags:
            date_avg = bhdr['DATE_OBS']
        elif 'date_obs' in btags:
            date_avg = bhdr['date_obs']
        elif 'DATE-OBS' in btags:
            date_avg = bhdr['DATE-OBS']
        elif 'date-obs' in btags:
            date_avg = bhdr['date-obs']
        else:
            sys.exit('Issue getting background image date')
            
    #|------------------|
    #|--- Match size ---|
    #|------------------|    
    i_reduce = bhdr['naxis1'] / hdr['naxis1']
    j_reduce = bhdr['naxis2'] / hdr['naxis2']
        
    if (i_reduce != 1) or (j_reduce != 1):
        bim = rebinIDL(bIm, [hdr['naxis1'], hdr['naxis2']])
  
    # Copy header info for some reason (prob rebin?)
    if not isL1:
        bhdr['dstop1'] = hdr['dstop1']
        bhdr['dstop2'] = hdr['dstop2']
        bhdr['naxis1'] = hdr['naxis1']
        bhdr['naxis2'] = hdr['naxis2']
        bhdr['summed'] = hdr['summed']
        bhdr['crpix1'] = hdr['crpix1']
        bhdr['crpix2'] = hdr['crpix2']
        bhdr['crpix1a'] = hdr['crpix1a']
        bhdr['crpix2a'] = hdr['crpix2a']
        bhdr['CDELT1'] = hdr['CDELT1']
        bhdr['CDELT2'] = hdr['CDELT2']
        bhdr['CDELT1A'] = hdr['CDELT1A']
        bhdr['CDELT2A'] = hdr['CDELT2A']
        
    # Shift things for HI, ignore for now 760-786
    maxshift = 0
    if doshift:
        print ('havent coded HI shifting in scc_getbkgimg')
        print (Quit)
        
    # No interp skipping 791 - 854
    
    # skipping another shift thing 856-859
    
    # skipping match check 864
    
    # skipping header correction
    
    # Check for rotate correction
    rolldif = hdr['crota'] - bhdr['crota']
    if (np.abs(rolldif) > 1) and doRoll:
        print ('Havent coded background roll')
        print (Quit)
        
    # skipping double exposure correction
    
    # skipping match corrction
    
    # skipping HI L1 correction
    
    return bim, bhdr
                        
#|-------------------------------|
#|--- Correct diffusion in HI ---|
#|-------------------------------|
def scc_hi_diffuse(hdr, ipsum=None):   
    """
    Function that gets a correction factor for diffusion
    in HI images. The returned factor is the size of the
    image as defined in the header

    Input:
        hdr: an HI header file
    
    Optional Input:
        ipsum: an ipsum value. it will be pulled from the hdr if
               not provided (defaults to None)

    Output:
        correct: a 2d array correction factor


    """    
    dtor = np.pi / 180.            
    if ipsum == None:
        ipsum = hdr['ipsum']
    
    #|--- Calculate summing ---|    
    summing = 2 ** (ipsum - 1)
    
    #|--- Case with ravg in hdr ---|
    cdelt = None
    if 'ravg' in hdr:
        if hdr['ravg'] > 0:
            mu = hdr['pv2_1']
            cdelt = hdr['cdelt1'] * dtor
    
    #|--- Other cases ---|
    if cdelt == None:
        if hdr['detector'] == 'HI1':
            if hdr['OBSRVTRY'] == 'STEREO_A':
                mu = 0.102422
                cdelt = 35.96382 / 3600 * dtor * summing
            elif hdr['OBSRVTRY'] == 'STEREO_B':
                mu = 0.095092
                cdelt = 35.89977 / 3600 * dtor * summing
        if hdr['detector'] == 'HI2':
            if hdr['OBSRVTRY'] == 'STEREO_A':
                mu = 0.785486
                cdelt = 130.03175 / 3600 * dtor * summing
            elif hdr['OBSRVTRY'] == 'STEREO_B':
                mu = 0.68886
                cdelt = 129.80319 / 3600 * dtor * summing
                
    # Compute pixel size in mm and paraxial focal length
    pixelSize = 0.0135 * summing
    fp = pixelSize / cdelt
    
    #|-----------------------------|
    #|--- Make Correction Array ---|
    #|-----------------------------|
    # Compute linear distance from center of ccd
    x = np.arange(hdr['naxis1']) - hdr['crpix1'] + hdr['dstart1']
    y = np.arange(hdr['naxis2']) - hdr['crpix2'] + hdr['dstart2']
    xx = np.zeros([hdr['naxis1'],hdr['naxis1']])
    yy = np.zeros([hdr['naxis1'],hdr['naxis1']])    
    for i in range(hdr['naxis1']):
        xx[i,:] = x
        yy[:,i] = y
    r = np.sqrt(xx**2 +yy**2) * pixelSize
    
    # get solid angle
    gamma = fp * (mu + 1.0) / r
    cosalpha1 = (-1.0 * mu + gamma * np.sqrt(1.0-mu*mu+gamma*gamma))/(1.0+gamma*gamma)
    
    correct = ((mu+1.0)**2 *(mu*cosalpha1+1.0)) / ((mu+cosalpha1)**3)
   
    return correct
    
#|---------------------|
#|--- Trim an image ---|
#|---------------------|
def scc_img_trim(im, hdr, gtFile=None):
    """
    Function that trims an image based on the dstart/dstop
    values from the header.

    Input:
        im: a array with SECCHI image data
    
        hdr: the corresponding header 
    
    Optional Input:
        gtFile: a prep file for precommcorrect to use

    Output:
        img: the trimmed image data
        
        hdr: the corresponding header
    
    Notes:
        Only a partial port for the piece used by wombat

    """    
    #|----------------------------|
    #|--- Check if unprocessed ---|
    #|----------------------------|
    if (hdr['DSTOP1'] < 1) or (hdr['DSTOP1'] > hdr['NAXIS1']) or  (hdr['DSTOP2'] > hdr['NAXIS2']):
        im, hdr = precommcorrect(im, hdr, gtFile=gtFile )
    
    
    #|---------------|
    #|--- Trim it ---|
    #|---------------|
    x1 = int(hdr['DSTART1']-1)
    x2 = int(hdr['DSTOP1']-1)
    y1 = int(hdr['DSTART2']-1)
    y2 = int(hdr['DSTOP2']-1)
    img = im[y1:y2+1,x1:x2+1]
    
    #|-------------------------|
    #|--- Flag uncoded part ---|
    #|-------------------------|
    s = img.shape
    if (hdr['naxis1'] != s[0]) or (hdr['naxis2'] != s[1]):
        sys.exit('Havent implemented section of scc img trim but doable')
        
    return img, hdr
            
#|------------------------------------|
#|--- Pre commissioning correction ---|
#|------------------------------------|
def precommcorrect(im, hdr, gtFile=None):
    """
    Function that applies the pre commissioning correction

    Input:
        im: a array with SECCHI image data
    
        hdr: the corresponding header 
    
    Optional Input:
        gtFile: a prep file used by euvi_point (defaults to None)

    Output:
        im: the processed image data
        
        hdr: the corresponding header
    
    Notes:
        This has been ported for wombat using EUVI images. The
        other instruments have not been full ported.

    """    
    #|---------------------------|
    #|--- ICERDIV2 Correction ---|
    #|---------------------------|
    if (hdr['comprssn'] > 89) & (hdr['comprssn'] < 102):
        if hdr['DIV2CORR'] == False:
            hdr, im = scc_icerdiv2(hdr,im)
        else:
            biasmean = hdr['biasmean']
            p01mbias = hdr['datap01'] - biasmean
            if (p01mbias > 0.8 * biasmean):
        	    im = im / 2
        	    hdr['datap01'] = hdr['datap01'] / 2
        	    hdr['datamin'] = hdr['datamin'] / 2
        	    hdr['datamax'] = hdr['datamax'] / 2
        	    hdr['dataavg'] = hdr['dataavg'] / 2
        	    hdr['datap10'] = hdr['datap10'] / 2
        	    hdr['datap25'] = hdr['datap25'] / 2
        	    hdr['datap75'] = hdr['datap75'] / 2
        	    hdr['datap90'] = hdr['datap90'] / 2
        	    hdr['datap95'] = hdr['datap95'] / 2
        	    hdr['datap98'] = hdr['datap98'] / 2
        	    hdr['datap99'] = hdr['datap99'] / 2
        	    hdr['div2corr'] = False 
                
    hdr['mask_tbl'] = 'NONE'
    
    #|------------------------------------|
    #|--- EUVI Image Center Correction ---|
    #|------------------------------------|
    if hdr['DETECTOR'] == 'EUVI':
        hdr = euvi_point(hdr, gtFile)
    else:
        sys.exit('COR point corrections not ported in precommcorrect')
    
    #|-------------------------------|
    #|--- DSTART/STOP Corrections ---|
    #|-------------------------------|
    #--Code taken from revision 1.19 of scc_img_trim.pro  
    #--Calculate data area from un-rectified image cooridinates
    if (hdr['DSTOP1'] < 1) or (hdr['DSTOP1'] > hdr['NAXIS1']) or (hdr['DSTOP2'] > hdr['NAXIS2']):
        x1 = 51 - hdr['P1COL']
        if x1 < 0: x1 = 0
        x2 = hdr['P2COL'] - hdr['P1COL']
        if x2 > 2048 + x1 - 1: x2 = 2048 + x1 - 1

        y1 = 1 - hdr['P1ROW']
        if y1 < 0: y1 = 0
        y2 = hdr['P2ROW'] - hdr['P1ROW']
        if y2 > 2048 + y1 - 1: y2 = 2048 + y1 - 1
        
        #--Reset P1(2)COL and P1(2)ROW to trimmed values
        if hdr['P1COL'] < 51: hdr['P1COL'] = 51
        hdr['P2COL'] = hdr['P1COL'] + (x2-x1)
        if hdr['P1ROW'] < 1 : hdr['P1ROW'] = 1
        hdr['P2ROW'] = hdr['P1ROW'] + (y2-y1)

        #--Correct data area cooridinates for summing
        x1 = int(x1 / 2**(hdr['summed']-1))
        x2 = ((hdr['P2COL'] - hdr['P1COL'] + 1)/ 2**(hdr['summed'] - 1)) + x1 - 1 
        y1 = int(y1 / 2**(hdr['summed']-1))
        y2 =((hdr['P2ROW'] - hdr['P1ROW'] + 1) / 2**(hdr['summed'] - 1)) + y1 - 1
        
        #|---------------------------------|
        #|--- Rectified Img Corrections ---|
        #|---------------------------------|
        if hdr['RECTIFY'] in ['T', True, 'True', 1, '1']:
            if (hdr['OBSRVTRY'] == 'STEREO_A'):
                if hdr['DETECTOR'] ==  'EUVI':
                    rx1 = hdr['naxis1'] - y2 - 1
                    rx2 = hdr['naxis1'] - y1 - 1
                    ry1 = hdr['naxis2'] - x2 - 1
                    ry2 = hdr['naxis2'] - x1 - 1
                    hdr['R1COL'] = 2176 - hdr['P2ROW'] + 1
                    hdr['R2COL'] = 2176 - hdr['P1ROW'] + 1
                    hdr['R1ROW'] = 2176 - hdr['P2COL'] + 1
                    hdr['R2ROW'] = 2176 - hdr['P1COL'] + 1
                else:
                    sys.exit('Havent ported other detectors in precommcorrect')
        
            if (hdr['OBSRVTRY'] == 'STEREO_B'):
                if hdr['DETECTOR'] ==  'EUVI':
                    rx1 = y1
                    rx2 = y2
                    ry1 = hdr['naxis2'] - x2 - 1
                    ry2 = hdr['naxis2'] - x1 - 1
                    hdr['R1COL'] = hdr['P1ROW']
                    hdr['R2COL'] = hdr['P2ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL'] + 1
                    hdr['R2ROW'] = 2176 - hdr['P1COL'] + 1
            x1 = rx1
            x2 = rx2 
            y1 = ry1
            y2 = ry2
        hdr['DSTART1'] = x1+1 
        hdr['DSTART2'] = y1+1
        hdr['DSTOP1'] = x2+1
        hdr['DSTOP2'] = y2+1
  
    return im, hdr           
        
#|-------------------------|
#|--- Get EUVI pointing ---|
#|-------------------------|
def euvi_point(hdr, gtFile):
    """
    Function that corrects the pointing and updates the information
    in an EUVI header

    Input:
        hdr: an EUVI hdr
    
        gtFile: a file used to get sunvec
    
    Optional Input:
        downSize: size of the output image (in pixels)
                  *** assuming a square output

    Output:
        maps_out: a list of maps corresponding to the input fits files


    """    
    radeg = 180 / np.pi
    # assume passed a single header
    
    #|------------------------|
    #|--- Set to run SPICE ---|
    #|------------------------|
    icy = True    
    
    #|--------------------|
    #|--- Get sat info ---|
    #|--------------------|
    obss = hdr['obsrvtry'].upper()
    dets = hdr['detector'].upper()
    
    #|---------------------|
    #|--- STA EUVI Case ---|
    #|---------------------|
    if (obss == 'STEREO_A') & (dets == 'EUVI'):
        obs = 0
        obs_s = 'a'
        flp1 = 1.0 # E-W flipped during rectify, 1=true
        flp2 = 1.0 # S-N flipped during rectify, 1=true
        p2a   = 1.590 * 0.9986 # EUVI plate scale in arcsec (A=0.9986*B)
        off1  = 1027.5 # GT axis location on EUVI CCD
        off2  = 1110.6  #relative to p1col=0,p1row=0
        erol  = (1.245-1.125) / radeg # Roll of EUVI-N measured east from S/C N
        grol  = -0.005  # Roll of GT-N   measured east from EUVI N
        if 'rectrota' in hdr:
            if hdr['rectrota'] == 4:
                flp1 = (1.0-flp1)
                flp2 = (1.0-flp2)
                grol = grol + np.pi
                erol = erol + np.pi
        
        #|----------------------------------|
        #|--- Build pointing drift array ---|
        #|----------------------------------|  
        # Define the pointing drift records for A
        # If were assigning all this garbage by hand just skipping
        # the anytim part
        pdrec = []
        # k = 0
        aPD = {'obs':'a', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':344.6*86400.0,'c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 0
        aPD['ts'] = 6.6268800e+08
        aPD['te'] = 3.7869120e+09
        aPD['t0'] = 8.8361280e+08
        # no correction to cs
        pdrec.append(aPD)
        # k = 1
        aPD = {'obs':'a', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':344.6*86400.0,'c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1016
        aPD['ts'] = 8.9138880e+08
        aPD['te'] = 9.5454720e+08
        aPD['t0'] = 8.8361280e+08
        aPD['c1']  = np.array([ -0.19698,  0.40215, -0.02734,  0.02834,  0.02683, 0.01781,  0.00528,  0.00799,  0.70364])
        aPD['c2']  = np.array([  2.01189, -2.56016,  0.08225,  0.05025, -0.01835,  0.01585,  0.00151,  0.00655,  0.78864])
        pdrec.append(aPD)
        # k = 2 
        aPD = {'obs':'a', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':344.6*86400.0,'c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1016
        aPD['ts'] = 9.5454720e+08
        aPD['te'] = 1.0176192e+09
        aPD['t0'] = 9.4677120e+08
        aPD['c1']  = np.array([-1.21082,  1.17563, -0.03543, -0.01422, -0.00833, 0.01370, -0.00109, -0.01159,  0.33896])
        aPD['c2']  = np.array([ 1.61051,  0.02998,  0.01900,  0.16401, -0.01370, 0.04178,  0.00161, -0.01293, -0.99862])
        pdrec.append(aPD)
        # k = 3
        aPD = {'obs':'a', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':344.6*86400.0,'c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1016
        aPD['ts'] = 1.0176192e+09
        aPD['te'] = 3.7869120e+09
        aPD['t0'] = 1.0098432e+09
        aPD['c1']  = np.array([-1.59089,  0.96487, -0.00403, -0.00339, -0.02923, -0.04176,  0.01798,  0.01447,  0.26979])
        aPD['c2']  = np.array([ 2.28952, -0.68690, -0.10883,  0.22979, -0.00715,  0.03479,  0.00254,  0.01736,  3.86424])
        pdrec.append(aPD)
        # k = 4
        aPD = {'obs':'a', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':344.6*86400.0,'c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1021
        aPD['ts'] = 1.1479968e+09
        aPD['te'] = 1.4078016e+09
        aPD['t0'] = 1.0098432e+09
        aPD['c1']     = -pdrec[3]['c1']
        aPD['c1'][8]  =  pdrec[3]['c1'][8]   # no polarity change for exponent term
        aPD['c2']     = -pdrec[3]['c2']      # polarity change
        aPD['c2'][8]  =  pdrec[3]['c2'][8]
        pdrec.append(aPD)
        # k = 5
        aPD = {'obs':'a', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':344.6*86400.0,'c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1023
        aPD['ts'] = 1.1638080e+09
        aPD['te'] = 1.3437792e+09
        aPD['t0'] = 1.1638080e+09
        aPD['c1']  = np.array([1.16430,  0.11202,  0.05279,  0.05599, -0.01810, 0.01131,  0.00301, -0.00421, -0.40094])
        aPD['c2']  = np.array([-2.44524, -0.10207,  0.10195,  0.00222, -0.01788, 0.00797, -0.01294,  0.02516, -0.42572])
        pdrec.append(aPD)
        # k = 6
        aPD = {'obs':'a', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':344.6*86400.0,'c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1023
        aPD['ts'] = 1.3437792e+09
        aPD['te'] = 3.7869120e+09
        aPD['t0'] = 1.3437792e+09
        aPD['c1']  = np.array([ 2.94178, -0.19203, -0.00031,  0.07529, -0.02484, -0.00588, -0.00306,  0.00382,  1.35169])
        aPD['c2']  = np.array([-6.85677,  3.03264,  0.09955,  0.00847,  0.02142, 0.01430, -0.01344,  0.02884,  0.10000])
        pdrec.append(aPD)
        # k = 7
        aPD = {'obs':'a', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':344.6*86400.0,'c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1023
        aPD['ts'] = 1.4078016e+09
        aPD['te'] = 1.6645824e+09
        aPD['t0'] = 1.3437792e+09
        aPD['c1']     = -pdrec[6]['c1']
        aPD['c1'][8]  =  pdrec[6]['c1'][8]   # no polarity change for exponent term
        aPD['c2']     = -pdrec[6]['c2']      # polarity change
        aPD['c2'][8]  =  pdrec[6]['c2'][8]
        pdrec.append(aPD)
        
    #|---------------------|
    #|--- STB EUVI Case ---|
    #|---------------------|
    elif (obss == 'STEREO_B') & (dets == 'EUVI'):
        obs = 1
        obs_s = 'b'
        flp1 = 0.0 # E-W flipped during rectify, 1=true
        flp2 = 1.0 # S-N flipped during rectify, 1=true
        p2a   = 1.590 # EUVI-B plate scale in arcsec
        off1  = 1034.4  # GT axis location on EUVI CCD
        off2  = 1095.7  #relative to p1col=0,p1row=0
        erol  = -1.125 / radeg # Roll of EUVI-N measured east from S/C N
        grol  = -0.015  # Roll of GT-N   measured east from EUVI N
        if 'rectrota' in hdr:
            if hdr['rectrota'] == 1:
                flp1 = (1.0-flp1)
                flp2 = (1.0-flp2)
                grol = grol + np.pi
                erol = erol + np.pi
        
        #|----------------------------------|
        #|--- Build pointing drift array ---|
        #|----------------------------------|  
        # Define the pointing drift records for B
        pdrec = []
        # k = 0
        aPD = {'obs':'b', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':389.0*86400.0,' c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 0
        aPD['ts'] = 6.6268800e+08
        aPD['te'] = 3.7869120e+09
        aPD['t0'] = 8.8361280e+08
        pdrec.append(aPD)
        # no correction to cs
        # k = 1
        aPD = {'obs':'b', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':389.0*86400.0,' c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1016
        aPD['ts'] = 8.9398080e+08
        aPD['te'] = 9.5981760e+08
        aPD['t0'] = 8.8620480e+08
        aPD['c1']  = np.array([-3.61068,  3.71167,  0.04840, -0.34213, -0.00013, -0.02756, -0.00926, -0.00304,  0.10000])
        aPD['c2']  = np.array([-3.50374,  3.65205,  0.06367, -0.01608,  0.03177, -0.12241, -0.01370,  0.08606,  0.38241])
        pdrec.append(aPD)
        # k = 2 
        aPD = {'obs':'b', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':389.0*86400.0,' c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1016
        aPD['ts'] = 9.5981760e+08
        aPD['te'] = 1.0149408e+09
        aPD['t0'] = 9.5204160e+08
        aPD['c1']  = np.array([-7.40505,  6.93326, -0.06104, -0.46446, -0.04072, -0.09851,  0.00199, -0.02096,  0.10000])
        aPD['c2']  = np.array([ -3.09684,  1.37612,  0.03446,  0.08630, -0.01247, -0.04378,  0.04979,  0.10301,  0.49868])
        pdrec.append(aPD)
        # k = 3
        aPD = {'obs':'b', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':389.0*86400.0,' c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1016
        aPD['ts'] = 1.0149408e+09
        aPD['te'] = 3.7869120e+09
        aPD['t0'] = 1.0071648e+09
        aPD['c1']  = np.array([-8.56397,  7.15365, -0.49817,  0.52057,  0.02147, -0.05027, -0.00185, -0.02525,  0.10000])
        aPD['c2']  = np.array([-2.68742,  0.23191,  0.14165, -0.38149,  0.09431, 0.05550,  0.09927,  0.07492,  4.66914])
        pdrec.append(aPD)
        # k = 4
        aPD = {'obs':'b', 'ver':0, 'ts':0., 'te':0., 't0':8.8361280e+08, 'torb':389.0*86400.0,' c1':np.zeros(9), 'c2':np.zeros(9)}
        aPD['ver'] = 1021
        aPD['ts'] = 1.1479968e+09
        aPD['te'] = 1.4078016e+09
        aPD['t0'] = 1.0098432e+09
        aPD['c1']     = -pdrec[3]['c1']
        aPD['c1'][8]  =  pdrec[3]['c1'][8]   # no polarity change for exponent term
        aPD['c2']     = -pdrec[3]['c2']      # polarity change
        aPD['c2'][8]  =  pdrec[3]['c2'][8]
        pdrec.append(aPD)

    #|-----------------------------|
    #|--- Other Cases (uncoded) ---|
    #|-----------------------------|
    else:
        sys.exit('Non EUIVI header passed to euvi_point')
        
    crpix0=[hdr['crpix1'],hdr['crpix2']]
    cdelt0=[hdr['cdelt1'],hdr['cdelt2']]
    
    er1_1 =  np.cos(erol) # convert EUVI roll into PC matrix
    er1_2 = -np.sin(erol)
    er2_1 =  np.sin(erol)
    er2_2 =  np.cos(erol)
    
    #|----------------|
    #|--- Get svec ---|
    #|----------------|
    # Just take the att_file from hdr, not entirely sure what 370-400 doing
    #att_file = hdr['att_file'].replace('+1GT','')
    t = datetime.datetime.strptime(hdr['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f" )
    anytim = (t-idl_base_date).total_seconds() +  hdr['exptime'] + 2.0 # end exp + 2 sec
    dsun = hdr['dsun_obs'] / 1.496e11 # sun distance in AU
    fpsoff = [hdr['fpsoffy'], hdr['fpsoffz']] # FPS pointing offset
    svecg = scc_gt2sunvec(anytim, dsun, fpsoff, obs_s, gtFile)
    # rotate svec from GT to EUVI system
    svec = [svecg[0]*np.cos(grol) - svecg[1]*np.sin(grol), svecg[0]*np.sin(grol) + svecg[1]*np.cos(grol)] 
    
    # Assume we are improving the pointing
    
    #|----------------------|
    #|--- Pull pd record ---|
    #|----------------------|
    # Cannot figure out where IDL pulls this saved info from or how two calculate
    # but the few (from 2023 test cases) all have this so run for now
    # Suspect may need to change for earlier and/or STB events
    usever = 1023
    #find the appropriate pointing drift coefficient record
    pdidx = None
    for ii in range(len(pdrec)):
        if (pdrec[ii]['ver'] <= usever) &  (anytim >= pdrec[ii]['ts']) & (anytim <= pdrec[ii]['te']):
            pdidx = ii
    if pdidx:
        pd = pdrec[pdidx]
    else:
        sys.exit('Error finding matching pd record in euvi_point')
        
    #|--------------------------------|
    #|--- Calc pointing correction ---|
    #|--------------------------------|
    tpd = (anytim - pd['t0']) / pd['torb'] # orb. phase
    pd1 = pd['c1'][0] + pd['c1'][1] * np.exp(-pd['c1'][8] * tpd) # const+exp term
    pd2 = pd['c2'][0] + pd['c2'][1] * np.exp(-pd['c2'][8] * tpd) # const+exp term
    for k in [1,2,3]:
        ppd = 2.0 * np.pi * k * tpd
        pd1 = pd1 + pd['c1'][2*k] * np.sin(ppd) + pd['c1'][2*k+1] * np.cos(ppd) 
        pd2 = pd2 + pd['c2'][2*k] * np.sin(ppd) + pd['c2'][2*k+1] * np.cos(ppd)
    
    #|----------------------------|
    #|--- Recalc header values ---|
    #|----------------------------|
    # get the summing factors, assuming image is rectified:
    sumfac1 = 2.**int(hdr['summed'] - 1)   
    sumfac2 = sumfac1
    # recalculate crpix from scratch if feasible (assuming it is)
    hdr['crpix1'] = flp1 * hdr['naxis1'] + ((1.-2.*flp1) * (off1-hdr['p1row']) - flp1 + (1.-2.*obs) * svec[1] / p2a - (sumfac1-1.)/2.) / sumfac1 + 1.
    hdr['crpix2'] = flp2 * hdr['naxis2'] + ((1.-2.*flp2) * (off2-hdr['p1col']) - flp2 + (1.-2.*obs) * svec[0] / p2a - (sumfac2-1.)/2.) / sumfac2 + 1.
    
    # subtract pointing drift
    hdr['crpix1'] = hdr['crpix1'] - pd1/sumfac1
    hdr['crpix2'] = hdr['crpix2'] - pd2/sumfac2
    
    #make sure crval1,2 are zero
    hdr['crval1'] = 0.0
    hdr['crval2'] = 0.0
    
    # calculate cdelt.
    hdr['cdelt1'] = p2a * sumfac1
    hdr['cdelt2'] = p2a * sumfac2
    
    # Set instrument offset keywords
    if 'ins_x0' in hdr:
        hdr['ins_x0'] = p2a*((1.-2.*flp1)*((2048+1)/2.-off1)+pd1)    # arcsec
        hdr['ins_y0'] = p2a*((1.-2.*flp2)*((2048+1)/2.-off2)+pd2)    # arcsec
    if 'crpix1a' in hdr:    
        hdr['crpix1a'] = hdr['crpix1']
        hdr['crpix2a'] = hdr['crpix2']
    if 'cdelt1a' in hdr:    
        hdr['cdelt1a'] = -hdr['cdelt1']/3.6e3
        hdr['cdelt2a'] = hdr['cdelt2']/3.6e3
    # dont hit the ra/dec section
    # Don't need to show changes
 
    #|-------------------------|
    #|--- Check roll values ---|
    #|-------------------------|
    #sanity check between existing PC, CROTA and SC_ROLL:
    roll_bad = 0
    if 'sc_roll' in hdr:
        scrota = hdr['sc_roll'] + erol* 180 / np.pi
        pcrota = 180 / np.pi * np.atan2(hdr['pc2_1'],hdr['pc1_1'])
        if np.abs(scrota - pcrota) > 0.1: roll_bad = 1
        if 'crota' in hdr:
            if np.abs(hdr['crota'] - scrota) > 0.1: roll_bad = 1
            
    #|------------------------|
    #|--- Roll corrections ---|
    #|------------------------|
    # Assume we are doing the roll 
    yr0 = datetime.datetime(int(hdr['date-obs'][:4]), 1, 1, 0,0,0)
    sDOY = str(int((t-yr0).total_seconds()/3600./24. + 1)).zfill(3)
    # Haven't found a test case that hits this post kernal reloc but keeping it
    # here in case need to re incorporate later
    #loadSomeSTEREO(hdr['date-obs'][:4]+'_'+sDOY)

    rollrada = get_sunspyce_roll(hdr['date-obs'],'st'+obs_s,system='GEI')[0] / (180 / np.pi)
    # convert roll into PC matrix
    hdr['pc1_1a'] =  np.cos(rollrada)        
    hdr['pc1_2a'] = -np.sin(rollrada)
    hdr['pc2_1a'] =  np.sin(rollrada)
    hdr['pc2_2a'] =  np.cos(rollrada)
    pc1_1 = hdr['pc1_1a']
    pc1_2 = hdr['pc1_2a']
    pc2_1 = hdr['pc2_1a']
    pc2_2 = hdr['pc2_2a']
    # calculate updated pc matrix: equivalent to er # pc.
    hdr['pc1_1a'] = er1_1 * pc1_1 + er1_2 * pc2_1
    hdr['pc1_2a'] = er1_1 * pc1_2 + er1_2 * pc2_2
    hdr['pc2_1a'] = er2_1 * pc1_1 + er2_2 * pc2_1
    hdr['pc2_2a'] = er2_1 * pc1_2 + er2_2 * pc2_2
    
    hcv = get_sunspyce_hpc_point(hdr['date-obs'],'st'+obs_s)
    rollrad = hcv[2] / (180 / np.pi)
    if 'crota' in hdr:
        hdr['crota'] = (hcv[2] + erol*(180 / np.pi)+180.0) % 360.0 - 180.0
    
    hdr['pc1_1'] =  np.cos(rollrad)    # convert roll into PC mtrx
    hdr['pc1_2'] = -np.sin(rollrad)
    hdr['pc2_1'] =  np.sin(rollrad)
    hdr['pc2_2'] =  np.cos(rollrad)
    pc1_1 = hdr['pc1_1']
    pc1_2 = hdr['pc1_2']
    pc2_1 = hdr['pc2_1']
    pc2_2 = hdr['pc2_2']
    # calculate updated pc matrix: equivalent to er # pc.
    hdr['pc1_1'] = er1_1 * pc1_1 + er1_2 * pc2_1
    hdr['pc1_2'] = er1_1 * pc1_2 + er1_2 * pc2_2
    hdr['pc2_1'] = er2_1 * pc1_1 + er2_2 * pc2_1
    hdr['pc2_2'] = er2_1 * pc1_2 + er2_2 * pc2_2
    
    # unfortunately, SECCHI supports a slew of secondary and obsolete keywords
    if 'xcen' in hdr:
        ceni = (hdr['naxis1']+1)/2. - hdr['crpix1']
        cenj = (hdr['naxis2']+1)/2. - hdr['crpix2']
        hdr['xcen'] = hdr['cdelt1'] * (hdr['pc1_1']*ceni + hdr['pc1_2']*cenj)
        hdr['ycen'] = hdr['cdelt2'] * (hdr['pc2_1']*ceni + hdr['pc2_2']*cenj)
    return hdr
        
#|------------------------------|
#|--- Get sunvec in GT Frame ---|
#|------------------------------|
def scc_gt2sunvec(anytim, sund, gtdata, obs, gtFile, doRad=False):
    """
    Function that calculates a sun vector

    Input:
        anytim: a time in the IDL anytime format. This is total seconds from
                some arbitrary time that IDL picked
    
        sund: distance from the sun (in AU)
    
        gtdata: the FPS offset from a header
    
        obs: an observatory string
    
        gtFile: a helper file (IDL save file)
    
    Optional Input:
        doRad: return the result in radians (defaults to False )
    
    Output:
        sunvec: the y and z sun vector components in the GT frame 


    """    
    #|----------------|
    #|--- STA Case ---|
    #|----------------|
    if obs.lower() == 'a':
        cc    = [[1.0000, 0.715e-8, 1.341e-5, 0.329e-8], [1.2843, 1.883e-7, 2.203e-4, 0.625e-7], [1.1739, 1.759e-6, 1.718e-3, 0.511e-6]]
        yrgain = (378./256.) * 1.100    # as of 2007-04-30
        zrgain = (378./256.) * 1.100    # as of 2007-04-30
        ypgain = yrgain * 1.389  # pri/red based on nominal preamp resistor values
        zpgain = zrgain * 1.389  # pri/red based on nominal preamp resistor values
    #|----------------|
    #|--- STB Case ---|
    #|----------------|
    else:
        cc    = [[1.0000, 0.176e-8, 0.641e-5, 0.162e-8], [0.9505, 0.265e-7, 0.451e-4, 0.281e-7], [4.7232, 0.915e-6, 1.020e-3, 0.065e-6]]
        yrgain = (378./256.) * 0.920    # as of 2007-04-30
        zrgain = (378./256.) * 0.920    # as of 2007-04-30
        ypgain = yrgain * 1.381  # pri/red based on nominal preamp resistor values
        zpgain = zrgain * 1.381  # pri/red based on nominal preamp resistor values
    
    #|--------------------------------|
    #|--- Port of scc_time2gtparms ---|
    #|--------------------------------|
    # yes it's parms not params    
    gt = readsav(gtFile)
    if obs.lower() == 'a':
        fulldb = gt['p0'].a[0]
    fulldb = fulldb[np.where(fulldb.flg ==1 )]
    myIdx = np.max(np.where(fulldb.t <= anytim))
    mydb = fulldb[myIdx]
    
    #|---------------------------|
    #|--- Pull GT file values ---|
    #|---------------------------|
    # Determine whether prime or redundant diodes, based on GT gain
    if np.sum(mydb.cg2) < 450*4:
        redun = 1
    else:
        redun = 0
    # Create raw diode difference signal
    if len(gtdata) == 2: # on-board calibrated signals
        # "undo" on-board calibration: 1. re-apply bias
        gtr  = [float(gtdata[0] + mydb.cg4[0]), float(gtdata[1] + mydb.cg4[1])]
        # "undo" on-board calibration: 2. de-apply gain
        yraw = gtr[0] * 512.0 / (mydb.cg2[0]+mydb.cg2[1])
        zraw = gtr[1] * 512.0 / (mydb.cg2[2]+mydb.cg2[3])
    else:
        sys.exit('Havent ported cases where gt has more than two values')
        
    #|------------------------|
    #|--- Calculate Things ---|
    #|------------------------|
    # Apply sun distance to the coefficients
    rr = [1.0,sund-1.0,(sund-1.0)*(sund-1.0)]
    cs  = np.matmul(rr, cc)
    
    # Calibrate yraw,zraw for 1 AU (correction matrix is normalized to 1 AU)
    # Flip signs of cs[2] for Y redundant and Z prime
    # (signs of initial coefficient matrix are for Y prime and Z redundant)
    flip = [1.0,1.0,-1.0,1.0]
    if redun:
        y1 = yraw * yrgain
        z1 = zraw * zrgain
        cy = cs * flip
        cz = cs
    else:
        y1 = yraw * ypgain
        z1 = zraw * zpgain
        cy = cs
        cz = cs * flip
    
    # prepare polynomial of GT signal for correction
    yy = [y1,y1*y1*y1,y1*z1,y1*z1*z1]
    zz = [z1,z1*z1*z1,z1*y1,z1*y1*y1]
    
    # apply correction
    svec = [np.matmul(cy, np.transpose(yy)), np.matmul(cz, np.transpose(zz))]
    
    # so far, units are 2e-7 rad.  Change to arcsec (default) or radian
    fact= 180 / np.pi *3.6*2e-4
    if doRad:
        fact = 2e-7
    svec = np.array(svec) * fact 
    return svec     
    
#|---------------------------|
#|--- Icerdiv2 Correction ---|
#|---------------------------|
def scc_icerdiv2(hdr,img):
    """
    Function that does the icerdiv2 correction

    Input:
        hdr: an fits file header
    
        img: the associated fits image data
    
    Output:
        hdr: an updated header
    
        img: the corrected image


    """    
    #|-------------------------------|
    #|--- Process IP tag from hdr ---|
    #|-------------------------------|    
    ip = hdr['ip_00_19']
    # Make sure the string is long enough, could be trimmed
    if len(ip) < 60: ip=' '+ip  
    if len(ip) < 60: ip=' '+ip  
    # Sorted out this python equiv in sebip
    ipEnc = ip.encode(encoding='utf-8')
    byteIt = np.array([ipEnc[i] for i in range(60)]) 
    ip = np.array([chr(byteIt[i*3])+chr(byteIt[i*3+1])+chr(byteIt[i*3+2]) for i in range(20)])
    ip = ip.astype(int)
    w = np.where(ip != 0)[0]
    nip = len(w)
    icradiv2=0
    idecdiv2=0
    datap01=hdr['datap01']
    biasmean=hdr['biasmean']
    
    # Not hitting pipeline 
    
    #|---------------------------|
    #|--- Apply IP conditions ---|
    #|---------------------------|
    # Calculate various factors based on conditions
    icer = (ip[nip-1] >= 90) & (ip[nip-1] < 102)
    div2 = ip[nip-2] == 1
    noticfilt = (ip[nip-2] < 106) or (ip[nip-2] > 112)
    nosubbias = not (103 in ip)
    biasmp01  = (biasmean/2)-datap01
    p01ltbias = (np.abs(biasmp01) < 0.02*(biasmean/2))
    
    # logic to determine whether data was most likely divided by 2
    # the first part finds an explicit DIV2, the second an implicit one in ICER
    # this logic does not determine whether the correction was already applied    
    domul2 =  icradiv2 or idecdiv2 or (icer & noticfilt & nosubbias & p01ltbias)
    
    #|---------------------|
    #|--- Multiply by 2 ---|
    #|---------------------|
    if domul2:
        # not making the same form of 2
        img = img * 2
        hdr['datap01'] = hdr['datap01'] * 2
        hdr['datamin'] = hdr['datamin'] * 2
        hdr['datamax'] = hdr['datamax'] * 2
        hdr['dataavg'] = hdr['dataavg'] * 2
        hdr['datap10'] = hdr['datap10'] * 2
        hdr['datap25'] = hdr['datap25'] * 2
        hdr['datap75'] = hdr['datap75'] * 2
        hdr['datap90'] = hdr['datap90'] * 2
        hdr['datap95'] = hdr['datap95'] * 2
        hdr['datap98'] = hdr['datap98'] * 2
        hdr['datap99'] = hdr['datap99'] * 2
        hdr['div2corr'] = 'T'
        
        if (idecdiv2 & icradiv2): hdr['div2corr'] = 'F'
        print ('Corrected for icerdiv2')
        
    return hdr, img
        
    
    
    
    
    