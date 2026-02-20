"""
Module to process observations for the wombat GUI 

The main wrapper assumes that files are stored in the structure set up
by wombatPullObs and will search for files in a given time range
and pass those to the appropriate processing scripts. The process functions
can be called on their own with a different wrapper if a different folder
structure is used


Inputs:
    times: an array with [startTime, endTime] where both
           times are of the format YYYY-MM-DDTHH:MM 
           (or any format that sunpy parse_time likes)
    
    inst: an array with the tags for which data to pull
        Available tags:
            AIAnum  = SDO AIA where num represents a wavelength from [94, 131, 171*, 
                      193*, 211, 304*, 335, 1600, 1700] with * most common
            C2      = LASCO C2
            C3      = LASCO C3
            COR1    = STEREO COR1
            COR2    = STEREO COR2    
            EUVInum = STEREO EUVI where num is a wavelength from [171, 195, 284, 304]
            HI1     = STEREO HI1
            HI2     = STEREO H2
            SoloHI  = All quadrants from Solar Orbiter HI
            SoloHI1 = Quadrant 1 from Solar Orbiter HI
            SoloHI2 = Quadrant 2 from Solar Orbiter HI
            SoloHI3 = Quadrant 3 from Solar Orbiter HI
            SoloHI4 = Quadrant 4 from Solar Orbiter HI
            WISPR   = Both inner and outer from PSP WISPR
            WISPRI  = Inner only from PSP WISPR
            WISRPO  = Outer only from PSP WISPR
            * all STEREO values will pull A and B (as available)
    
Optional Inputs:
    inFolder:  the top level folder name for where processed data will be saved
               defaults to pullFolder/ (from wombatPullObs)

    outFolder: the top level folder name for where processed data will be saved
               defaults to wbFits/ 

    outFile:   the name of a file with the names of the processed files formatted
               so that it can be used to launch the wombat gui
               defaults to WBobslist.txt
    
    downSize:  maximum resolution to save the processed fits files 

Outputs:

    If saveFits is set the code sets up a structure at outFolder with nested satellite
    and instrument folders (if it doesn't already exist). It will then dump the 
    processed files into the appropriate folders. It also creates a file with
    the names of the processed files separated by instrument, which can be used
    to launch the wombat gui

External Calls:
    setupFolderStructure from wombatProcessObs
    secchi_prep from secchi_prep
    c2_prep, c3_prep from lasco_prep
    wispr_prep from wispr_prep
    solohi_fits2grid from solohi_prep
    aia_prep from aia_prep

Usage:
    from wombatPullObs import pullObs
    times = ['YYYY/MM/DDTHH:MM', 'YYYY/MM/DDTHH:MM']
    sats  = ['AIA', 'COR2', 'WISPR']
    waves = [171]
    inFolder = '/Users/kaycd1/wombat/pullFolder/'
    outFolder = '/Users/kaycd1/wombat/wbFits/'
    outFile   = 'WBobslist.txt'
    pullObs(times, sats, waves, inFolder=inFolder, outFolder=outFolder, outFile=outFile)

"""

import sys, os
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import TimeDelta
from sunpy.time import parse_time
import sunpy
import pickle

sys.path.append('prepCode/') 
sys.path.append('wombatCode/') 
from secchi_prep import secchi_prep
from wispr_prep import wispr_prep
from lasco_prep import c2_prep, c3_prep, reduce_level_1
from solohi_prep import solohi_fits2grid
from aia_prep import aia_prep
from wcs_funs import fitshead2wcs, wcs_get_pixel, wcs_get_coord
from wombatPullObs import setupFolderStructure
from sunspyce import load_common_kernels, load_psp_kernels, load_solo_kernels
import wombatMass as wM

# |-------------------------------|
# |------- Setup Time Stuff ------|
# |-------------------------------|
def setupTimeStuff(times):
    """
    Function to convert pair of strings in times into useful other variables

    Inputs:
        times: an array with [startTime, endTime] in any string format accepted by 
               parse_time (from sunpy)
     
    Outputs:
        ymds: an array of dates within the time range in the format YYYYMMDD
    
        hms: the start time and end time in the format HHMM

    """    
    # Convert strings to astropy time objects
    startT = times[0]
    endT   = times[1]
    try:
        startAPT = parse_time(startT)
    except:
        sys.exit('Error in starting time format. Need any format accepted by sunpy parse_time')
    try:
        endAPT = parse_time(endT)
    except:
        sys.exit('Error in ending time format. Need any format accepted by sunpy parse_time')

    # |----------------------------------|
    # |-------- Get Day of Year ---------|
    # |----------------------------------|
    # Want to account for overlapping an end of year so
    # don't confuse anything going from 365 to 1
    # Get start DoY
    stDOY = int(startAPT.yday.split(':')[1])
    # Get end DoY
    # -> easy if same year
    if startAPT.datetime.year == endAPT.datetime.year:
        enDOY = int(endAPT.yday.split(':')[1])
    # Split year case
    else:
        nye = parse_time(str(startAPT.datetime.year) + '/12/31')
        nye_doy = int(nye.yday.split(':')[1]) # get last doy (good for leap years)
        enDOY = int(endAPT.yday.split(':')[1]) + nye_doy
    
    # |----------------------------------|
    # |----- Check number of days -------|
    # |----------------------------------|
    # Check if start/end same day, get ymd string
    if stDOY == enDOY:
        ymds = [str(startAPT)[:10].replace('-','')]
    # Otherwise get ymd strings for each date
    else:
        nDays = enDOY - stDOY + 1
        ymds = []
        for i in range(nDays):
            nowDay = startAPT + TimeDelta(i * u.day)
            ymds.append(str(nowDay)[:10].replace('-',''))
        
    # |----------------------------------|
    # |------ Get start/end times -------|
    # |----------------------------------|
    hm0 = str(startAPT)[11:16].replace(':','')
    hmf = str(endAPT)[11:16].replace(':','')
    hms = [hm0,hmf]
    
    return ymds, hms
    
    

# |------------------------------------------------------------|
# |----------------- Process AIA Observations -----------------|
# |------------------------------------------------------------|
def processAIA(times, wavs, inFolder='pullFolder/SDO/AIA/', saveFits=False, outFolder='wbFits/SDO/AIA/', downSize=1024):
    """
    Function to process level 1 AIA observations using AIA_prep
    
    This is just a basic wrapper of existing aiapy procedures, not a port
    of IDL code like the other process scripts. The resulting fits files
    are in DN units, not total brightness but this is sufficient since we
    do not support EUV masses. 

    Inputs:
        times: an array with [startTime, endTime] where the times are strings
               that can be interpreted by sunpy parse_time
        
        wavs:  an array of wavelength strings
    
    Optional Inputs:
        inFolder: top folder for unprocessed results, will open from outFolder/wav/
                  defaults to pullFolder/SDO/AIA/
    
        saveFits: flag whether to save the processed results as new fits files or not
                  defaults to False since wombat intends to pass the results to the wrapper
                  that organizes a pickle file to quick load the GUI
               
        outFolder: top folder for processed results, will be save in outFolder/SDO/AIA/wav/
                   defaults to wbFits/SDO/AIA/
    
        downSize: maximum resolution for the processed images  
                  defaults to 1024 (square image)
        
    Outputs:
        proIms:     a dictionary with the instrument as key and the entry corresponding to
                    [[im1, im2, im3,...], [hdr1, hdr2, hdr3,...]] for the processed data
    
        outLines:   a dictionary of paths to the original fits files used to make processed results
    
    Actions:
        If saveFits is set the processed fits files within the appropriate location in the
        outFolder substructure.   

    """
    
    # |----------------------------------|
    # |--------- Find the files ---------|
    # |----------------------------------|
    nWavs = len(wavs)
    AIAfiles = [[] for i in range(nWavs)]
    
    # Pull everything in that wavelength folder
    for i in range(nWavs):
        AIAfiles[i] = os.listdir(inFolder+wavs[i])
    
    # Make sure we found something before moving on
    nFound = 0
    for i in range(nWavs):
        nFound += len(AIAfiles[i])
    # Return if nothing found
    if nFound == 0:
        print ('No AIA files found')
        return None
        
    # Get the date and time strings
    ymds, hms = setupTimeStuff(times)
    nDays = len(ymds)
    
    # Format the desired date string(s) for AIA file names
    AIAdatestrs = [ymd[:4]+'_'+ymd[4:6]+'_'+ymd[6:] for ymd in ymds]
    
    
    # |----------------------------------|
    # |--------- Sort the files ---------|
    # |----------------------------------|
    # Check all the files to see if they have a matching date str
    # If date matches then check the hour/min on first/last date
    # Add into separate arrays for each wavelength
    goodFiles = [[] for i in range(nWavs)]
    for i in range(nWavs):
        for aF in AIAfiles[i]:
            # Check if it matches any of the days
            for j in range(nDays):
                if AIAdatestrs[j] in aF:
                    hm = aF[25:30]
                    addIt = True
                    # If on first day but early toss
                    if (j == 0) & (hm < hms[0]):
                        addIt = False
                    # If on last day but late toss
                    elif (j == nDays-1) & (hm > hms[1]):
                        addIt = False
                    # Add it to the list if a keeper   
                    if addIt:
                        goodFiles[i].append(inFolder+wavs[i]+'/'+aF)
                        
    # Make sure we found something in the date range before moving on
    nFound = 0
    for i in range(nWavs):
        nFound += len(goodFiles[i])
    # Return if nothing found
    if nFound == 0:
        print ('No matching AIA files found')
        return None
    
    
    # |----------------------------------|
    # |------- Process the files --------|
    # |----------------------------------|                    
    print ('|--- Processing SDO AIA ---|') 
    outLines = {}   
    # Set up dictionary to hold the im/hdr outputs
    proIms = {}
    
    # Loop through each wavelength        
    for i in range(nWavs):
        proIms['aia'+str(wavs[i])] = [[], []]
        outLines['aia'+str(wavs[i])] = []
        if len(goodFiles[i]) > 0:
            print ('|--- Processing SDO AIA '+str(wavs[i])+'---|')      
            # Make an array and sort alphabetically = time sorted      
            goodFiles[i] = np.sort(np.array(goodFiles[i]))
            # Pass to the prep scripts
            print ('Running aia_prep, this may take some time...')
            ims = aia_prep(goodFiles[i], downSize=downSize)
            # Old load script version
            # If this is the first image add the instrument header to outlines
            #if saveFits:
            #    if len(ims) > 0:
            #        outLines.append('SDO_AIA_'+str(wavs[i]) + '\n')
            # Take processed result, add keywords, and save
            for k in range(len(ims)):
                #print ('On file '+str(k+1)+' out of '+str(len(ims)))
                im = ims[k]
                # Add extra keywords wombat wants
                im.meta['OBSRVTRY'] = 'SDO'
                im.meta['DETECTOR'] = 'AIA'
                im.meta['SC_ROLL'] = 0.
                # Set up output filename
                ymd = im.meta['DATE-OBS'].replace('-','').replace(':','')[:15]  
                fitsName = 'wbpro_aia'+str(wavs[i])+'_'+ymd+'.fits'
                if saveFits:
                    # Save the fits file
                    fullOut = outFolder+wavs[i]+'/'+fitsName
                    im.save(fullOut, overwrite=True)
                    # Add it to the list of avail obs
                outLines['aia'+str(wavs[i])].append(goodFiles[i][k])
                proIms['aia'+str(wavs[i])][0].append(im.data)
                proIms['aia'+str(wavs[i])][1].append(im.meta)             
            
        else:
            print ('No files found for AIA '+str(wavs[i]))
            proIms, outLines = None, None     
    return proIms, outLines

    

# |------------------------------------------------------------|
# |--------------- Process LASCO Observations -----------------|
# |------------------------------------------------------------|
def processLASCO(times, insts, inFolder='pullFolder/SOHO/LASCO/', outFolder='wbFits/SOHO/LASCO/', downSize=1024, prepDir='prepFiles/soho/lasco/', saveFits=False):
    """
    Function to process level 0.5 LASCO observations using c2_prep/c3_prep
    
    This is a wrapper to find the data and pull the ported versions of IDL solarsoft
    routines. The resulting fits files are in total brightness and a near exact match
    to using the original IDL procedures. These files can be passed directly to the
    wombat mass calculation procedure

    Inputs:
        times: an array with [startTime, endTime] where the times are strings
               that can be interpreted by sunpy parse_time
        
        insts: strings for selected instruments (C2 or C3)
    
    Optional Inputs:
        inFolder: top folder for unprocessed results, will open from outFolder/inst/
                  defaults to pullFolder/SOHO/LASCO/
               
        saveFits: flag whether to save the processed results as new fits files or not
                  defaults to False since wombat intends to pass the results to the wrapper
                  that organizes a pickle file to quick load the GUI

        outFolder: top folder for processed results, will be save in outFolder/SOHO/inst/
                   defaults to wbFits/SOHO/LASCO/
        
        prepDir:   folder where the prep files for LASCO are stored
                   defaults ot prepFiles/soho/lasco
    
        downSize: maximum resolution to save the processed fits files 
                  *** not implemented in c2/3_prep yet so key is currently ignored ***
        
    Outputs:
        proIms:     a dictionary with the instrument as key and the entry corresponding to
                    [[im1, im2, im3,...], [hdr1, hdr2, hdr3,...]] for the processed data
   
        outLines:   a dictionary of paths to the original fits files used to make processed results
    
    Actions:
        If saveFits is set the processed fits files within the appropriate location in the
        outFolder substructure.   
    
    """
    # |----------------------------------|
    # |--------- Find the files ---------|
    # |----------------------------------|
    nInsts = len(insts)
    LASCOfiles = [[] for i in range(nInsts)]
    for i in range(nInsts):
        LASCOfiles[i] = os.listdir(inFolder+insts[i])
    
    # Make sure we found something before moving on
    nFound = 0
    for i in range(nInsts):
        nFound += len(LASCOfiles[i])
    # Return if nothing found
    if nFound == 0:
        print ('No LASCO files found')
        return None
        
    # Get the date and time strings
    ymds, hms = setupTimeStuff(times)
    nDays = len(ymds)
    
    # Don't need to format time strings for LASCO, already did during pull
    
    # |----------------------------------|
    # |--------- Sort the files ---------|
    # |----------------------------------|
    goodFiles = [[] for i in range(nInsts)]
    for i in range(nInsts):
        for aF in LASCOfiles[i]:
            for j in range(nDays):
                if (ymds[j] in aF):
                    hm = aF[9:13]
                    addIt = True
                    if (j == 0) & (hm < hms[0]):
                        addIt = False
                    elif (j == nDays-1) & (hm > hms[1]):
                        addIt = False
                    if addIt:
                        goodFiles[i].append(inFolder+insts[i]+'/'+aF)
    # Make sure we found something in the date range before moving on
    nFound = 0
    for i in range(nInsts):
        nFound += len(goodFiles[i])
    # Return if nothing found
    if nFound == 0:
        print ('No matching LASCO files found')
        return None
    
    
    # |----------------------------------|
    # |------- Process the files --------|
    # |----------------------------------|                    
    # Sort and array-ify
    for i in range(nInsts):
        goodFiles[i] = np.sort(np.array(goodFiles[i]))
    
    # Add in the actual processing, saving, and output to runFile  
    print ('|---- Processing LASCO ----|')
    # Set up dictionary to hold the im/hdr outputs
    proIms = {}
    outLines = {}       
    for i in range(nInsts):
        if len(goodFiles[i]) > 0:
            print ('|---- Processing LASCO ' + insts[i] + ' ----|')
            proIms[insts[i]] = [[], []]
            outLines[insts[i]] = []
            # Make an array and sort alphabetically = time sorted      
            goodFiles[i] = np.sort(np.array(goodFiles[i]))
            # Old file loading stuff - to rm
            #if saveFits:
            #    outLines.append('LASCO_' + str(insts[i]) + '\n')
                
            if insts[i] == 'C2':
                ims, hdrs = reduce_level_1(goodFiles[i], prepDir)  
            if insts[i] == 'C3':    
                ims, hdrs = reduce_level_1(goodFiles[i], prepDir)     
                 
            for j in range(len(ims)):
                ymd = hdrs[j]['DATE-OBS'].replace('/','')+'T'+hdrs[j]['TIME-OBS'].replace(':','')[:6]
                fitsName = 'wbpro_lasco'+insts[i]+'_'+ymd+'.fits'
                hdrs[j]['OBSRVTRY'] = 'SOHO'
                hdrs[j].remove('HISTORY', remove_all=True, ignore_missing=True)
                hdrs[j]['HISTORY'] = 'Offset_bias applied but header stripped bc made astropy angry'
                outLines[insts[i]].append(goodFiles[i][j])
                proIms[insts[i]][0].append(ims[j])
                proIms[insts[i]][1].append(hdrs[j])
                if saveFits:
                    # Save the fits file
                    fullOut = outFolder+insts[i]+'/'+fitsName
                    fits.writeto(fullOut, ims[j], hdrs[j], overwrite=True)
                    
                           
    return proIms, outLines



# |------------------------------------------------------------|
# |-------------- Process SoloHI Observations -----------------|
# |------------------------------------------------------------|
def processSoloHI(times, insts, inFolder='pullFolder/SolO/SoloHI/', outFolder='wbFits/SolO/SoloHI/', saveFits=False):
    """
    Function to process the SoloHI images to mosaics or just pass along singles
    
    This is a wrapper to find the data and pull the ported versions of IDL solarsoft
    routines. The resulting fits files are in total brightness and a near exact match
    to using the original IDL procedures. These files can be passed directly to the
    wombat mass calculation procedure

    Inputs:
        times: an array with [startTime, endTime] where the times are strings
               that can be interpreted by sunpy parse_time
        
        insts: strings for selected instruments (Mosaic, SoloHI# where # in 1-4)
    
    Optional Inputs:
        inFolder: top folder for unprocessed results, will open from outFolder/#/
                  defaults to pullFolder/SolO/SoloHI/
               
        saveFits: flag whether to save the processed results as new fits files or not
                  defaults to False since wombat intends to pass the results to the wrapper
                  that organizes a pickle file to quick load the GUI

        outFolder: top folder for processed results, will be save in outFolder/inst/
                   defaults to wbFits/SolO/SoloHI/
    
    Outputs:
        proIms:     a dictionary with the instrument as key and the entry corresponding to
                    [[im1, im2, im3,...], [hdr1, hdr2, hdr3,...]] for the processed data
   
        outLines:   a dictionary of paths to the original fits files used to make processed results
                    for mosaic images this is a series of arrays of the four fits files used for
                    each mosaic image
    
    Actions:
        If saveFits is set the processed fits files within the appropriate location in the
        outFolder substructure.   

    """
    
    # |----------------------------------|
    # |------ Check mosaic/single -------|
    # |----------------------------------|
    # See if were doing a mosaic or just a single
    # Single just moves the fits without processing
    nInsts = len(insts)
    doMosaic = False
    if 'Mosaic' in insts:
        doMosaic = True
        insts = ['1', '2', '3', '4'] 
        nInsts = 4
    else:
        for i in range(nInsts):
            insts[i] = insts[i].replace('SoloHI','')
 
    # |----------------------------------|
    # |--------- Find the files ---------|
    # |----------------------------------|
    
    SolOfiles = [[] for i in range(nInsts)]
    
    for i in range(nInsts):
        SolOfiles[i] = os.listdir(inFolder+insts[i]+'/')
        
    # Make sure we found something before moving on
    nFound = 0
    for i in range(nInsts):
        nFound += len(SolOfiles[i])
    # Return if nothing found
    if nFound == 0:
        print ('No SolO files found')
        return None
        
    # Get the date and time strings
    ymds, hms = setupTimeStuff(times)
    nDays = len(ymds)
    
    
    # |----------------------------------|
    # |--------- Sort the files ---------|
    # |----------------------------------|
    goodFiles = [[] for i in range(nInsts)]
    for i in range(nInsts):
        for aF in SolOfiles[i]:
            for j in range(nDays):
                if ymds[j] in aF:
                    hm = aF[28:32]
                    addIt = True
                    if (j == 0) & (hm < hms[0]):
                        addIt = False
                    elif (j == nDays-1) & (hm > hms[1]):
                        addIt = False
                    if addIt:
                        goodFiles[i].append(aF)
                                    
    # Make sure we found something in the date range before moving on
    nFound = 0
    for i in range(nInsts):
        nFound += len(goodFiles[i])
    # Return if nothing found
    if nFound == 0:
        print ('No matching SolO files found')
        return None
      
    # Make an array and sorted 
    for i in range(nInsts):
        goodFiles[i] = np.sort(np.array(goodFiles[i]))     
      
    # |----------------------------------|
    # |---------- Mosaic files ----------|
    # |----------------------------------|   
    # To make mosaics we need to pass it four images with (as close as possible)
    # matching time stamps. Figure out the max resolution quadrant and just find
    # the closest friend for each of the other quadrants
    if doMosaic:
        # |----------------------------------|
        # |-------- Match the files ---------|
        # |----------------------------------|
        myTimes = [[], [], [], []]
        prefixs = ['solo_L2_solohi-1ft_', 'solo_L2_solohi-2ft_', 'solo_L2_solohi-3fg_', 'solo_L2_solohi-4fg_']
        for i in range(4):
            for j in range(len(goodFiles[i])):
                thisT = parse_time(goodFiles[i][j].replace(prefixs[i],'').replace('_V01.fits','').replace('_V02.fits',''))
                myTimes[i].append(thisT)
        # Figure out which has the most obs
        nEach =[len(myTimes[i]) for i in range(4)]
        nMost = np.where(nEach == np.max(nEach))[0][0]
        nQuads = nEach[nMost] # total number of quad images
        quadFiles = [[] for i in range(4)]
            
        # Find the closest time match for each panel
        # This will duplicate panels as needed if nEach < nQuads
        for i in range(4):
            if i == nMost:
                quadFiles[i] = goodFiles[i]
            else:
                quadFiles[i] = np.copy(goodFiles[nMost])
                for j in range(nQuads):
                    maxDiff = 9999
                    for k in range(nEach[i]):
                        nowDiff = np.abs(myTimes[i][k] - myTimes[nMost][j])
                        if nowDiff < maxDiff:
                            quadFiles[i][j] = goodFiles[i][k]
                            maxDiff = nowDiff
                quadFiles[i] = np.array(quadFiles[i])
        quadFiles = np.transpose(quadFiles)
        
        # Add in the full path for each file
        # Doesn't like just updating quadFiles so make new array
        fullQuadFiles = [[] for i in range(nQuads)]
        for j in range(nQuads):
            fullLine = []
            for i in range(4):    
                fullLine.append(inFolder+insts[i]+'/'+ quadFiles[j][i])
            fullQuadFiles[j] = fullLine
         
        # |----------------------------------|
        # |------- Process the files --------|
        # |----------------------------------|    
        # Run the processing
        ims = []
        hdrs = []
        print ('|--- Processing SoloHI ---|')
        proIms = {}
        proIms['SoloHI'] = [[], []]
        outLines = {}   
        outLines['SoloHI'] = []
        for i in range(nQuads):
            im, hdr = solohi_fits2grid(fullQuadFiles[i])
            ims.append(im)
            hdrs.append(hdr)
            proIms['SoloHI'][0].append(im)
            proIms['SoloHI'][1].append(hdr)
            outLines['SoloHI'].append(fullQuadFiles[i])

        # Save the results
        if len(ims) > 0:
            if saveFits:
                #outLines.append('SolOHI_Quad \n')
                if not os.path.exists(outFolder+'Mosaic/'):
                    # Make if if it doesn't exist
                    os.mkdir(outFolder+'Mosaic/')
                    for i in range(len(ims)):
                        print('Processing image',i,'of', len(ims))
                        ymd = hdrs[i]['DATE-OBS'].replace('-','').replace(':','')[:15]  
                        fitsName = 'wbpro_solohiquad_'+ymd+'.fits'      
                        fullName = outFolder+'Mosaic/' + fitsName
                        fits.writeto(fullName, ims[i], hdrs[i], overwrite=True)
                        #outLines.append(fullName+'\n')
                
    else:
        outLines = {}
        proIms = {}
        for i in range(nInsts):
            if len(goodFiles[i]) > 0:
                proIms['SoloHI'+insts[i]] = [[], []]
                outLines['SoloHI'+insts[i]] = []
                for j in range(len(goodFiles[i])):
                    aF = inFolder+insts[i]+'/'+goodFiles[i][j]
                    with fits.open(aF) as hdulist:
                        im  = hdulist[0].data
                        hdr = hdulist[0].header
                    proIms['SoloHI'+insts[i]][0].append(im)
                    proIms['SoloHI'+insts[i]][1].append(hdr)
                    outLines['SoloHI'+insts[i]].append(inFolder+insts[i]+'/'+goodFiles[i][j])
                    
                    # Don't bother saving these bc unmodified?
                                  
                #outLines.append('SolOHI_'+str(i+1)+'\n')
                #for j in range(len(goodFiles[i])):
                #    outLines.append(inFolder+insts[i]+'/'+goodFiles[i][j]+'\n')

    return proIms, outLines
 

# |------------------------------------------------------------|
# |-------------- Process STEREO Observations -----------------|
# |------------------------------------------------------------|
def processSTEREO(times, insts, inFolder='pullFolder/STEREO/', outFolder='wbFits/STEREO/', downSize=1024, prepDir='prepFiles/stereo/', saveFits=False):
    """
    Function to process STEREO observations using secchi_prep
    
    This is a wrapper to find the data and pull the ported versions of IDL solarsoft
    routines. The resulting fits files are in total brightness and a near exact match
    to using the original IDL procedures. These files can be passed directly to the
    wombat mass calculation procedure

    Inputs:
        times: an array with [startTime, endTime] where the times are strings
               that can be interpreted by sunpy parse_time
        
        insts: strings for selected instruments (EUVI# where # in [171, 195, 284, 304]
                COR1, COR2, HI1, HI2)
                *** will automatically pull both A/B as available ***
    
    Optional Inputs:
        inFolder: top folder for unprocessed results, will open from outFolder/inst/
                  defaults to pullFolder/SOHO/LASCO/
               
        saveFits: flag whether to save the processed results as new fits files or not
                  defaults to False since wombat intends to pass the results to the wrapper
                  that organizes a pickle file to quick load the GUI

        outFolder: top folder for processed results, will be save in outFolder/SOHO/inst/ 
                   defaults to wbFits/SOHO/LASCO/
    
        downSize: maximum resolution to save the processed fits files 
    
        gtFileIn: save file (ecchi_gtdbase.geny) needed for EUVI processing in scc_funs/scc_gt2sunvec
                  defaults to prepFiles/stereo/secchi_gtdbase.geny
        
    Outputs:
        proIms:     a dictionary with the instrument as key and the entry corresponding to
                    [[im1, im2, im3,...], [hdr1, hdr2, hdr3,...]] for the processed data
   
        outLines:   a dictionary of paths to the original fits files used to make processed results
                    for polarized triplets it will have an array of 3 images for each time step
    
    Actions:
        If saveFits is set the processed fits files within the appropriate location in the
        outFolder substructure.   
    
    """
    # |----------------------------------|
    # |--------- Find the files ---------|
    # |----------------------------------|
    nInsts = len(insts)
    STEREOfiles = [[[], []] for i in range(nInsts)]
    
    AB = ['A', 'B']
    ABtoDo = [range(2) for i in range(nInsts)]
    # Check if passed specifically A or B in each inst
    # and change check list but rm tag from inst name 
    # bc originally coded to just search for both
    for i in range(nInsts):
        if 'A' in insts[i]:
            ABtoDo[i] = [0]
            insts[i] = insts[i].replace('A','')
        elif 'B' in insts[i]:
            ABtoDo[i] = [1]
            insts[i] = insts[i].replace('B','')
    
    for i in range(nInsts):
        inst = insts[i]
        for j in ABtoDo[i]:
            if 'EUVI' in inst:
                myFold = inFolder+inst.replace('EUVI', 'EUVI'+AB[j]+'/')
            else:
                myFold = inFolder+inst+AB[j]
            STEREOfiles[i][j] =  os.listdir(myFold)
    
    # Make sure we found something before moving on
    nFound = 0
    for i in range(nInsts):
        nFound += len(STEREOfiles[i])
    # Return if nothing found
    if nFound == 0:
        print ('No STEREO files found')
        return None
        
    # Get the date and time strings
    ymds, hms = setupTimeStuff(times)
    nDays = len(ymds)
    
    
    # |----------------------------------|
    # |--------- Sort the files ---------|
    # |----------------------------------|
    goodFiles = [[[],[]] for i in range(nInsts)]
    for i in range(nInsts):
        inst = insts[i]
        # Loop through A/B 
        for k in ABtoDo[i]:
            if 'EUVI' in inst:
                myFold = inFolder+inst.replace('EUVI', 'EUVI'+AB[k]+'/')
            else:
                myFold = inFolder+inst+AB[k]
            for aF in STEREOfiles[i][k]:
                for j in range(nDays):
                    if (ymds[j] in aF):
                        # Check if EUVI which has two _ before date
                        if aF[0] == 'E':
                            stidx = 9
                        # Otherwise COR/HI have single _
                        else:
                            stidx = aF.find('_')
                        hm = aF[stidx+10:stidx+14]
                        addIt = True
                        if (j == 0) & (hm < hms[0]):
                            addIt = False
                        elif (j == nDays-1) & (hm > hms[1]):
                            addIt = False
                        if addIt:
                            goodFiles[i][k].append(myFold+'/'+aF)
                                    
    # Make sure we found something in the date range before moving on
    nFound = 0
    for i in range(nInsts):
        for j in ABtoDo[i]:
            nFound += len(goodFiles[i][j])
    # Return if nothing found
    if nFound == 0:
        print ('No matching STEREO files found')
        return None
    
    # Make an array and sorted 
    for i in range(nInsts):
        for j in ABtoDo[i]:
            goodFiles[i][j] = np.sort(np.array(goodFiles[i][j]))
    
    # |----------------------------------|
    # |------- Process the files --------|
    # |----------------------------------|                    
    outLines = {}   
    proIms = {}
    fronts = ['STA_', 'STB_']
    fronts2 = ['wbpro_sta', 'wbpro_stb']
    # Add in the actual processing, saving, and output to runFile  
    

    print ('|---- Processing STEREO ----|')
    for i in range(nInsts):
        inst = insts[i]
        # Anyone thats not a pB triplet
        if inst != 'COR1':
            for j in ABtoDo[i]:
                proIms[inst+AB[j]] = [[], []]
                outLines[inst+AB[j]] = []
                if len(goodFiles[i][j]) > 0:
                    print ('|--- Processing STEREO '+inst+' '+AB[j]+' ---|')                   
                    #outLines.append(fronts[j]+inst.replace('EUVI','EUVI_')+'\n')
                    ims, hdrs = secchi_prep(goodFiles[i][j], outSize=[downSize, downSize], prepDir=prepDir) 
                    for k in range(len(ims)):
                        if saveFits:
                            ymd = hdrs[k]['DATE-OBS'].replace('-','').replace(':','')[:15]  
                            fitsName = fronts2[j]+inst.lower()+'_'+ymd+'.fits' 
                            if 'EUVI' in inst:
                                nowFold = outFolder+inst.replace('EUVI', 'EUVI'+AB[j]+'/')
                            else:
                                nowFold = outFolder+inst+AB[j] 
                            fullName = nowFold + '/' + fitsName   
                            fits.writeto(fullName, ims[k], hdrs[k], overwrite=True)
                            #outLines.append(fullName+'\n')
                        proIms[inst+AB[j]][0].append(ims[k])
                        proIms[inst+AB[j]][1].append(hdrs[k])
                        outLines[inst+AB[j]].append(goodFiles[i][j])
                        

        # Process the triplets
        else:
            for j in ABtoDo[i]:
                myCOR = 'COR1'+AB[j]
                proIms[inst+AB[j]] = [[], []]
                outLines[inst+AB[j]] = []
                # Need to check on making triples, missing files will cause isses
                fileYMDs = []
                for aF in goodFiles[i][j]:
                    pref = inFolder+'COR1'+AB[j]+'/COR1'+AB[j]+'_'
                    fileYMDs.append(aF.replace(pref, '')[:15][:-2]) # this is yyyymmdd_hhmm
                fileYMDs = np.array(fileYMDs)
                
                haveDone = []
                goodTrips = []
                for kk in range(len(fileYMDs)):
                    if kk not in haveDone:
                        # trips should only differ in seconds
                        noSecs = fileYMDs[kk]
                        matches = np.where(fileYMDs == noSecs)[0]
                        if len(matches) == 3:
                            myTrip = []
                            for iii in range(3):
                                haveDone.append(matches[iii])
                                myTrip.append(goodFiles[i][j][matches[iii]])
                            goodTrips.append(myTrip)

                
                if len(goodTrips) > 0:
                    ims, hdrs = [], []
                    print ('|---- Processing '+ str(len(goodTrips)) +' triplets for COR1'+AB[j]+' ----|')
                    #outLines.append(fronts[j]+inst+'\n')
                    for k in range(len(goodTrips)):
                        print ('      on triplet ' +str(1+k))
                        aIm, aHdr = secchi_prep(goodTrips[k], polarizeOn=True, silent=True, prepDir=prepDir)
                        ims.append(aIm[0])
                        hdrs.append(aHdr[0])
                        if saveFits:
                            ymd = hdrs[k]['DATE-OBS'].replace('-','').replace(':','')[:15]  
                            fitsName = fronts2[j]+inst.lower()+'_'+ymd+'.fits' 
                            nowFold = outFolder+inst+AB[j] 
                            fullName = nowFold + '/' + fitsName   
                            fits.writeto(fullName, ims[k], hdrs[k], overwrite=True)
                            #outLines.append(fullName+'\n')
                        proIms[inst+AB[j]][0].append(ims[k])
                        proIms[inst+AB[j]][1].append(hdrs[k])
                        outLines[inst+AB[j]].append(goodTrips[k])
                        
    return proIms, outLines
    

# |------------------------------------------------------------|
# |--------------- Process WISPR Observations -----------------|
# |------------------------------------------------------------|
def processWISPR(times, insts, wcalpath='prepFiles/psp/wispr/',  inFolder='pullFolder/PSP/WISPR/', outFolder='wbFits/PSP/WISPR/', downSize=1024, saveFits=False):
    """
    Function to process the level 2 WISPR data
    
    This is a wrapper to find the data and pull the ported versions of IDL solarsoft
    routines. The resulting fits files are in total brightness and a near exact match
    to using the original IDL procedures. These files can be passed directly to the
    wombat mass calculation procedure

    Inputs:
        times: an array with [startTime, endTime] where the times are strings
               that can be interpreted by sunpy parse_time
        
        insts: strings for selected instruments (Mosaic, SoloHI# where # in 1-4)
    
    Optional Inputs:
        wcalpath: folder where the WISPR calibration files are stored
                  defaults to prepFiles/psp/wispr/
    
        inFolder: top folder for unprocessed results, will open from outFolder/#/
                  defaults to pullFolder/PSP/WISPR/
               
        saveFits: flag whether to save the processed results as new fits files or not
                  defaults to False since wombat intends to pass the results to the wrapper
                  that organizes a pickle file to quick load the GUI

        outFolder: top folder for processed results, will be save in outFolder/inst/
                   defaults to wbFits/PSP/WISPR/
    
    Outputs:
        proIms:     a dictionary with the instrument as key and the entry corresponding to
                    [[im1, im2, im3,...], [hdr1, hdr2, hdr3,...]] for the processed data
   
        outLines:   a dictionary of paths to the original fits files used to make processed results
    
    Actions:
        If saveFits is set the processed fits files within the appropriate location in the
        outFolder substructure.   
    
    """
    # |----------------------------------|
    # |--------- Find the files ---------|
    # |----------------------------------|
    nInsts = len(insts)
    PSPfiles = [[] for i in range(nInsts)]
    for i in range(nInsts):
        PSPfiles[i] = os.listdir(inFolder+insts[i])
    
    # Make sure we found something before moving on
    nFound = 0
    for i in range(nInsts):
        nFound += len(PSPfiles[i])
    # Return if nothing found
    if nFound == 0:
        print ('No WISPR files found')
        return None
        
    # Get the date and time strings
    ymds, hms = setupTimeStuff(times)
    nDays = len(ymds)
    
    
    # |----------------------------------|
    # |--------- Sort the files ---------|
    # |----------------------------------|
    goodFiles = [[] for i in range(nInsts)]
    for i in range(nInsts):
        for aF in PSPfiles[i]:
            for j in range(nDays):
                if (ymds[j] in aF):
                    hm = aF[22:26]
                    addIt = True
                    if (j == 0) & (hm < hms[0]):
                        addIt = False
                    elif (j == nDays-1) & (hm > hms[1]):
                        addIt = False
                    if addIt:
                        goodFiles[i].append(inFolder+insts[i]+'/'+aF)

    # Make sure we found something in the date range before moving on
    nFound = 0
    for i in range(nInsts):
        nFound += len(goodFiles[i])
    # Return if nothing found
    if nFound == 0:
        print ('No matching WISPR files found')
        return None
    
    
    # |----------------------------------|
    # |------- Process the files --------|
    # |----------------------------------|                    
    outLines = {}   
    
    # Add in the actual processing, saving, and output to runFile  
    print ('|---- Processing WISPR ----|')
    proIms = {}
    keySwitch = {'Inner':'WISPRI', 'Outer':'WISPRO'}
    for i in range(nInsts):
        if len(goodFiles[i]) > 0:
            print ('|---- Processing WISPR ' + insts[i] + ' ----|')
            proIms[keySwitch[insts[i]]] = [[], []]
            outLines[keySwitch[insts[i]]] = []
            # Make an array and sort alphabetically = time sorted      
            goodFiles[i] = np.sort(np.array(goodFiles[i]))
            ims, hdrs = wispr_prep(goodFiles[i], wcalpath, straylightOff=True)
            #if saveFits:
            #    outLines.append('WISPR_'+ insts[i] + '\n')
            for j in range(len(ims)):
                proIms[keySwitch[insts[i]]][0].append(ims[j])
                proIms[keySwitch[insts[i]]][1].append(hdrs[j])
                outLines[keySwitch[insts[i]]].append(goodFiles[i][j])
                if saveFits:
                    ymd = hdrs[j]['DATE-OBS'].replace('-','').replace(':','')[:15]  
                    fitsName = 'wbpro_wispr'+insts[i]+'_'+ymd+'.fits'
                    fullName = outFolder+insts[i]+'/' + fitsName
                    fits.writeto(fullName, ims[j], hdrs[j], overwrite=True)
                    #outLines.append(fullName+'\n')
             
    return proIms, outLines



# |------------------------------------------------------------|
# |----------- Main function to process observations ----------|
# |------------------------------------------------------------|
def processObs(times, insts, inFolder='pullFolder/', outFolder='wbFits/', outFile='WBobslist.txt', downsize=1024):
    """
    This wrapper assumes that files are stored in the structure set up
    by wombatPullObs and will search for files in a given time range
    and pass those to the appropriate processing scripts. The process functions
    can be called on their own with a different wrapper if a different folder
    structure is used

    Inputs:
        times: an array with [startTime, endTime] where both
               times are of the format YYYY-MM-DDTHH:MM 
               (or any format that sunpy parse_time likes)
    
        inst: an array with the tags for which data to pull
            Available tags:
                AIAnum  = SDO AIA where num represents a wavelength from [94, 131, 171*, 
                          193*, 211, 304*, 335, 1600, 1700] with * most common
                C2      = LASCO C2
                C3      = LASCO C3
                COR1    = STEREO COR1
                COR2    = STEREO COR2    
                EUVInum = STEREO EUVI where num is a wavelength from [171, 195, 284, 304]
                HI1     = STEREO HI1
                HI2     = STEREO H2
                SoloHI  = All quadrants from Solar Orbiter HI
                SoloHI1 = Quadrant 1 from Solar Orbiter HI
                SoloHI2 = Quadrant 2 from Solar Orbiter HI
                SoloHI3 = Quadrant 3 from Solar Orbiter HI
                SoloHI4 = Quadrant 4 from Solar Orbiter HI
                WISPR   = Both inner and outer from PSP WISPR
                WISPRI  = Inner only from PSP WISPR
                WISRPO  = Outer only from PSP WISPR
                * all STEREO values will pull A and B (as available)
    
    Optional Inputs:
        inFolder:  the top level folder name for where processed data will be saved
                   defaults to pullFolder/ (from wombatPullObs)

        outFolder: the top level folder name for where processed data will be saved
                   defaults to wbFits/ 
    
        outFile:   the name of a file with the names of the processed files formatted
                   so that it can be used to launch the wombat gui
                   defaults to WBobslist.txt
    

    Outputs:
        allProIms:
    
        allfnames:  
    
        The code will set up a folder structure at outFolder with nested satellite
        and instument folders (if it doesn't already exist). It will then dump the 
        processed files into the appropriate folders
    
    External Calls:
        setupFolderStructure from wombatProcessObs
        secchi_prep from secchi_prep
        c2_prep, c3_prep from lasco_prep
        wispr_prep from wispr_prep
        solohi_fits2grid from solohi_prep
        aia_prep from aia_prep
    
    """

    # |---------------------------------------| 
    # |---- Check the top level directory ----|
    # |---------------------------------------| 
    setupFolderStructure(outFolder)
    
    
    # |---------------------------------------| 
    # |---- Check all inst keys are valid ----|
    # |---------------------------------------| 
    goods = np.array(['AIA94', 'AIA131', 'AIA171','AIA193','AIA211','AIA304','AIA335','AIA1600','AIA1700', 'C2', 'C3', 'COR1', 'COR2', 'EUVI171', 'EUVI195', 'EUVI284', 'EUVI304', 'HI1', 'HI2', 'COR1A', 'COR2A', 'EUVI171A', 'EUVI195A', 'EUVI284A', 'EUVI304A', 'HI1A', 'HI2A', 'COR1B', 'COR2B', 'EUVI171B', 'EUVI195B', 'EUVI284B', 'EUVI304B', 'HI1B', 'HI2B' ,'SoloHI', 'SoloHI1', 'SoloHI2', 'SoloHI3', 'SoloHI4', 'WISPR', 'WISPRI', 'WISPRO'])
    quitIt = False
    for inst in insts:
        if inst not in goods:
            print ('Unknown instrument tag', inst)
            quitIt = True
    if quitIt:
        sys.exit('Quitting processObs since passed invalid instrument tag')
        
    # |---------------------------------------| 
    # |--------- Check outFolder name --------|
    # |---------------------------------------|
    if outFolder[-1] != '/':
        outFolder = outFolder+'/'
    
    
    # |-------------------------------|
    # |--- Open up the output file ---|
    # |-------------------------------|
    #f1 = open(outFile, 'w')
    
    allProIms = {}  
    allfnames = {}
            
    # |-----------------------------------------|    
    # |--------- Loop through each sat ---------|   
    # |-----------------------------------------|   
        
    # |-------------------------------|
    # |------------- AIA -------------|
    # |-------------------------------|
    doAIA = []
    for inst in insts:
        if 'AIA' in inst:
            # If found just save the wavelength
            doAIA.append(inst.replace('AIA',''))
    if len(doAIA) > 0:
        proIms, outLines = processAIA(times, doAIA, inFolder='pullFolder/SDO/AIA/')
        for key in proIms:
            allProIms[key] = proIms[key]
            allfnames[key] = outLines[key]
            
        #if type(outLines) != type(None):
        #    for line in outLines:
        #        f1.write(line)
        #else:
        #    print('Unable to process any AIA images')
                
    # |-------------------------------|
    # |------------ SOHO -------------|
    # |-------------------------------|
    doLASCO = []
    if 'C2' in insts: doLASCO.append('C2')
    if 'C3' in insts: doLASCO.append('C3')
    if len(doLASCO) > 0:
        proIms, outLines = processLASCO(times, doLASCO)
        for key in proIms:
            allProIms[key] = proIms[key]
            allfnames[key] = outLines[key]
            
        #if type(outLines) != type(None):
        #    for line in outLines:
        #        f1.write(line)
        #else:
        #    print('Unable to process any LASCO images')
   
    # |-------------------------------|
    # |------------ STEREO -----------|
    # |-------------------------------|
    doSTEREO = []
    STkeys = ['COR1', 'COR2', 'EUVI171', 'EUVI195', 'EUVI284', 'EUVI304', 'HI1', 'HI2', 'COR1A', 'COR2A', 'EUVI171A', 'EUVI195A', 'EUVI284A', 'EUVI304A', 'HI1A', 'HI2A', 'COR1B', 'COR2B', 'EUVI171B', 'EUVI195B', 'EUVI284B', 'EUVI304B', 'HI1B', 'HI2B']
    for inst in insts:
        if inst in STkeys:
            doSTEREO.append(inst)
    if len(doSTEREO) > 0:
        proIms, outLines = processSTEREO(times, doSTEREO)
        for key in proIms:
            # Check for lack of STB
            if len(proIms[key][0]) > 0:
                allProIms[key] = proIms[key]
                allfnames[key] = outLines[key]
            
        #if type(outLines) != type(None):
        #    for line in outLines:
        #        f1.write(line)
        #else:
        #    print('Unable to process any STEREO images')
        
    

    # |-------------------------------|
    # |------------ WISPR ------------|
    # |-------------------------------|
    doWISPR = []
    if 'WISPR' in insts: doWISPR = ['Inner', 'Outer']
    elif 'WISPRI' in insts: doWISPR.append('Inner')
    elif 'WISPRO' in insts: doWISPR.append('Outer')
    if len(doWISPR) > 0:
        proIms, outLines = processWISPR(times, doWISPR)
        for key in proIms:
            allProIms[key] = proIms[key]
            allfnames[key] = outLines[key]
        
        #if type(outLines) != type(None):
        #    for line in outLines:
        #        f1.write(line)
        #else:
        #    print('Unable to process any WISPR images')
    

    # |-------------------------------|
    # |----------- SoloHI ------------|
    # |-------------------------------|
    doSoloHI = []
    for inst in insts:
        if inst == 'SoloHI': 
            doSoloHI = ['Mosaic']
        elif inst == 'SoloHI1': 
            doSoloHI.append('SoloHI1')
        elif inst == 'SoloHI2': 
            doSoloHI.append('SoloHI2')
        elif inst == 'SoloHI3': 
            doSoloHI.append('SoloHI3')
        elif inst == 'SoloHI4': 
            doSoloHI.append('SoloHI4')
    
    
    if len(doSoloHI) > 0:
        proIms, outLines = processSoloHI(times, doSoloHI)
        for key in proIms:
            allProIms[key] = proIms[key]
            allfnames[key] = outLines[key]
            
        #if type(outLines) != type(None):
        #    for line in outLines:
        #        f1.write(line)
        #else:
        #    print('Unable to process any SoloHI images')
        
    # |-------------------------------|
    # |---- Close the output file ----|
    # |-------------------------------|
    #f1.close()
    
    return allProIms, allfnames


# |------------------------------------------------------------|
# |-------------------- Command Line Wrapper ------------------|
# |------------------------------------------------------------|
def thePickler(proIms, fnames, pickleJar='wbPickles/', name='temp'):
    """
    Wrapper to make a pickle with all the info that the GUI will need
    for the background images. Everything will be fully processed pre
    GUI to minimize computation time there
    

    Inputs:        
        Required:
            proIms:   A dictionary of process images that are the results
                      of the processObs function. The key is the instrument
                      tag name and each entry has the form [[im1, im2, ...],
                      [hdr1, hdr2, ...]] where im are arrays.
    
            fnames:   A dictionary of the paths/names of the original files used
                      by processObs to generate proIms. Each time step will have 
                      either a single file, an array of three for STEREO polarized 
                      images or an array of four for SoloHI mosaics
    
        Optional:
            name:        A unique name tag for the pickle, which will be save as WBGUI_name.pkl
                         Defaults to "temp" and will overwrite an existing temp pkl
    
            pickleJar:   Where to save the pickles that are generated
                         (Defaults to /wbPickles/ )        
   
        Actions:
            Saves a pickle in the pickleJar folder as WBGUI_temp.pkl
    
        
    """
    
    # |--- Set up the main pickle dict ---|
    bigDill = {}
    # |--- (sub) Dictionary with useful info ---|
    bigDill['WBinfo'] = {}
    # |--- (sub) Dictionary with standard processed data ---|
    bigDill['proIms0'] = {} # base images
    bigDill['proIms']  = {} # all images 
    bigDill['proImMaps'] = {} # images 1 - end, converted to maps. matches mass/diff indexing
    # |--- (sub) Dictionary with mass image data ---|
    bigDill['massIms'] = {}
    # |--- (sub) Dictionary with difference images scaled for display ---|
    bigDill['scaledIms'] = {}
    # |--- (sub) Dictionary with satStuff -> hdr-like with extra stuff ---|
    bigDill['satStuff'] = {}
    
    # |----------------------------|
    # |---- Fill the info dict ----|
    # |----------------------------|
    # List of instruments
    insts = np.array([key for key in fnames])
    bigDill['WBinfo']['Insts'] = insts
    bigDill['WBinfo']['OrigFiles'] = {}
    bigDill['WBinfo']['isEUV'] = {}
    for key in insts:
        bigDill['WBinfo']['OrigFiles'][key] = fnames[key]
        if ('euvi' in key) or ('aia' in key):
            bigDill['WBinfo']['isEUV'][key] = True
        else:
            bigDill['WBinfo']['isEUV'][key] = False
            
    # Other things to track? Arrays of times?    
    
    # |---------------------------------------|
    # |---- Fill the basic processed data ----|
    # |---------------------------------------|
    tempMaps = {}
    for key in insts:
        if bigDill['WBinfo']['isEUV'][key]:
            bigDill['proIms0'][key] = [None, None]
        else:
            bigDill['proIms0'][key] = [proIms[key][0][0], proIms[key][1][0]]
        
        bigDill['proIms'][key] = [[], []]    
        for i in range(len(proIms[key][0])):
            bigDill['proIms'][key][0].append(proIms[key][0][i])
            bigDill['proIms'][key][1].append(proIms[key][1][i])
        
        # |--------------------------------------|
        # |---- Make running difference maps ----|
        # |--------------------------------------|
        if bigDill['WBinfo']['isEUV'][key]:
            tempMaps[key] = arr2maps(bigDill['proIms'][key][0], bigDill['proIms'][key][1], doDiff=False)           
        else:
            tempMaps[key] = arr2maps(bigDill['proIms'][key][0], bigDill['proIms'][key][1])           
        
        # |---------------------------------------|
        # |---- Process headers into satStuff ----|
        # |---------------------------------------|
        # Have to do at some point and next pieces of code were originally
        # written to use it so easier to do here and pass than rewrite it
        mySatStuff = []
        for j in range(len(tempMaps[key][0])):
            aMap = tempMaps[key][0][j]
            aSStuff =  getSatStuff(aMap) # A Sat Stuff, don't be immature...
            if bigDill['WBinfo']['isEUV'][key]:
                aSStuff['myFits'] = fnames[key][j]
            else:    
                aSStuff['myFits'] = fnames[key][j+1]
                aSStuff['myRDfits'] = fnames[key][j]
                aSStuff['myBDfits'] = fnames[key][0]
                
            mySatStuff.append(aSStuff)
        
        # |------------------------------------|
        # |---- Calculate masses and store ----|
        # |------------------------------------|
        if bigDill['WBinfo']['isEUV'][key]:
            bigDill['massIms'][key] = [None, None]
        else:
            bigDill['massIms'][key] = []
            nims = len(tempMaps[key][0])
            for j in range(nims):
                print('Calculating mass image ', j+1, ' out of ', nims, 'for ', key)
                # Mass calc needs the rsun keyword that not all hdrs initially have
                if 'rsun' not in tempMaps[key][2][j]:
                    if 'RSUN_ARC' in tempMaps[key][2][j]:
                        tempMaps[key][2][j]['rsun'] =  tempMaps[key][2][j]['RSUN_ARC']
                    else:
                        calcRsun = mySatStuff[j]['ONERSUN']*mySatStuff[j]['SCALE']
                        if mySatStuff[j]['OBSTYPE'] == 'HI':
                            calcRsun *= 3600
                        tempMaps[key][2][j]['rsun'] = calcRsun
                   
                massIm, hdrM = wM.TB2mass(tempMaps[key][1][j].data, tempMaps[key][2][j])
                if 'MASK' in mySatStuff[j]:
                    massIm = (1-mySatStuff[j]['MASK']) * massIm
                #bigDill['massIms'][key].append(np.transpose(massIm))
                bigDill['massIms'][key].append(massIm)

        # |-------------------------------------------|
        # |---- Calculate scaled images and store ----|
        # |-------------------------------------------|
        sclIms, satstuff2 = scaleIt(tempMaps[key], mySatStuff)
        bigDill['scaledIms'][key] = sclIms
        bigDill['satStuff'][key] = satstuff2
        
        # |---------------------------------|
        # |---- Make min processed maps ----|
        # |---------------------------------|
        bigDill['proImMaps'][key] = [[], []]
        maxlen = len(bigDill['proIms'][key][0])-1
        if bigDill['WBinfo']['isEUV'][key]: maxlen += 1
        for j in range(maxlen):
            myMap = sunpy.map.Map(bigDill['proIms'][key][0][j], bigDill['proIms'][key][1][j])
            bigDill['proImMaps'][key][0].append(myMap)
            bigDill['proImMaps'][key][1].append(bigDill['proIms'][key][1][j])

    
        
    # |-------------------------|
    # |---- Save the pickle ----|
    # |-------------------------|
    if not os.path.exists(pickleJar):
        # Make if if it doesn't exist
        print ('Creating directory ', pickleJar)
        os.mkdir(pickleJar)
    print('Saving pickle at ', pickleJar+'WBGUI_'+name+'.pkl')
    with open(pickleJar+'WBGUI_'+name+'.pkl', 'wb') as file:
        pickle.dump(bigDill, file, -1)
    
    
# |------------------------------------------------------------|
# |------------ Convert array+hdrs to diff maps ---------------|
# |------------------------------------------------------------|
def arr2maps(dataIn, hdrIn, doDiff=True):
    """
    Function to convert a list with arrays+headers into an array of maps+headers
    
    This will take an array of process data for a single instrument, along with the
    corresponding headers and convert them into maps that will be used by the GUI.
    These are intended to be passed to the scaling function to make them on nice ranges
    for visualization but that portion is done separate on the off chance we may just
    want to mapify and difference but not scale something
    

    Inputs:
        dataIn: an array with a time series of 2D observation data
    
        hdrIn:  the corresponding headers for dataIn
        
    Optional Inputs:
        doDiff: whether calculate differences or just return a map version of the
                input data. Defaults to True but expect it is typically switched to 
                False when EUV data is passed 
           
    Outputs:
        allFH: an array in the form [[RD1, RD2,...], [BD1, BD2,...], [hdr1, hdr2,...]] 
               where RD and BD are astropy/sunpy maps with either the running or base
               difference. Unless doDiff switched off these arrays will be one shorter
               than the input arrays due to taking a difference along the time series.
     
    """

    rds, bds, hdrs = [], [], []
    
    # |--- Get running and base diffs ---|
    if doDiff:
        for j in range(len(dataIn)-1):
            # Get the data for a time step
            myData = dataIn[j+1]
            myHdr  = hdrIn[j+1]
            
            # Get the bases
            runBase  = dataIn[j]
            runFile  = hdrIn[j]
            baseBase = dataIn[0]
            baseFile = hdrIn[0]
            
            # Make sure files are same shape and diff    
            if (myData.shape == runBase.shape) & (myData.shape == baseBase.shape):
                # Run diff
                diffData = myData - runBase
                #myHdr['diffFile'] = runFile
                diffMap = sunpy.map.Map(diffData, myHdr)
                rds.append(diffMap)
                # Base diff
                diffData = myData - baseBase
                #myHdr['diffFile'] = baseFile
                diffMap = sunpy.map.Map(diffData, myHdr)
                bds.append(diffMap)
                # Add hdr
                hdrs.append(myHdr)

            else:
                print('Size mismatch for ' +names[i] + allFH0[i][1][j+1]['DATE-OBS'])
                
    # |--- No diff, just mapify and restructure ---|
    else:
        for j in range(len(dataIn)):
            myData = dataIn[j]
            myHdr  = hdrIn[j]
            myMap  = sunpy.map.Map(myData, myHdr)
            rds.append(myMap)
            bds.append(myMap)
            hdrs.append(myHdr)
    return [rds, bds, hdrs]
    
    
# |------------------------------------------------------------|
# |----------------- Setup up satStuff dicts ------------------|
# |------------------------------------------------------------|
def getSatStuff(imMap):
    """
    Function to make a header like structure with keywords and 
    values but specific to what wombat desires
    
    Inputs:
        imMap:     a single observation map
           
    Outputs:
        satDict:   a dictionary with useful satellite information
                   the keys are as follows:
                        OBS:       observatory/satellite name
                        INST:      instrument name
                        MYTAG:     nice name string with obs+inst
                        OBSTYPE:   type of observation (EUV, COR, HI)
                        WAVE:      wavelength in angstroms (only for EUV)
                        SHORTNAME: shorter version of MYTAG
                        DATEOBS:   string for the date as YYYYMMDDTHH:MM:SS
                        POS:       postion of sat [lat, lon, R] in [deg, deg, m]
                        POINTING:  unit vector pointing from sat to sun center (in Stony Cart)
                        POSLON:    longitudes of the plane of sky (in equitorial plane, Stony deg)
                        SCALE:     plate scale in arcsec/pix (or deg/pix for HI)
                        CRPIX:     pixel location of the reference pixel
                        WCS:       wcs structure
                        SUNPIX:    pixel location of the center of the sun
                        ONERSUN:   one solar radius in pixels
                        FOV:       maximum radial distance of a corner in Rs
                        MASK:      array with masked pixels set to 1 (0 elsewhere)
                        OCCRPIX:   radius of occulter in pixels
                        OCCRARC:   radius of occulter in arcsecs
                        SUNCIRC:   array with the xy pixels for an outline of the sun
                        SUNNORTH:  array with xy pixels to indicate solar north direction
                        MYFITS:    the name/path of the original fits file
                        DIFFFITS:  the name/path of the file used to calc a difference img
    
    External Calls:
        fitshead2wcs from wcs_funs                       
    
    """
    
    # Nominal radii (in Rs) for the occulters for each instrument. Pulled from google so 
    # generally correct (hopefully) but not the most precise
    occultDict = {'STEREO_SECCHI_COR2':[3,14], 'STEREO_SECCHI_COR1':[1.5,4], 'SOHO_LASCO_C1':[1.1,3], 'SOHO_LASCO_C2':[2,6], 'SOHO_LASCO_C3':[3.7,32], 'STEREO_SECCHI_HI1':[15,80], 'STEREO_SECCHI_HI2':[80,215], 'STEREO_SECCHI_EUVI':[0,1.7],'SDO_AIA':[0,1.35]}
    
    #|---- Initialize dictionary ----|
    satDict = {}
    
    # |-----------------|
    # |---- Get OBS ----|
    # |-----------------|
    # Pull the hdr/map metadata
    myhdr   = imMap.meta   
    satDict['OBS'] =  myhdr['obsrvtry']
   
    # |------------------|
    # |---- Get INST ----|
    # |------------------|
    # All the satellites have different options on saving
    # instrument names vs detector names. Combine into a single
    # INST tag for wombat
    # PSP format
    if myhdr['obsrvtry'] == 'Parker Solar Probe':
        satDict['OBS'] =  myhdr['obsrvtry']
        satDict['INST'] =  myhdr['instrume'] + '_HI' + str(myhdr['detector'])
        myTag   = myhdr['obsrvtry'] + '_' + myhdr['instrume'] + '_HI' + str(myhdr['detector'])
    # SolO format
    elif myhdr['obsrvtry'] == 'Solar Orbiter':
        satDict['OBS'] =  myhdr['obsrvtry']
        satDict['INST'] = myhdr['instrume'] 
        myTag   = myhdr['obsrvtry'] + '_' + myhdr['instrume']
    elif myhdr['telescop'] == 'STEREO':
        satDict['OBS'] =  myhdr['obsrvtry'] 
        satDict['INST'] = myhdr['instrume'] + '_' + myhdr['detector']
        myTag   = myhdr['telescop'] + '_' + myhdr['instrume'] + '_' + myhdr['detector']
    elif myhdr['obsrvtry'] == 'SDO':
        satDict['OBS'] =  myhdr['obsrvtry'] 
        satDict['INST'] = myhdr['detector']
        myTag   = myhdr['obsrvtry'] + '_' + myhdr['detector']
    # Other less picky sats
    else:
        satDict['OBS'] =  myhdr['telescop']
        satDict['INST'] = myhdr['instrume'] + '_' + myhdr['detector']
        myTag   = myhdr['telescop'] + '_' + myhdr['instrume'] + '_' + myhdr['detector']
    satDict['MYTAG'] = myTag
     
    # |---------------------|
    # |---- Get OBSTYPE ----|    
    # |---------------------|
    # Flag between HI, COR, EUV
    if satDict['OBS'] in ['Parker Solar Probe', 'Solar Orbiter']:
        satDict['OBSTYPE'] = 'HI'
    elif satDict['OBS'] in ['STEREO_A', 'STEREO_B']:
        if myhdr['detector'] in ['COR1', 'COR2']:
            satDict['OBSTYPE'] = 'COR'
        elif myhdr['detector'] in ['HI1', 'HI2']:
            satDict['OBSTYPE'] = 'HI'
        else:
            satDict['OBSTYPE'] = 'EUV'
    elif satDict['OBS'] == 'SOHO':
        if myhdr['detector'] in ['C2', 'C3']:
            satDict['OBSTYPE'] = 'COR'
        else:
            satDict['OBSTYPE'] = 'EUV'
    elif satDict['OBS'] == 'SDO':
         satDict['OBSTYPE'] = 'EUV'
    
    # |------------------|
    # |---- Get WAVE ----|   
    # |------------------|
    # Add the wavelength if EUV
    if satDict['OBSTYPE'] == 'EUV':
        satDict['WAVE'] = str(myhdr['WAVELNTH'])
        satDict['MYTAG'] = satDict['MYTAG'] + '_' + satDict['WAVE']
            
    # |-----------------------|
    # |---- Get SHORTNAME ----|    
    # |-----------------------|
    shortNames = {'Parker Solar Probe':'PSP', 'Solar Orbiter':'SolO', 'STEREO_A':'STA', 'STEREO_B':'STB', 'SOHO':'SOHO', 'SDO':'SDO'}
    satDict['SHORTNAME'] = shortNames[satDict['OBS']]
    
    
    # |---------------------|
    # |---- Get DATEOBS ----|    
    # |---------------------|
    if len(myhdr['date-obs']) > 13:
        satDict['DATEOBS'] = myhdr['date-obs']
    else:
        satDict['DATEOBS'] = myhdr['date-obs'] + 'T' + myhdr['time-obs']
    satDict['DATEOBS'] = satDict['DATEOBS'].replace('/','-')
    if '.' in satDict['DATEOBS']:
        dotidx = satDict['DATEOBS'].find('.')
        satDict['DATEOBS'] = satDict['DATEOBS'][:dotidx]
     
    # |-----------------|
    # |---- Get POS ----|    
    # |-----------------|
    # Get satellite position
    obsLon = imMap.observer_coordinate.lon.degree
    obsLat = imMap.observer_coordinate.lat.degree
    obsR = imMap.observer_coordinate.radius.m
    satDict['POS'] = [obsLat, obsLon,  obsR]
    # Get sat to sun direction
    latd = obsLat * np.pi / 180.
    lond = obsLon * np.pi / 180.
    xyz = [np.cos(latd)*np.cos(lond), np.cos(latd)*np.sin(lond), np.sin(latd)]
    satDict['POINTING'] = -np.array(xyz)
    pointLon = np.arctan2(satDict['POINTING'][1], satDict['POINTING'][0]) * 180 / np.pi
    PoSlon1 = (pointLon - 90) % 360
    PoSlon2 = (pointLon + 90) % 360
    satDict['POSLON'] = [PoSlon1, PoSlon2]
    
    # |-------------------|
    # |---- Get SCALE ----|    
    # |-------------------|
    # Plate scale in arcsec/pix for EUV/COR
    # or deg/pix for HI
    # Check to make sure same in x/y since we will assume as much
    if (imMap.scale[0].to_value() != imMap.scale[1].to_value()):
        sys.exit('xy scales not equilent. Not set up to handle this. Exiting from getSatStuff')    
    obsScl  = imMap.scale[0].to_value()
    satDict['SCALE'] = obsScl
    
    # |-------------------|
    # |---- Get CRPIX ----|    
    # |-------------------|
    # Reference pixel
    cx,cy = int(myhdr['crpix1'])-1, int(myhdr['crpix2'])-1
    satDict['CRPIX'] = [cx, cy]
    
    # |-----------------|
    # |---- Get WCS ----|    
    # |-----------------|
    myWCS = fitshead2wcs(myhdr)
    satDict['WCS'] = myWCS
    
    # |--------------------|
    # |---- Get SUNPIX ----|    
    # |--------------------|
    centS = wcs_get_pixel(myWCS, [0.,0.])
    sx, sy = centS[0], centS[1]
    satDict['SUNPIX'] = [sx, sy]
    
    # |---------------------|
    # |---- Get ONERSUN ----|    
    # |---------------------|
    # Get 1 Rs in pix
    if 'rsun' in imMap.meta:
        myRs = imMap.meta['rsun'] # in arcsec
    else:
        myDist = imMap.observer_coordinate.radius.m / 7e8
        myRs   = np.arctan2(1, myDist) * 206265
    oners = myRs/imMap.scale[0].to_value()
    if imMap.scale[0].unit == 'deg / pix':
        oners = oners / 3600
    satDict['ONERSUN'] = oners
    
    # |-----------------|
    # |---- Get FOV ----|    
    # |-----------------|
    # Get maximum radial distance of the corners
    myFOV = 0
    for i in [0,imMap.data.shape[0]-1]:
        for j in [0,imMap.data.shape[1]-1]:
            coord = wcs_get_coord(myWCS, pixels = np.array([i,j]))
            edgeR = np.sqrt(coord[0]**2 + coord[1]**2)
            thisFOV = edgeR / obsScl / oners
            if thisFOV > myFOV: myFOV = thisFOV
    satDict['FOV'] = myFOV
    
    # |---------------------------|
    # |---- Add Occulter Mask ----|    
    # |---------------------------|
    # Make mask array
    mask = np.zeros(imMap.data.shape)
    # Check that not SolO/PSP or STEREO HI
    if (satDict['OBSTYPE'] != 'HI'):   
        myOccR  = occultDict[myTag][0] # radius of the occulter in Rs
        occRpix = int(myOccR * oners)
        # Add radius of occulter in pix and arcsecs
        satDict['OCCRPIX'] = myOccR * oners
        satDict['OCCRARC'] = myOccR * oners * imMap.scale[0].to_value()
        
         # Fill in a circle around the occulter center
        for i in range(occRpix):
            j = int(np.sqrt(occRpix**2 - i**2))
            lowY = np.max([0,cy-j])
            hiY  = np.min([imMap.meta['naxis2']-2, cy+j])
            if cx+i <= imMap.meta['naxis2']-1:
                mask[cx+i, lowY:hiY+1] = 1
            if cx-i >=0:
                mask[cx-i, lowY:hiY+1] = 1    
    
        # Fill in outside FoV
        outRpix = int(occultDict[myTag][1] * oners) 
        for i in range(imMap.meta['naxis1']):
            myHdist = np.abs(cx-i)
            if myHdist >= outRpix:
                mask[i,:] = 1
            else:
                possY = int(np.sqrt(outRpix**2 - myHdist**2))
                lowY = np.max([0,cy - possY])
                hiY  = np.min([imMap.meta['naxis2'],cy + possY])
                mask[i,:lowY+1] = 1
                mask[i,hiY:] = 1
                
        # Add to dict
        satDict['MASK'] = mask
        
        # |-------------------------|
        # |---- Add Sun Outline ----|    
        # |-------------------------|
        thetas = np.linspace(0, 2.1*3.14159,100)
        xs = oners * np.cos(thetas) + sx
        ys = oners * np.sin(thetas) + sy
        satDict['SUNCIRC'] = [xs, ys]
        
        # |--------------------------|
        # |---- Add Solar N Line ----|    
        # |--------------------------|
        # Can use slow skycoord/map version bc only call once
        skyPt = SkyCoord(x=0, y=0, z=1, unit='R_sun', representation_type='cartesian', frame='heliographic_stonyhurst')
        myPt2 = imMap.world_to_pixel(skyPt)
        satDict['SUNNORTH'] = [[sx, myPt2[0].to_value()], [sy, myPt2[1].to_value()]]

    # |----------------------------|
    # |---- Add fits file/path ----|    
    # |----------------------------|
    #satDict['MYFITS'] = myhdr['myFits']
    #satDict['DIFFFITS'] = myhdr['diffFile']

    return satDict

# |------------------------------------------------------------|
# |---------------- Scale the Background Imgs -----------------|
# |------------------------------------------------------------|
def scaleIt(obsIn, satStuffs):
    """
    Function to convert input maps into scaled arrays with values
    between 0-255 that are ready to show as is in the plot windows.
    We process this all ahead of time so the GUI flips through 
    existing data without needing new calculations

    Inputs:
        obsIn:     The observations from a single instrument in the form
                   [[map1, map2, ...], [hdr1, hdr2, ...]]
    
        satStuffs: the header like structure created by getSatStuff.
           
    Outputs:
        allScls:   an array of three times series the data scaled using diffent
                   methods (linear, logarithmic, square root). This data is in 
                   array form, not maps.
                   e.g. [[lin1, log1, sqrt1], [lin2, log2, sqrt2], ...]
    
        satStuffs: the header like structure created by getSatStuff 
                   but with a few additional entries
    
    """
    #|-------------------------------------| 
    #|---- Configuration Dictionaries -----|
    #|-------------------------------------|
    # Dictionaries that establish the scaling of things
    # Pull the desired values for each instrument
    
    # mins/maxs on percentiles by instrument [[lower], [upper]] with [lin, log, sqrt]
    pMMs = {'AIA':[[0.001,10,1], [99,99,99]], 'SECCHI_EUVI':[[0.001,10,1], [99,99,99]], 'LASCO_C2':[[15,1,15], [97,99,97]], 'LASCO_C3':[[40,1,10], [99,99,90]], 'SECCHI_COR1':[[30,1,10], [99,99,90]], 'SECCHI_COR2':[[20,1,10], [92,99,93]], 'SECCHI_HI1':[[1,40,1], [99.9,80,99.9]], 'SECCHI_HI2':[[1,40,1],[99.9,80,99.9]], 'WISPR_HI1':[[1,40,1], [99.9,80,99.9]], 'WISPR_HI2':[[1,40,1], [99.9,80,99.9]], 'SoloHI':[[1,40,1], [99.5,80,99.5]] }
    
    # Where the background sliders start (between 0 and 255)
    sliVals = {'AIA':[[0,0,0], [191,191,191]], 'SECCHI_EUVI':[[0,32,0], [191,191,191]], 'LASCO_C2':[[0,0,21],[191,191,191]], 'LASCO_C3':[[37,0,37],[191,191,191]], 'SECCHI_COR1':[[63,0,21],[191,191,191]], 'SECCHI_COR2':[[63,0,21],[191,191,191]], 'SECCHI_HI1':[[63,0,21],[128,191,191]], 'SECCHI_HI2':[[63,0,21],[128,191,191]],  'WISPR_HI1':[[0,0,21],[128,191,191]], 'WISPR_HI2':[[0,0,21],[128,191,191]], 'SoloHI':[[0,0,21],[128,191,191]]}
    
    # Pull the configuration based on instrument
    myInst = satStuffs[0]['INST']
    myMM = pMMs[myInst]
    mySliVals = sliVals[myInst]
    
    #|--- Loop through both RD and BD ---|
    bothScls = []
    bothSatStuffs = []   
     
    for k in range(2):
        #|-------------------------------------| 
        #|------- Pull/Clean Map data ---------|
        #|-------------------------------------|
        #|---- Make empty holder ----|
        sz = obsIn[k][0].data.shape
        allObs = np.zeros([len(satStuffs), sz[0], sz[1]])
        #|---- Fill from map data ----|
        for i in range(len(satStuffs)):
            #allObs[i,:,:] = np.transpose(obsIn[k][i].data)
            allObs[i,:,:] = obsIn[k][i].data
        
        #|---- Get overall median ----|
        imNonNaN = allObs[~np.isnan(allObs)]   
        medval  = np.median(np.abs(imNonNaN))
    
        #|---- Check if diff image ----|
        # Get the median negative value to comp to the median abs
        # value. If neg med big enough assume that is diff image
        negmed  = np.abs(np.median(imNonNaN[np.where(imNonNaN < 0)]))
        diffImg = False
        if (negmed / medval) > 0.25: # guessing at cutoff of 25%, might tune
            diffImg = True
    
        #|---- Clean out NaNs ----|
        if not diffImg:
            allObs[np.isnan(allObs)] = 0
        else:
            allObs[np.isnan(allObs)] = -9999
    
        #|---- Clean out Infs ----| 
        if not diffImg:
            allObs[np.isinf(np.abs(allObs))] = 0
            imNonNaN[np.isinf(np.abs(imNonNaN))] = 0
        else:
            allObs[np.isinf(np.abs(allObs))] = -9999
            imNonNaN[np.isinf(np.abs(imNonNaN))] = -9999
    
        #|-------------------------------------| 
        #|--------- Process the data ----------|
        #|-------------------------------------|    
        #|---- Scaled image holder ----|   
        allScls = []    
    
        #|---- Process linear imgs ----|   
        # Get vals at min/max percentile from the config dictionary
        linMin, linMax = np.percentile(imNonNaN, myMM[0][0]), np.percentile(imNonNaN, myMM[1][0])   
        # If a diff image reset min to neg val based on max   
        if diffImg:
            linMin = - 0.5*linMax
        # Calc range and scale to 0 - 255
        rng = linMax- linMin
        linIm = (allObs - linMin) * 255 / rng

        #|---- Process log imgs ----|   
        # Normalize to keep things in nice ranges
        tempIm = allObs / medval
        tempNonNan = imNonNaN / medval
        # Get min val based on config dict
        minVal = np.percentile(np.abs(tempNonNan),myMM[0][1])
        # Separate into positive and negative values
        pidx = np.where(tempIm > minVal)
        nidx = np.where(tempIm < -minVal)
        # Make new img
        logIm = np.zeros(tempIm.shape)
        # Set where abs val < min to 1
        logIm[np.where(np.abs(tempIm) < minVal)] = 1
        # Positive is just log + 1
        logIm[pidx] = np.log(tempIm[pidx] - minVal + 1)  
        # Negative is -log(abs) + 1
        logIm[nidx] = -np.log(-tempIm[nidx] - minVal + 1)  
        # Get max val from config dict and rescale
        percX = np.percentile(logIm, myMM[1][1])
        logIm = 191 * logIm / percX
    
        #|---- Process sqrt imgs ----|   
        # Normalize to keep things in nice ranges
        tempIm = allObs / medval
        # Get min val based on config dict
        minVal = np.percentile(tempNonNan,myMM[0][2])
        # Set min val to zero
        tempIm = tempIm - minVal 
        # Set all neg to zero
        tempIm[np.where(tempIm < 0)] = 0
        # Sqrt now that everyone is positive
        sqrtIm = np.sqrt(tempIm)
        # Get max val from config dict and rescale
        percX = np.percentile(sqrtIm, myMM[1][2])
        sqrtIm = 191 * sqrtIm / percX
    
    
        #|-------------------------------------| 
        #|--------- Package Results -----------|
        #|-------------------------------------|
        for i in range(len(satStuffs)):
            #|---- Add the slider init values to satStuff ----|
            satStuffs[i]['SLIVALS'] = mySliVals
    
            #|---- Package this time step ----|
            sclIms = [linIm[i], logIm[i],  sqrtIm[i]]
        
            #|---- Add a mask (if needed) ----|
            if 'MASK' in satStuffs[i]:
                midx = np.where(satStuffs[i]['MASK'] == 1)
                for k in range(3):
                    # black out all occulted
                    sclIms[k][midx] = -100. # might need to change if adjust plot ranges
                
            #|---- Append masked/scaled images to out list ----|
            allScls.append(sclIms)
        bothScls.append(allScls)
        bothSatStuffs.append(satStuffs)
    return bothScls, bothSatStuffs


# |------------------------------------------------------------|
# |-------------------- Command Line Wrapper ------------------|
# |------------------------------------------------------------|
def commandLineWrapper():
    """
    Wrapper to only be used to make wombatProcessObs run from the command line using
    a list of arguments. Any external program should call processObs directly
    

    Inputs:
        None to the function itself, but will pull the sys.argv. The possible arguments
        are the relevant time range, the desired instruments, and optionally the input
        folder. The outputFolder, outFile, and downsize option are not accessible via
        the command line at this point.
        
        Required:
            starting time - YYYY-MM-DDTHH:MM recommended, but anything supported by
                            parse_time from sunpy will work.
                            *** must be first argument after wombatProcessObs.py ***

            ending time -   YYYY-MM-DDTHH:MM recommended, but anything supported by
                            parse_time from sunpy will work.
                            *** must be second argument after wombatProcessObs.py ***
            
            inst tags -     instrument tags to search for from the following list
                            can be multiple tags, not just a single
                            *** must appear directly after ending time ***
        Optional:
            inFolder -      top directory where the files will pulled from their
                            appropriate subfolders
                            defaults to pullFolder/
    
        
        Available Instrument Tags:
            AIAnum  = SDO AIA where num represents a wavelength from [94, 131, 171*, 
                      193*, 211, 304*, 335, 1600, 1700] with * most common
            C2      = LASCO C2
            C3      = LASCO C3
            COR1    = STEREO COR1
            COR2    = STEREO COR2    
            EUVInum = STEREO EUVI where num is a wavelength from [171, 195, 284, 304]
            HI1     = STEREO HI1
            HI2     = STEREO H2
            SoloHI  = All quadrants from Solar Orbiter HI
            SoloHI1 = Quadrant 1 from Solar Orbiter HI
            SoloHI2 = Quadrant 2 from Solar Orbiter HI
            SoloHI3 = Quadrant 3 from Solar Orbiter HI
            SoloHI4 = Quadrant 4 from Solar Orbiter HI
            WISPR   = Both inner and outer from PSP WISPR
            WISPRI  = Inner only from PSP WISPR
            WISRPO  = Outer only from PSP WISPR
            * all STEREO values will both pull A and B (as available) if written as 
            above but one can add the A/B tag on the end (e.g. COR2A) to select a
            single STEREO
    
        Actions:
            Calls processObs
    
        
    """
    
    #|---- All the instrument tags ----|
    tags = ['AIA94', 'AIA131', 'AIA171','AIA193','AIA211','AIA304','AIA335','AIA1600','AIA1700', 'C2', 'C3', 'COR1', 'COR2', 'COR1A', 'COR2A', 'COR1B', 'COR2B', 'EUVI171', 'EUVI195', 'EUVI284', 'EUVI304', 'EUVI171A', 'EUVI195A', 'EUVI284A', 'EUVI304A', 'EUVI171B', 'EUVI195B', 'EUVI284B', 'EUVI304B', 'HI1', 'HI2', 'HI1A', 'HI2A', 'HI1B', 'HI2B','SOLOHI', 'SOLOHI1', 'SOLOHI2', 'SOLOHI3', 'SOLOHI4', 'WISPR', 'WISPRI', 'WISPRO']
    
    #|---- Pull the command line args ----|
    vals = sys.argv[1:]
    if vals[1] < vals[0]:
        sys.exit('Exiting processObs, start time is after end time...')
    
    #|---- Pull times and check format ----|
    try:
        temp = parse_time(vals[0])
    except:
        print('Error in start time format')
    try:
        temp = parse_time(vals[1])
    except:
        print('Error in end time format')
    times = [vals[0], vals[1]]
    
    vals = vals[2:]
    
    #|---- Check rest for inst/folder ----|
    insts = []
    inFolder = 'pullFolder/'
    for val in vals:
        if val.upper() not in tags:
            if os.path.isdir(val):
                inFolder = val
            else:
                sys.exit(val + 'is not inst tag or exisiting intput folder. Exiting... ')
        else:
            insts.append(val.upper().replace('SOLO', 'Solo'))
     
    if len(insts) < 1:
        sys.exit('No instrument tag provided')
    
    #|---- Set up spice kernels as needed ----|
    kernelSpot = os.getcwd() + '/spiceKernels/'
    # Kernels everyone likely needs
    load_common_kernels(kernelSpot)
    # Check for psp or solo
    loadPSP, loadSOLO = False, False
    for inst in insts:
        if inst.upper() in ['WISPR', 'WISPRI', 'WISPRO']:
            loadPSP = True
        elif inst.upper() in ['SOLOHI', 'SOLOHI1', 'SOLOHI2', 'SOLOHI3', 'SOLOHI4',]:
            loadSOLO = True
    if loadPSP:
        load_psp_kernels(kernelSpot+'psp/')
    if loadSOLO:
        load_solo_kernels(kernelSpot+'solo/')
    
    proIms, fnames = processObs(times, insts, inFolder=inFolder)    
    
    thePickler(proIms, fnames)
           

if __name__ == '__main__':
    commandLineWrapper()