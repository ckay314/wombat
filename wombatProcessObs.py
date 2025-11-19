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
    The code will set up a folder structure at outFolder with nested satellite
    and instument folders (if it doesn't already exist). It will then dump the 
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
import astropy.units as u
from astropy.time import TimeDelta
from sunpy.time import parse_time

sys.path.append('prepCode/') 
from secchi_prep import secchi_prep
from wispr_prep import wispr_prep
from lasco_prep import c2_prep, c3_prep
from solohi_prep import solohi_fits2grid
from aia_prep import aia_prep
from wombatPullObs import setupFolderStructure
from sunspyce import load_common_kernels, load_psp_kernels, load_solo_kernels


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
def processAIA(times, wavs, inFolder='pullFolder/SDO/AIA/', outFolder='wbFits/SDO/AIA/', downSize=1024):
    """
    Function to process level 1 AIA observations using AIA_prep
    
    This is just a basic wrapper of existing aiapy procedures, not a port
    of IDL code like the other process scripts. The resulting fits files
    are in DN units, not total brightness but this is sufficient since we
    do not support EUV masses. 

    Inputs:
        times: an array with [startTime, endTime] where both
        
        wavs:  an array of wavelength strings
    
    Optional Inputs:
        inFolder: top folder for unprocessed results, will open from outFolder/wav/
                  defaults to pullFolder/SDO/AIA/
               
        outFolder: top folder for processed results, will be save in outFolder/SDO/AIA/wav/
                   defaults to wbFits/SDO/AIA/
    
        downSize: maximum resolution to save the processed fits files 
                  defaults to 1024 (square image)
        
    Outputs:
        The processed fits files will be placed in the appropriate folders within outFolder
        and the function returns an array of strings/lines corresponding to the instrument
        header line and the locations of the processed files. A wrapper function can dump these
        directly into a text file that is then used to launch the wombat gui.
    

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
    outLines = []   
    # Loop through each wavelength        
    for i in range(nWavs):
        if len(goodFiles[i]) > 0:
            print ('|--- Processing SDO AIA '+str(wavs[i])+'---|')      
            # Make an array and sort alphabetically = time sorted      
            goodFiles[i] = np.sort(np.array(goodFiles[i]))
            # Pass to the prep scripts
            print ('Running aia_prep, this may take some time...')
            ims = aia_prep(goodFiles[i], downSize=downSize)
            # If this is the first image add the instrument header to outlines
            if len(ims) > 0:
                outLines.append('SDO_AIA_'+str(wavs[i]) + '\n')
            # Take processed result, add keywords, and save
            for k in range(len(ims)):
                print ('On file '+str(k+1)+' out of '+str(len(ims)))
                im = ims[k]
                # Add extra keywords wombat wants
                im.meta['OBSRVTRY'] = 'SDO'
                im.meta['DETECTOR'] = 'AIA'
                im.meta['SC_ROLL'] = 0.
                # Set up output filename
                ymd = im.meta['DATE-OBS'].replace('-','').replace(':','')[:15]  
                fitsName = 'wbpro_aia'+str(wavs[i])+'_'+ymd+'.fits'
                # Save the fits file
                fullOut = outFolder+wavs[i]+'/'+fitsName
                im.save(fullOut, overwrite=True)
                # Add it to the list of avail obs
                outLines.append(fullOut+'\n')
        else:
            print ('No files found for AIA '+str(wavs[i]))
                           
    return outLines

    

# |------------------------------------------------------------|
# |--------------- Process LASCO Observations -----------------|
# |------------------------------------------------------------|
def processLASCO(times, insts, inFolder='pullFolder/SOHO/LASCO/', outFolder='wbFits/SOHO/LASCO/', downSize=1024, prepDir='prepFiles/soho/lasco/'):
    """
    Function to process level 0.5 LASCO observations using c2_prep/c3_prep
    
    This is a wrapper to find the data and pull the ported versions of IDL solarsoft
    routines. The resulting fits files are in total brightness and a near exact match
    to using the original IDL procedures. These files can be passed directly to the
    wombat mass calculation procedure

    Inputs:
        times: an array with [startTime, endTime] where both
        
        insts: strings for selected instruments (C2 or C3)
    
    Optional Inputs:
        inFolder: top folder for unprocessed results, will open from outFolder/inst/
                  defaults to pullFolder/SOHO/LASCO/
               
        outFolder: top folder for processed results, will be save in outFolder/SOHO/inst/
                   defaults to wbFits/SOHO/LASCO/
        
        prepDir:   folder where the prep files for LASCO are stored
                   defaults ot prepFiles/soho/lasco
    
        downSize: maximum resolution to save the processed fits files 
                  *** not implemented in c2/3_prep yet so key is currently ignored ***
        
    Outputs:
        The processed fits files will be placed in the appropriate folders within outFolder
        and the function returns an array of strings/lines corresponding to the instrument
        header line and the locations of the processed files. A wrapper function can dump these
        directly into a text file that is then used to launch the wombat gui.
    
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
    outLines = []   
    # Sort and array-ify
    for i in range(nInsts):
        goodFiles[i] = np.sort(np.array(goodFiles[i]))
    
    # Add in the actual processing, saving, and output to runFile  
    print ('|---- Processing LASCO ----|')
    for i in range(nInsts):
        if len(goodFiles[i]) > 0:
            print ('|---- Processing LASCO ' + insts[i] + ' ----|')
            # Make an array and sort alphabetically = time sorted      
            goodFiles[i] = np.sort(np.array(goodFiles[i]))
            outLines.append('LASCO_' + str(insts[i]) + '\n')
            if insts[i] == 'C2':
                ims, hdrs = c2_prep(goodFiles[i], prepDir)  
            if insts[i] == 'C3':    
                ims, hdrs = c3_prep(goodFiles[i], prepDir)      
            for j in range(len(ims)):
                ymd = hdrs[j]['DATE-OBS'].replace('/','')+'T'+hdrs[j]['TIME-OBS'].replace(':','')[:6]
                fitsName = 'wbpro_lasco'+insts[i]+'_'+ymd+'.fits'
                hdrs[j]['OBSRVTRY'] = 'SOHO'
                hdrs[j].remove('HISTORY', remove_all=True, ignore_missing=True)
                hdrs[j]['HISTORY'] = 'Offset_bias applied but header stripped bc made astropy angry'
                # Save the fits file
                fullOut = outFolder+insts[i]+'/'+fitsName
                print (fullOut)
                fits.writeto(fullOut, ims[j], hdrs[j], overwrite=True)
                outLines.append(fullOut+'\n')
                           
    return outLines



# |------------------------------------------------------------|
# |-------------- Process SoloHI Observations -----------------|
# |------------------------------------------------------------|
def processSoloHI(times, insts, inFolder='pullFolder/SolO/SoloHI/', outFolder='wbFits/SolO/SoloHI/'):
    """
    Function to process the SoloHI images to mosaics or just pass along singles
    
    This is a wrapper to find the data and pull the ported versions of IDL solarsoft
    routines. The resulting fits files are in total brightness and a near exact match
    to using the original IDL procedures. These files can be passed directly to the
    wombat mass calculation procedure

    Inputs:
        times: an array with [startTime, endTime] where both
        
        insts: strings for selected instruments (Mosaic, SoloHI# where # in 1-4)
    
    Optional Inputs:
        inFolder: top folder for unprocessed results, will open from outFolder/#/
                  defaults to pullFolder/SolO/SoloHI/
               
        outFolder: top folder for processed results, will be save in outFolder/inst/
                   defaults to wbFits/SolO/SoloHI/
    
    Outputs:
        The processed fits files will be placed in the appropriate folders within outFolder
        and the function returns an array of strings/lines corresponding to the instrument
        header line and the locations of the processed files. A wrapper function can dump these
        directly into a text file that is then used to launch the wombat gui.
    

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
        outLines = []   
        ims = []
        hdrs = []
        print ('|--- Processing SoloHI ---|')
        for i in range(nQuads):
            im, hdr = solohi_fits2grid(fullQuadFiles[i])
            ims.append(im)
            hdrs.append(hdr)
            
        # Save the results
        if len(ims) > 0:
            outLines.append('SolOHI_Quad \n')
            if not os.path.exists(outFolder+'Mosaic/'):
                # Make if if it doesn't exist
                os.mkdir(outFolder+'Mosaic/')
            for i in range(len(ims)):
                print('Processing image',i,'of', len(ims))
                ymd = hdrs[i]['DATE-OBS'].replace('-','').replace(':','')[:15]  
                fitsName = 'wbpro_solohiquad_'+ymd+'.fits'      
                fullName = outFolder+'Mosaic/' + fitsName
                fits.writeto(fullName, ims[i], hdrs[i], overwrite=True)
                outLines.append(fullName+'\n')
    else:
        outLines = []
        for i in range(nInsts):
            if len(goodFiles[i]) > 0:
                outLines.append('SolOHI_'+str(i+1)+'\n')
                for j in range(len(goodFiles[i])):
                    outLines.append(inFolder+insts[i]+'/'+goodFiles[i][j]+'\n')
                                           
    return outLines
 

# |------------------------------------------------------------|
# |-------------- Process STEREO Observations -----------------|
# |------------------------------------------------------------|
def processSTEREO(times, insts, inFolder='pullFolder/STEREO/', outFolder='wbFits/STEREO/', downSize=1024, prepDir='prepFiles/stereo/'):
    """
    Function to process STEREO observations using secchi_prep
    
    This is a wrapper to find the data and pull the ported versions of IDL solarsoft
    routines. The resulting fits files are in total brightness and a near exact match
    to using the original IDL procedures. These files can be passed directly to the
    wombat mass calculation procedure

    Inputs:
        times: an array with [startTime, endTime] where both
        
        insts: strings for selected instruments (EUVI# where # in [171, 195, 284, 304]
                COR1, COR2, HI1, HI2)
                *** will automatically pull both A/B as available ***
    
    Optional Inputs:
        inFolder: top folder for unprocessed results, will open from outFolder/inst/
                  defaults to pullFolder/SOHO/LASCO/
               
        outFolder: top folder for processed results, will be save in outFolder/SOHO/inst/ 
                   defaults to wbFits/SOHO/LASCO/
    
        downSize: maximum resolution to save the processed fits files 
    
        gtFileIn: save file (ecchi_gtdbase.geny) needed for EUVI processing in scc_funs/scc_gt2sunvec
                  defaults to prepFiles/stereo/secchi_gtdbase.geny
        
    Outputs:
        The processed fits files will be placed in the appropriate folders within outFolder
        and the function returns an array of strings/lines corresponding to the instrument
        header line and the locations of the processed files. A wrapper function can dump these
        directly into a text file that is then used to launch the wombat gui.
    
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
    outLines = []   
    fronts = ['STA_', 'STB_']
    fronts2 = ['wbpro_sta', 'wbpro_stb']
    # Add in the actual processing, saving, and output to runFile  
    

    print ('|---- Processing STEREO ----|')
    for i in range(nInsts):
        inst = insts[i]
        # Anyone thats not a pB triplet
        if inst != 'COR1':
            for j in ABtoDo[i]:
                if len(goodFiles[i][j]) > 0:
                    print ('|--- Processing STEREO '+inst+' '+AB[j]+' ---|')
                    outLines.append(fronts[j]+inst.replace('EUVI','EUVI_')+'\n')
                    ims, hdrs = secchi_prep(goodFiles[i][j], outSize=[downSize, downSize], prepDir=prepDir) 
                    for k in range(len(ims)):
                        ymd = hdrs[k]['DATE-OBS'].replace('-','').replace(':','')[:15]  
                        fitsName = fronts2[j]+inst.lower()+'_'+ymd+'.fits' 
                        if 'EUVI' in inst:
                            nowFold = outFolder+inst.replace('EUVI', 'EUVI'+AB[j]+'/')
                        else:
                            nowFold = outFolder+inst+AB[j] 
                        fullName = nowFold + '/' + fitsName   
                        fits.writeto(fullName, ims[k], hdrs[k], overwrite=True)
                        outLines.append(fullName+'\n')

        # Process the triplets
        else:
            for j in ABtoDo[i]:
                myCOR = 'COR1'+AB[j]
                
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
                    outLines.append(fronts[j]+inst+'\n')
                    for k in range(len(goodTrips)):
                        print ('      on triplet ' +str(1+k))
                        aIm, aHdr = secchi_prep(goodTrips[k], polarizeOn=True, silent=True, prepDir=prepDir)
                        ims.append(aIm[0])
                        hdrs.append(aHdr[0])
                        ymd = hdrs[k]['DATE-OBS'].replace('-','').replace(':','')[:15]  
                        fitsName = fronts2[j]+inst.lower()+'_'+ymd+'.fits' 
                        nowFold = outFolder+inst+AB[j] 
                        fullName = nowFold + '/' + fitsName   
                        fits.writeto(fullName, ims[k], hdrs[k], overwrite=True)
                        outLines.append(fullName+'\n')
    
    return outLines
    

# |------------------------------------------------------------|
# |--------------- Process WISPR Observations -----------------|
# |------------------------------------------------------------|
def processWISPR(times, insts, wcalpath='prepFiles/psp/wispr/',  inFolder='pullFolder/PSP/WISPR/', outFolder='wbFits/PSP/WISPR/', downSize=1024):
    """
    Function to process the level 2 WISPR data
    
    This is a wrapper to find the data and pull the ported versions of IDL solarsoft
    routines. The resulting fits files are in total brightness and a near exact match
    to using the original IDL procedures. These files can be passed directly to the
    wombat mass calculation procedure

    Inputs:
        times: an array with [startTime, endTime] where both
        
        insts: strings for selected instruments (Mosaic, SoloHI# where # in 1-4)
    
    Optional Inputs:
        wcalpath: folder where the WISPR calibration files are stored
                  defaults to prepFiles/psp/wispr/
    
        inFolder: top folder for unprocessed results, will open from outFolder/#/
                  defaults to pullFolder/PSP/WISPR/
               
        outFolder: top folder for processed results, will be save in outFolder/inst/
                   defaults to wbFits/PSP/WISPR/
    
    Outputs:
        The processed fits files will be placed in the appropriate folders within outFolder
        and the function returns an array of strings/lines corresponding to the instrument
        header line and the locations of the processed files. A wrapper function can dump these
        directly into a text file that is then used to launch the wombat gui.
    
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
    outLines = []   
    
    # Add in the actual processing, saving, and output to runFile  
    print ('|---- Processing WISPR ----|')
    for i in range(nInsts):
        if len(goodFiles[i]) > 0:
            print ('|---- Processing WISPR ' + insts[i] + ' ----|')
            # Make an array and sort alphabetically = time sorted      
            goodFiles[i] = np.sort(np.array(goodFiles[i]))
            ims, hdrs = wispr_prep(goodFiles[i], wcalpath, straylightOff=True)
            outLines.append('WISPR_'+ insts[i] + '\n')
            for j in range(len(ims)):
                ymd = hdrs[j]['DATE-OBS'].replace('-','').replace(':','')[:15]  
                fitsName = 'wbpro_wispr'+insts[i]+'_'+ymd+'.fits'
                fullName = outFolder+insts[i]+'/' + fitsName
                fits.writeto(fullName, ims[j], hdrs[j], overwrite=True)
                outLines.append(fullName+'\n')
            
            

                           
    return outLines



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
    f1 = open(outFile, 'w')
        
            
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
        outLines = processAIA(times, doAIA, inFolder='pullFolder/SDO/AIA/')
        if type(outLines) != type(None):
            for line in outLines:
                f1.write(line)
        else:
            print('Unable to process any AIA images')
            
                
    # |-------------------------------|
    # |------------ SOHO -------------|
    # |-------------------------------|
    doLASCO = []
    if 'C2' in insts: doLASCO.append('C2')
    if 'C3' in insts: doLASCO.append('C3')
    if len(doLASCO) > 0:
        outLines = processLASCO(times, doLASCO)
        if type(outLines) != type(None):
            for line in outLines:
                f1.write(line)
        else:
            print('Unable to process any LASCO images')
   
    # |-------------------------------|
    # |------------ STEREO -----------|
    # |-------------------------------|
    doSTEREO = []
    STkeys = ['COR1', 'COR2', 'EUVI171', 'EUVI195', 'EUVI284', 'EUVI304', 'HI1', 'HI2', 'COR1A', 'COR2A', 'EUVI171A', 'EUVI195A', 'EUVI284A', 'EUVI304A', 'HI1A', 'HI2A', 'COR1B', 'COR2B', 'EUVI171B', 'EUVI195B', 'EUVI284B', 'EUVI304B', 'HI1B', 'HI2B']
    for inst in insts:
        if inst in STkeys:
            doSTEREO.append(inst)
    if len(doSTEREO) > 0:
        outLines = processSTEREO(times, doSTEREO)
        if type(outLines) != type(None):
            for line in outLines:
                f1.write(line)
        else:
            print('Unable to process any STEREO images')
        
    

    # |-------------------------------|
    # |------------ WISPR ------------|
    # |-------------------------------|
    doWISPR = []
    if 'WISPR' in insts: doWISPR = ['Inner', 'Outer']
    elif 'WISPRI' in insts: doWISPR.append('Inner')
    elif 'WISPRO' in insts: doWISPR.append('Outer')
    if len(doWISPR) > 0:
        outLines = processWISPR(times, doWISPR)
        if type(outLines) != type(None):
            for line in outLines:
                f1.write(line)
        else:
            print('Unable to process any WISPR images')
    

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
        outLines = processSoloHI(times, doSoloHI)
        if type(outLines) != type(None):
            for line in outLines:
                f1.write(line)
        else:
            print('Unable to process any SoloHI images')
        
    # |-------------------------------|
    # |---- Close the output file ----|
    # |-------------------------------|
    f1.close()
    
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
    
    processObs(times, insts, inFolder=inFolder)    
           

if __name__ == '__main__':
    commandLineWrapper()