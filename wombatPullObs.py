"""
Module to pull observations for the wombat GUI using Sunpy Fido/wget

Note that this only works as well as Fido works and there seems to be a lot
of issues in terms of communication with JSOC, especially with VPNs and firewalls
and whatnot. We already pull PSP/SolO using wget as they are available online
and we can take the Fido find results and avoid the download issues. Fixing Fido
is outside of wombat scope.


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
    outFolder: the top level folder name for where data will be saved
               defaults to pullFolder/ (from wombatPullObs)

    EUVtime: time resolution in minutes to use for EUV images
    
    CORtime: time resolution in minutes to use for coronagraph images

    HItime:  time resolution in minutes to use for HI images
    
    ---> default values of 10, 20, 30. Will downselect only if avail at higher res

Outputs:
    The code will set up a folder structure at outFolder with nested satellite
    and instument folders (if it doesn't already exist). It will then dump the 
    unprocessed files into the appropriate folders

Usage:
    from wombatPullObs import pullObs
    times = ['YYYY/MM/DDTHH:MM', 'YYYY/MM/DDTHH:MM']
    sats  = ['AIA', 'COR2', 'WISPR']
    waves = [171]
    outFolder = '/Users/kaycd1/wombat/pullFolder/'
    pullObs(times, sats, waves, outFolder=outFolder)

"""


from sunpy.net import Fido, attrs as a
import astropy.units as u
import numpy as np
import wget
from sunpy.time import parse_time
from astropy.io import fits
import os, sys

# |------------------------------------------------------------|
# |------------ Check/Make Output Folder Structure ------------|
# |------------------------------------------------------------|
def setupFolderStructure(topName):
    """
    Helper function to set up the folder structure for the wombat observations

    Inputs:
        topName: the name of the top level directory

    Outputs:
        None
    
    """
    # |---------------------------------------|
    # |---- Check the top level directory ----|
    # |---------------------------------------|
    if not os.path.exists(topName):
        # Make if if it doesn't exist
        os.mkdir(topName)
    
    # |---------------------------------------|
    # |------------ Check subdirs ------------|
    # |---------------------------------------|
    # Satellite directory names 
    satDirs = ['PSP', 'SDO', 'SOHO', 'SolO', 'STEREO']
    # Instrument/wavelength directory names
    instDirs = [['WISPR','WISPR/Inner', 'WISPR/Outer'], 
                ['AIA', 'AIA/94', 'AIA/131', 'AIA/171', 'AIA/193', 'AIA/211', 'AIA/304', 
                'AIA/335', 'AIA/1600', 'AIA/1700'], 
                ['LASCO', 'LASCO/C2', 'LASCO/C3'],
                ['SoloHI', 'SoloHI/1', 'SoloHI/2', 'SoloHI/3', 'SoloHI/4' ], 
                ['EUVIA', 'EUVIA/171', 'EUVIA/195', 'EUVIA/284', 'EUVIA/304', 'EUVIB', 
                'EUVIB/171', 'EUVIB/195', 'EUVIB/284', 'EUVIB/304', 'COR1A', 'COR2A', 
                'COR1B', 'COR2B', 'HI1A', 'HI2A', 'HI1B', 'HI2B']]    
                
    # Check each dir/subdir
    for i in range(len(satDirs)):
        if not os.path.exists(topName+'/'+satDirs[i]):
            os.mkdir(topName+'/'+satDirs[i])
        for subDir in instDirs[i]:
            if not os.path.exists(topName+'/'+satDirs[i]+'/'+subDir):
                os.mkdir(topName+'/'+satDirs[i]+'/'+subDir)


# |------------------------------------------------------------|
# |------------------ Grab AIA Observations -------------------|
# |------------------------------------------------------------|
def pullAIA(times, wavs, EUVtime=10, outFolder='pullFolder/'):
    """
    Function to pull level 1 AIA observations using FIDO

    Inputs:
        times: an array with [startTime, endTime] where both
        
        wavs:  an array of wavelength strings
    
    Optional Inputs:
        EUVtime: time resolution in minutes, will set min spacing between images
        
        outFolder: top folder, results will be saved in outFolder/SDO/AIA/wav/

    Outputs:
        The downloaded fits files will be placed in the appropriate folders within outFolder

    """
    
    for wv in wavs:
        # |-----------------------------|
        # |--------- Searching ---------|
        # |-----------------------------|
        # Search Fido for a wavelength, sample time works here
        result = Fido.search(a.Time(times[0], times[1]), a.Instrument.aia, a.Wavelength(int(wv)*u.angstrom), a.Sample(EUVtime*u.minute))
        
        # |-----------------------------|
        # |-------- Downloading --------|
        # |-----------------------------|
        if len(result) > 0:
            print('Downloading AIA files...')
            downloaded_files = Fido.fetch(result[0,:], path=outFolder+'SDO/AIA/'+wv+'/{file}') 
            
        else:
            print ('Cannot find any files for AIA '+wv)
    

# |------------------------------------------------------------|
# |----------------- Grab LASCO Observations ------------------|
# |------------------------------------------------------------|
def pullLASCO(times, insts, CORtime=20, outFolder='pullFolder/'):
    """
    Function to pull level 0.5 LASCO observations using FIDO

    Inputs:
        times: an array with [startTime, endTime] where both
        
        insts: strings for selected instruments (C2 or C3)
    
    Optional Inputs:
        CORtime: time resolution in minutes, will set min spacing between images
        
        outFolder: top folder, results will be saved in outFolder/LASCO/C#/

    Outputs:
        The downloaded fits files will be placed in the appropriate folders within outFolder

    """
    # |-----------------------------|
    # |--------- Searching ---------|
    # |-----------------------------|
    # Search Fido, sample works here
    result = Fido.search(a.Time(times[0], times[1]), a.Instrument.lasco, a.Sample(CORtime*u.minute))

    
    # |-----------------------------|
    # |---------- Sorting ----------|
    # |-----------------------------|
    # arrays to hold file names and string datetimes
    whichC = [[], []]
    ymdts = [[], []]


    # Loop through what we found and store it if we like it
    for i in range(len(result[0]['fileid'])):
        # Don't pull the 512x512 images, have to check file size
        # Check if c2 via the file path
        if ('/c2/' in result[0]['fileid'][i]) & (result[0]['Size'][i].to_value() > 1.1):
            whichC[0].append(i)
            ymdts[0].append(str(result[0]['Start Time'][i])[0:10].replace('-','') + 'T' + str(result[0]['Start Time'][i])[11:16].replace(':',''))
            
        # Check if c3 via the file path
        elif ('/c3/' in result[0]['fileid'][i]) & (result[0]['Size'][i].to_value() > 1.1):
            whichC[1].append(i)
            ymdts[1].append(str(result[0]['Start Time'][i])[0:10].replace('-','') + 'T' + str(result[0]['Start Time'][i])[11:16].replace(':',''))
    
    
    # |-----------------------------|
    # |-------- Downloading --------|
    # |-----------------------------|
    # LASCO C2   
    if 'C2' in insts:
        if len(whichC[0]):
            print('Downloading LASCO C2 files...')
            c2path = outFolder + 'SOHO/LASCO/C2/'
            for i in range(len(whichC[0])):
                downloaded_files = Fido.fetch(result[0,whichC[0][i]], path= c2path + ymdts[0][i] + '_C2_{file}') 
        else:
            print('Cannot find any files for LASCO C2')
    # LASCO C3         
    if 'C3' in insts:
        if len(whichC[1]):
            print('Downloading LASCO C3 files...')
            c3path = outFolder + 'SOHO/LASCO/C3/'
            for i in range(len(whichC[1])):
                downloaded_files = Fido.fetch(result[0,whichC[1][i]], path= c3path + ymdts[1][i] + '_C3_{file}') 
        else:
            print('Cannot find any files for LASCO C3')



# |------------------------------------------------------------|
# |---------------- Grab SoloHI Observations ------------------|
# |------------------------------------------------------------|
def pullSoloHI(times, insts, HItime=30, outFolder='pullFolder/'):
    """
    Function to pull level 2 SoloHI observations using a combo of Fido search
    and wget from the NRL site directly. Only pulls those with 1ft, 2ft, 3fg,
    or 4fg tags

    Inputs:
        times: an array with [startTime, endTime] where both
        
        insts: strings for selected instruments (SoloHI, SoloHI# where # in 1-4)
    
    Optional Inputs:
        HItime: time resolution in minutes, will set min spacing between images
        
        outFolder: top folder, results will be saved in outFolder/SolO/SoloHI/#/

    Outputs:
        The downloaded fits files will be placed in the appropriate folders within outFolder

    """
    
    # |-----------------------------|
    # |--------- Searching ---------|
    # |-----------------------------|
    # Fido usually ok searching at least
    result = Fido.search(a.Time(times[0], times[1]), a.Instrument.solohi)
    
    # Pull the L2 data
    goodIdx = []
    if len(result) > 0:
        for i in range(len(result[result.keys()[0]]['fileid'])):
            if '/L2/' in result[result.keys()[0]]['fileid'][i]:
                goodIdx.append(i)
    else:
        print ('Cannot find any files for SoloHI')
            
    # |-----------------------------|
    # |---------- Sorting ----------|
    # |-----------------------------|
    # CK has given up on the vso downloading bc its almost always a 404 or 403 error
    # but we can wget these files
    # base path = 'https://solohi.nrl.navy.mil/so_data/L2/'
    # followed by YYYYMMDD/samefilename.fits
    basePath = 'https://solohi.nrl.navy.mil/so_data/L2/'
    if len(goodIdx) > 0:
        print('')
        print('Downloading SoloHI files...')
        
        # Track the last time we pulled so we can get the res we want
        lastTime = [[],[], [], []]
        for i in goodIdx:
            ogname = result[result.keys()[0]]['fileid'][i]
            fname = ogname[ogname.rfind('/')+1:]
            myQuad = fname[15:18]
            ymd = fname[19:27]
            myPath = basePath + ymd + '/' + fname
            grabThis = False
            
            # Quad 1
            if (myQuad == '1ft') & (('Mosaic' in insts) or ('SoloHI1' in insts)):
                savePath = outFolder + 'SolO/SoloHI/1/'
                if len(lastTime[0]) == 0:
                    grabThis = True
                    lastTime[0].append(i)
                elif (result[0]['Start Time'][i] - result[0]['Start Time'][lastTime[0][-1]]).to(u.min).to_value() >= HItime:
                    grabThis = True
                    lastTime[0].append(i)
            # Quad 2
            elif (myQuad == '2ft') & (('Mosaic' in insts) or ('SoloHI2' in insts)):
                savePath = outFolder + 'SolO/SoloHI/2/'
                if len(lastTime[1]) == 0:
                    grabThis = True
                    lastTime[1].append(i)
                elif (result[0]['Start Time'][i] - result[0]['Start Time'][lastTime[1][-1]]).to(u.min).to_value() >= HItime:
                    grabThis = True
                    lastTime[1].append(i)
            # Quad 3
            elif (myQuad == '3fg') & (('Mosaic' in insts) or ('SoloHI3' in insts)):
                savePath = outFolder + 'SolO/SoloHI/3/'
                if len(lastTime[2]) == 0:
                    grabThis = True
                    lastTime[2].append(i)
                elif (result[0]['Start Time'][i] - result[0]['Start Time'][lastTime[2][-1]]).to(u.min).to_value() >= HItime:
                    grabThis = True
                    lastTime[2].append(i)
            # Quad 4
            elif (myQuad == '4fg') & (('Mosaic' in insts) or ('SoloHI4' in insts)):
                savePath = outFolder + 'SolO/SoloHI/4/'
                if len(lastTime[3]) == 0:
                    grabThis = True
                    lastTime[3].append(i)
                elif (result[0]['Start Time'][i] - result[0]['Start Time'][lastTime[3][-1]]).to(u.min).to_value() >= HItime:
                    grabThis = True
                    lastTime[3].append(i)                
            
            # |-----------------------------|
            # |-------- Downloading --------|
            # |-----------------------------|
            if grabThis:   
                if not os.path.exists(savePath+fname):    
                    print('')
                    print('Downloading', fname) 
                    temp = wget.download(myPath, out=savePath)
 

# |------------------------------------------------------------|
# |---------------- Grab STEREO Observations ------------------|
# |------------------------------------------------------------|
def pullSTEREO(times, insts, EUVtime=10, CORtime=20, HItime=30, outFolder='pullFolder/'):
    """
    Function to pull STEREO observations using Fido 
    Will pull n4eu/n5eu for EUVI, s/n4c1 for COR1, 
    d4c2 for COR2 (no pB), s4c1/2 for HI1/2

    Inputs:
        times: an array with [startTime, endTime] where both
        
        insts: strings for selected instruments (EUVI# where # in [171, 195, 284, 304]
                COR1, COR2, HI1, HI2)
                *** will automatically pull both A/B as available ***
    
    Optional Inputs:
        CORtime: time resolution in minutes, will set min spacing between images
        
        outFolder: top folder, results will be saved in outFolder/SolO/SoloHI/#/

    Outputs:
        The downloaded fits files will be placed in the appropriate folders within outFolder

    """
    # |-----------------------------|
    # |-------- STEREO Prep --------|
    # |-----------------------------|
    AB = ['A', 'B']
    nInsts = len(insts)
    ABtoDo = {}
    # Check if passed specifically A or B in each inst
    # and change check list but rm tag from inst name 
    # bc originally coded to just search for both
    for i in range(nInsts):
        if 'A' in insts[i]:
            insts[i] = insts[i].replace('A','')
            ABtoDo[insts[i]] = [0]
        elif 'B' in insts[i]:
            insts[i] = insts[i].replace('B','')
            ABtoDo[insts[i]] = [1]
        else:
            ABtoDo[insts[i]] = [0,1]
            
    # Figure out which instruments we want since
    # process them differently
    EUVIwav = []
    CORs = []
    HIs  = []        
    for inst in insts:
        if 'EUVI' in inst:
            EUVIwav.append(int(inst.replace('EUVI','')))
        elif 'COR' in inst:
            CORs.append(inst)
        elif 'HI' in inst:
            HIs.append(inst)
    # Get time range in minutes, which we may use later
    # since the Sample option in Fido is some times problematic
    # make astropy time objects
    startAPT = parse_time(times[0])
    endAPT = parse_time(times[1])
    timeRange = (endAPT - startAPT).to(u.min).to_value()

    # |-----------------------------|
    # |--------- Searching ---------|
    # |-----------------------------|
    # Secchi doesn't like using Sample, randomly starts yeeting things so no files left
    result = Fido.search(a.Time(times[0], times[1]), a.Instrument.secchi)
    
            
    # |-----------------------------|
    # |---------- Sorting ----------|
    # |-----------------------------|
    # EUVI sorting 
    if len(EUVIwav) > 0:
        wavidx = [[[], []] for i in range(len(EUVIwav))]
    
        # Sometimes Fido decides to give no wavelengths for EUVI so have to work around
        # Test if all entries have two values, the EUVI none flags will trip this
        waveIssue = False
        try:
            temp = np.where(result[0]['Wavelength'][:,0] == EUVIwav[0]*u.AA)[0]
        except:
            waveIssue = True
            print("Fido has decided not to provide EUVI wavelengths so have to download them all")
        if not waveIssue:
            for i in range(len(EUVIwav)):
                wavidx[i][0] =np.where((result[0]['Wavelength'][:,0] == EUVIwav[i]*u.AA) & (result[0]['Source'] == 'STEREO_A'))[0]
                wavidx[i][1] =np.where((result[0]['Wavelength'][:,0] == EUVIwav[i]*u.AA) & (result[0]['Source'] == 'STEREO_B'))[0] 

                # Manually downselect 
                '''for j in range(2):
                    fullN = len(wavidx[i][j])
                    expN  = timeRange / EUVItime
                    if fullN > expN:
                        downN = int(fullN / expN)
                        wavidx[i][j] = wavidx[i][j][::downN]'''
                sublist = [[],[]]
                for j in range(2):
                    for idx in wavidx[i][j]:
                        if len(sublist[j]) == 0:
                            sublist[j].append(idx)
                        else:
                            if (result[0]['Start Time'][idx] - result[0]['Start Time'][sublist[j][-1]]).to(u.min).to_value() >= EUVtime:
                                 sublist[j].append(idx)
                wavidx[i] = sublist
            
            
            if (len(wavidx[i][0]) + len(wavidx[i][0]))  < 1:
                print('Cannot find any files for EUVI', EUVIwav[i])
        else:
            wavidx = [[],[]]
            for i in range(len(result[0]['fileid'])):
                if ('/euvi/' in result[0]['fileid'][i]) & ('_n7eu' not in result[0]['fileid'][i]):
                    if ('a/') in result[0]['fileid'][i]:
                        wavidx[0].append(i) 
                    elif ('b/') in result[0]['fileid'][i]:
                        wavidx[1].append(i)      
                
            # Downselect because Fido will def lag with too many    
            # There's no good way of picking which ones we keep    
            nCrit = 30
            for i in range(2):
                if len(wavidx[i]) > nCrit:  
                    downSel = int(len(wavidx[i]) / nCrit)
                    wavidx[i] = wavidx[i][::downSel]
                    
    # Sort the CORs and HIs
    whichC = [[[], []], [[], []], [[], []], [[], []]]   
    for i in range(len(result[0]['fileid'])):
        mySpot, mySpot = -9999, -9999 
        if ('/cor1/' in result[0]['fileid'][i]) & ('/seq/' in result[0]['fileid'][i]) :
            mySpot = 0
            #whichC[0].append(i)
        elif '/cor2/' in result[0]['fileid'][i]:
            mySpot = 1
            #whichC[1].append(i)
        elif ('/hi_1/' in result[0]['fileid'][i]) & (result[0]['Size'][i].to_value() > 2.4):
            mySpot = 2
            #whichC[2].append(i)
        elif ('/hi_2/' in result[0]['fileid'][i]) & (result[0]['Size'][i].to_value() > 2.4):
            mySpot = 3
            #whichC[3].append(i)
        if '/a/' in result[0]['fileid'][i]:
            mySat = 0
        if '/b/' in result[0]['fileid'][i]:
            mySat = 1
            
        if (mySpot != -9999) &  (mySat != -9999): 
            whichC[mySpot][mySat].append(i)

    # Need to filter COR1 but keep full pB series
    # the series will have s4c1 tags early that 
    # at some point in switch to n4c1 
    for j in range(2):
        #whichC[0][j] = whichC[0][j] # using this to test diff start points
        nTimes = len(whichC[0][j]) / 3
        if nTimes - int(nTimes) != 0:
            if (whichC[0][j][1]-whichC[0][j][0]) != 1:
                whichC[0][j] = whichC[0][j][1:]
            elif (whichC[0][j][2]-whichC[0][j][1]) != 1:
                whichC[0][j] = whichC[0][j][2:]
            nTimes = len(whichC[0][j]) / 3
        expN  = timeRange / CORtime
        if nTimes > expN:
           downN = int(nTimes / expN) * 3
           partial = whichC[0][j][::downN]   
           full = np.zeros(len(partial)*3, dtype=int)
           for i in range(len(partial)):
               #idx = int(partial[i])
               cidx = np.where(np.array(whichC[0][j]) == partial[i])[0][0]
               if cidx+2 < len(whichC[0][j]):
                   full[3*i: 3*(i+1)]= [whichC[0][j][cidx], whichC[0][j][cidx+1], whichC[0][j][cidx+2]]
               #full[3*i: 3*(i+1)]= [idx, idx+1, idx+2]
           whichC[0][j] = full
           

    # Filter out pB images based on file tags
    # d4c tag is totB, n4c is pB and skipping those (for now at least)
    justTot = [[], []]
    for j in range(2):
        for idx in whichC[1][j]:
            # d4c tag is totB, n4c is pB and skipping those (for now at least)
            # Also only want to pull the ~8 mb images
            if ('d4c' in result[0]['fileid'][idx]) & (result[0]['Size'][idx].to_value() > 4.):
                if len(justTot[j]) == 0:
                    justTot[j].append(idx)
                else:
                    if (result[0]['Start Time'][idx] - result[0]['Start Time'][justTot[j][-1]]).to(u.min).to_value() >= CORtime:
                         justTot[j].append(idx)            
    whichC[1] = justTot
    
    # Filter out HIs too (potentially, but prob not)
    for j in range(2):
        for k in [2,3]:
            myKeeps = []
            for idx in whichC[k][j]:
                if len(myKeeps) == 0:
                    myKeeps.append(idx)
                else:
                    if (result[0]['Start Time'][idx] - result[0]['Start Time'][myKeeps[-1]]).to(u.min).to_value() >= HItime:
                         myKeeps.append(idx)
            whichC[k][j] = myKeeps
                

    # |-----------------------------|
    # |-------- Downloading --------|
    # |-----------------------------|
    # Do the DLs        
    if len(EUVIwav) > 0:
        if not waveIssue:
            for i in range(len(wavidx)):
                for j in ABtoDo['EUVI'+str(EUVIwav[i])]:
                    idxs = wavidx[i][j]
                    if len(idxs) > 0:
                        print('Downloading STEREO EUVI'+ AB[j] + ' ' + str(EUVIwav[i]) + ' files...')
                        fullOut = outFolder + 'STEREO/EUVI'+ AB[j]+'/' + str(EUVIwav[i])+'/'
                        downloaded_files = Fido.fetch(result[0,idxs], path= fullOut + 'EUVI_'+ str(EUVIwav[i])+'a_{file}')
        else:
            # Just do A/B for mystery cases
            for j in range(2):
                if len(wavidx[j])>0:
                        print('Downloading Mystery STEREO EUVI files...')
                        downloaded_files = Fido.fetch(result[0,wavidx[j]], path=outFolder + 'STEREO/EUVI_UNKa_{file}')
                        for aF in downloaded_files:
                            mySat = aF[-5].upper()
                            with fits.open(aF) as hdulist:
                                im  = hdulist[0].data
                                hdr = hdulist[0].header
                                print(aF, 'has wavelength ', str(hdr['WAVELNTH']))
                                newName = aF.replace('_UNKa',mySat+'/'+str(hdr['WAVELNTH'])+'/EUVI_'+str(hdr['WAVELNTH'])+'a')
                                os.replace(aF,newName)
            
    if 'COR1' in CORs:
        for j in ABtoDo['COR1']:
            if len(whichC[0][j]) > 0:
                print('Downloading STEREO COR1'+AB[j]+' files...')
                fullOut = outFolder + 'STEREO/COR1'+ AB[j]+'/'
                downloaded_files = Fido.fetch(result[0,whichC[0][j]], path=fullOut+'/COR1'+ AB[j] +'_{file}') 

    if 'COR2' in CORs:
        for j in ABtoDo['COR2']:
            if len(whichC[1][j]) > 0:
                print('Downloading STEREO COR2'+AB[j]+' files...')
                fullOut = outFolder + 'STEREO/COR2'+ AB[j]+'/'
                downloaded_files = Fido.fetch(result[0,whichC[1][j]], path=fullOut+'/COR2'+ AB[j] +'_{file}') 
                    
    if 'HI1' in HIs:
        for j in ABtoDo['HI1']:
            if len(whichC[2][j]) > 0:
                print('Downloading STEREO HI1'+AB[j]+' files...')
                fullOut = outFolder + 'STEREO/HI1'+ AB[j]+'/'
                downloaded_files = Fido.fetch(result[0,whichC[2][j]], path=fullOut+'/HI1'+ AB[j] +'_{file}') 

    if 'HI2' in HIs:
        for j in ABtoDo['HI2']:
            if len(whichC[3][j]) > 0:
                print('Downloading STEREO HI2'+AB[j]+' files...')
                fullOut = outFolder + 'STEREO/HI2'+ AB[j]+'/'
                downloaded_files = Fido.fetch(result[0,whichC[3][j]], path=fullOut+'/HI2'+ AB[j] +'_{file}') 
    

# |------------------------------------------------------------|
# |----------------- Grab WISPR Observations ------------------|
# |------------------------------------------------------------|
def pullWISPR(times, insts, HItime=30, outFolder='pullFolder/'):
    """
    Function to pull level 2 WISPR observations using a combo of Fido search
    and wget from the NRL site directly
    
    The NRL repo does have some V2 files where the L2 data has been reprocessed for 
    straylight (?) (effective 2024/03). Isn't consistent when and where they show up
    so only pulling what Fido/VSO suggests

    Inputs:
        times: an array with [startTime, endTime] where both
        
        insts: strings for selected instruments (WISPRI, WISPRO)
    
    Optional Inputs:
        HItime: time resolution in minutes, will set min spacing between images
        
        outFolder: top folder, results will be saved in outFolder/PSP/WISPR/

    Outputs:
        The downloaded fits files will be placed in the appropriate folders within outFolder

    """
    # |-----------------------------|
    # |--------- Searching ---------|
    # |-----------------------------|
    # Fido usually ok searching at least
    result = Fido.search(a.Time(times[0], times[1]), a.Instrument.wispr)
   
    
    # Pull the L2 data
    goodIdx = []
    if len(result) > 0:
        for i in range(len(result[0]['fileid'])):
            if '/L2/' in result[0]['fileid'][i]:
                goodIdx.append(i)
    else:
        print ('Cannot find any files for WISPR')
 
    
    # |-----------------------------|
    # |---------- Sort/DL ----------|
    # |-----------------------------|
    # CK has given up on the vso downloading bc its almost always a 404 or 403 error
    # but we can wget these files
    # base path = 'https://wispr.nrl.navy.mil/data/rel/fits/L2/'
    # followed by YYYYMMDD/samefilename.fits
    basePath  = 'https://wispr.nrl.navy.mil/data/rel/fits/L2/'
    if len(goodIdx) > 0:
        print('')
        print('Downloading PSP WISPR files...')
           
        # Track the last time we pulled so we can get the res we want
        lastTime = [[],[]]
    
        for i in goodIdx:
            # Full path name from Fido
            ogname = result[result.keys()[0]]['fileid'][i]
            # Just the file name
            fname = ogname[ogname.rfind('/')+1:]
            # Just the date
            ymd = fname[13:21]
            # Build the NRL paths
            myPath = basePath + ymd + '/' + fname
            # Local Paths
            inPath = outFolder+'PSP/WISPR/Inner/'
            outPath = outFolder+'PSP/WISPR/Outer/'
            
            # Inner cases
            if ('V1_1' in fname) & ('WISPRI' in insts):
                # Save the first one, always
                if len(lastTime[0]) == 0:
                    # Make sure we don't already have it
                    print('')
                    print ('Downloading ', fname)
                    if not os.path.exists(inPath+fname):
                        temp = wget.download(myPath, out=inPath)
                    lastTime[0].append(i)
                # Otherwise check how long its been since last
                else:
                    if (result[0]['Start Time'][i] - result[0]['Start Time'][lastTime[0][-1]]).to(u.min).to_value() >= HItime:
                        print('')
                        print ('Downloading ', fname)
                        if not os.path.exists(inPath+fname):
                           temp = wget.download(myPath, out=inPath)
                        lastTime[0].append(i)
                   
            # Outer cases
            elif ('V1_2' in fname) & ('WISPRO' in insts):
                if len(lastTime[1]) == 0:
                    print('')
                    print ('Downloading ', fname)
                    if not os.path.exists(outPath+fname):
                        temp = wget.download(myPath, out=outPath)
                    lastTime[1].append(i)
                else:
                    if (result[0]['Start Time'][i] - result[0]['Start Time'][lastTime[1][-1]]).to(u.min).to_value() >= HItime:
                        print('')
                        print ('Downloading ', fname)
                        if not os.path.exists(outPath+fname):
                            temp = wget.download(myPath, out=outPath)
                        lastTime[1].append(i)
    


# |------------------------------------------------------------|
# |------------ Main function to pull observations ------------|
# |------------------------------------------------------------|
def pullObs(times, insts, outFolder='pullFolder/', EUVtime=10, CORtime=20, HItime=30):
    """
    Main wrapper function for pulling observations

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
        outFolder: the top level folder name for where data will be saved
                   defaults to pullFolder/
        
        EUVtime: time resolution in minutes to use for EUV images
    
        CORtime: time resolution in minutes to use for coronagraph images

        HItime:  time resolution in minutes to use for HI images
    
        ---> default values of 10, 20, 30. Will downselect only if avail at higher res
    

    Outputs:
        The downloaded fits files will be placed in the appropriate folders within outFolder

    """

    # |---------------------------------------| 
    # |---- Check the top level directory ----|
    # |---------------------------------------| 
    setupFolderStructure(outFolder)
    
    
    # |---------------------------------------| 
    # |---- Check all inst keys are valid ----|
    # |---------------------------------------| 
    goods = ['AIA94', 'AIA131', 'AIA171','AIA193','AIA211','AIA304','AIA335','AIA1600','AIA1700', 'C2', 'C3', 'COR1', 'COR2', 'COR1A', 'COR2A', 'COR1B', 'COR2B', 'EUVI171', 'EUVI195', 'EUVI284', 'EUVI304', 'EUVI171A', 'EUVI195A', 'EUVI284A', 'EUVI304A', 'EUVI171B', 'EUVI195B', 'EUVI284B', 'EUVI304B', 'HI1', 'HI2', 'HI1A', 'HI2A', 'HI1B', 'HI2B','SoloHI', 'SoloHI1', 'SoloHI2', 'SoloHI3', 'SoloHI4', 'WISPR', 'WISPRI', 'WISPRO']
    quitIt = False
    for inst in insts:
        if inst not in goods:
            print ('Unknown instrument tag', inst)
            quitIt = True
    if quitIt:
        sys.exit('Quitting pullObs since passed invalid instrument tag')
        
    # |---------------------------------------| 
    # |--------- Check outFolder name --------|
    # |---------------------------------------|
    if outFolder[-1] != '/':
        outFolder = outFolder+'/'    
            
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
        pullAIA(times, doAIA, EUVtime=EUVtime, outFolder=outFolder)
            
                
    # |-------------------------------|
    # |------------ SOHO -------------|
    # |-------------------------------|
    doLASCO = []
    if 'C2' in insts: doLASCO.append('C2')
    if 'C3' in insts: doLASCO.append('C3')
    if len(doLASCO) > 0:
        pullLASCO(times, doLASCO, CORtime=CORtime, outFolder=outFolder)

   
    # |-------------------------------|
    # |------------ STEREO -----------|
    # |-------------------------------|
    doSTEREO = []
    STkeys = ['COR1', 'COR2', 'EUVI171', 'EUVI195', 'EUVI284', 'EUVI304', 'HI1', 'HI2', 'COR1A', 'COR2A', 'EUVI171A', 'EUVI195A', 'EUVI284A', 'EUVI304A', 'HI1A', 'HI2A', 'COR1B', 'COR2B', 'EUVI171B', 'EUVI195B', 'EUVI284B', 'EUVI304B', 'HI1B', 'HI2B']
    for inst in insts:
        if inst in STkeys:
            doSTEREO.append(inst)
    if len(doSTEREO) > 0:
        pullSTEREO(times, doSTEREO, CORtime=CORtime, outFolder=outFolder)
    

    # |-------------------------------|
    # |------------ WISPR ------------|
    # |-------------------------------|
    doWISPR = []
    if 'WISPR' in insts: doWISPR = ['WISPRI', 'WISPRO']
    elif 'WISPRI' in insts: doWISPR.append('WISPRI')
    elif 'WISPRO' in insts: doWISPR.append('WISPRO')
    if len(doWISPR) > 0:
        pullWISPR(times, doWISPR, HItime=HItime, outFolder=outFolder)
    

    # |-------------------------------|
    # |----------- SoloHI ------------|
    # |-------------------------------|
    doSoloHI = []
    if 'SoloHI' in inst: doSoloHI = ['Mosaic']
    else:
        if 'SoloHI1' in inst: doSoloHI.append('SoloHI1')
        if 'SoloHI2' in inst: doSoloHI.append('SoloHI2')
        if 'SoloHI3' in inst: doSoloHI.append('SoloHI3')
        if 'SoloHI4' in inst: doSoloHI.append('SoloHI4')
    if len(doSoloHI) > 0:
        pullSoloHI(times, doSoloHI, HItime=HItime, outFolder=outFolder)
        

# |------------------------------------------------------------|
# |-------------------- Command Line Wrapper ------------------|
# |------------------------------------------------------------|
def commandLineWrapper():
    """
    Wrapper to only be used to make wombatPullObs run from the command line using
    a list of arguments. Any external program should call pullObs directly

    Inputs:
        None to the function itself, but will pull the sys.argv. The possible arguments
        are the relevant time range, the desired instruments, time resolutions, and
        an output directory. The time range and instruments are requires and the others
        are all optional. The arguments have form
        
        Required:
            starting time - YYYY-MM-DDTHH:MM recommended, but anything supported by
                            parse_time from sunpy will work.
                            *** must be first argument after wombatPullObs.py ***

            ending time -   YYYY-MM-DDTHH:MM recommended, but anything supported by
                            parse_time from sunpy will work.
                            *** must be second argument after wombatPullObs.py ***
            
            inst tags -     instrument tags to search for from the following list
                            can be multiple tags, not just a single
                            *** must appear directly after ending time ***
        Optional:
            EUVt -          time resolution in minutes for EUV observations.
                            provided as EUVIt# with no space (e.g EUVI10)
                            defaults to 10 minutes
            
            CORt -          time resolution in minutes for COR observations   
                            provided as CORt# with no space (e.g COR20)
                            defaults to 20 minutes
     
            HIt -           time resolution in minutes for HI observations   
                            provided as HIt# with no space (e.g HI30)
                            defaults to 30 minutes
            
            outFolder -     top directory where the files will be saved in their
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
    tags = ['AIA94', 'AIA131', 'AIA171','AIA193','AIA211','AIA304','AIA335','AIA1600','AIA1700', 'C2', 'C3', 'COR1', 'COR2', 'COR1A', 'COR2A', 'COR1B', 'COR2B', 'EUVI171', 'EUVI195', 'EUVI284', 'EUVI304', 'EUVI171A', 'EUVI195A', 'EUVI284A', 'EUVI304A', 'EUVI171B', 'EUVI195B', 'EUVI284B', 'EUVI304B', 'HI1', 'HI2', 'HI1A', 'HI2A', 'HI1B', 'HI2B', 'SOLOHI', 'SOLOHI1', 'SOLOHI2', 'SOLOHI3', 'SOLOHI4', 'WISPR', 'WISPRI', 'WISPRO']
    
    #|---- Pull the command line args ----|
    vals = sys.argv[1:]
    if vals[1] < vals[0]:
        sys.exit('Exiting pullObs, start time is after end time...')
    
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
    
    #|---- Set defaults for optional args ----|
    EUVt = 10
    CORt = 20
    HIt  = 30
    outFolder = 'pullFolder/'
    
    #|---- Check back end for optional args ----|
    toRm = 0
    for i in [-1,-2,-3,-4]:
        if len(vals) > np.abs(i):
            val = vals[i]
            if 'EUVT' in val.upper():
                EUVt = int(val.upper().replace('EUVT', ''))
                toRm += 1
            elif 'CORT' in val.upper():
                CORt = int(val.upper().replace('CORT', ''))
                toRm += 1
            elif 'HIT' in val.upper():
                HIt = int(val.upper().replace('HIT', ''))
                toRm += 1
            else:
                if val.upper() not in tags:
                    if os.path.isdir(val):
                        outFolder = val
                        toRm += 1
                    else:
                        sys.exit('Cannot find output folder '+val +'. Possibly missing a t in a time tag?')
                    
    # |---- Remove the optional args ----|               
    if toRm > 0:
        vals = vals[:-toRm]
    
    insts = []
    for val in vals:
        if val.upper() in tags:
            insts.append(val.upper().replace('SOLO', 'Solo'))  # adjust SoloHi format
    if len(insts) < 1:
        sys.exit('No instrument tag provided. Got nothing to show.')
        
    pullObs(times, insts, EUVtime=EUVt, CORtime=CORt, HItime=HIt, outFolder=outFolder)    
            
        

if __name__ == '__main__':
    commandLineWrapper()
            