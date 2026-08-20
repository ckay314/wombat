"""
Helper script to make a movie from a wombat log file

python3 wombatMovie.py

Unlike the other wombat scripts this does not support direct
command line interfacing. There are enough parameters that it is
best to just edit the top portion of the file instead of trying to
pass them all via tags. This is all done in the 'Configure here!!!'
section

!!! Note that this is set to save movies using ffmpeg (even tho there
is no explict import of ffmpeg). For this to work, at least on Mac,
ffmpeg must be installed via brew not pip. If you are getting codec
errors try uninstalling the pip version then using brew !!!

"""

#|---------------------------------------|
#|---------------------------------------|
#|---------- Configure here!!! ----------|
#|---------------------------------------|
#|---------------------------------------|
global doClean, customOrder, ovw

# Name of wombat log file
logFilePath = 'wbOutputs/201207full.txt'

# Movie save name. Should end with .mp4 (other formats untested)
movieName = '2012_COR.mp4'

# Lines to do, same string format as other wombat functions
idstr = '8-75'

# Time Resolution (in minutes)
tRes = 10

# Number of columns in movie (max 5)
nHoriz = 2

# Include clean imgs without wf proj
doClean = False

# Frames per second
fps = 4

# Flag to set the instrument order
# (must include all the inst in the pickle)
customOrder = True
#instOrder = ['C2', 'COR2A', 'C3', 'WISPRI', 'SoloHI', 'HI1A_SR']
instOrder = ['COR1B', 'COR1A','COR2B','COR2A']
    
# Running (0) or base diff (1)
didx = 0
# Scaling mode linear(0), log(1), or sqrt(2)
sclidx = 0 
# Wireframe scatter point size
wfSize = 3

# Option to include overview plot
ovw = False

# Option to set custom colors, otherwise will use standard
# wombat GUI colors based on WF type. Custom will cycle 
# through the array in alphabetical order. Can use names 
# or html tags 
doCustomColors = False
customColors = ['#9AE630', 'cyan', 'DeepPink', 'PeachPuff', 'Gold', 'BlueViolet', 'LimeGreen']

    
# Dictionary for min/max values based on inst/scale mode
# Edit these if you want to change the levels in the movie
# order is [[min], [max]] where each is [linear, log, sqrt]
aiamm = [[0,0,0], [191, 191, 191]]    # [[0,0,0], [191, 191, 191]]
euimm = [[63,67,32], [150, 230, 191]] # [63,67,32], [150, 230, 191]]
cormm = [[63,0,21], [191, 191, 191]]  # [[63,0,21], [191, 191, 191]]
c2mm  = [[0,0,21], [191,191,191]]     # [[0,0,21], [191,191,191]]
c3mm  = [[37,0,37], [191,191,191]]    # [[37,0,37], [191,191,191]]
himm  = [[63,0,21], [128,191,191]]    # [[63,0,21], [128,191,191]]
solomm = [[0,0,21], [128,191,191]]    # [[0,0,21], [128,191,191]]
wispmm = [[0,0,21], [128,191,191]]    # [[0,0,21], [128,191,191]]








#|----------------------------------------------|
#|----------------------------------------------|
#|--------------- No touching!!! ---------------|
#|----------------------------------------------|
#|----------------------------------------------|
# Standard use should not involved editing 
# anything below here
#|--- Imports ---|
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
import pickle
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


sys.path.append('wombatCode/') 
from wombatLoadCTs import *
import wombatPlots as wp
import wombatWF as wf
from wombatGUI import pts2proj

# Dump the min max in a dictionary to match inst tags
MMdict = {'AIA94':aiamm, 'AIA131':aiamm, 'AIA171':aiamm,'AIA193':aiamm,'AIA211':aiamm,'AIA304':aiamm,'AIA335':aiamm,'AIA1600':aiamm,'AIA1700':aiamm, 'C2':c2mm, 'C3':c3mm, 'COR1':cormm, 'COR2':cormm, 'COR1A':cormm, 'COR2A':cormm, 'COR1B':cormm, 'COR2B':cormm, 'EUI174':aiamm, 'EUI304':aiamm, 'EUVI171':aiamm, 'EUVI195':aiamm, 'EUVI284':aiamm, 'EUVI304':aiamm, 'EUVI171A':aiamm, 'EUVI195A':aiamm, 'EUVI284A':aiamm, 'EUVI304A':aiamm, 'EUVI171B':aiamm, 'EUVI195B':aiamm, 'EUVI284B':aiamm, 'EUVI304B':aiamm, 'HI1':himm, 'HI2':himm, 'HI1A':himm, 'HI2A':himm, 'HI1B':himm, 'HI2B':himm, 'HI1A_SR':himm, 'HI1B_SR':himm, 'HI2A_SR':himm, 'HI2B_SR':himm, 'SOLOHI':solomm, 'SOLOHI1':solomm, 'SOLOHI2':solomm, 'SOLOHI3':solomm, 'SOLOHI4':solomm, 'WISPR':wispmm, 'WISPRI':wispmm, 'WISPRO':wispmm, 'WISPR_LW':wispmm, 'WISPRI_LW':wispmm, 'WISPRO_LW':wispmm, 'WISPR_L3':wispmm, 'WISPRI_L3':wispmm, 'WISPRO_L3':wispmm}

#|----------------------------------|
#|--- ID string to integer array ---|
#|----------------------------------|
def processIDstring(idstr):
    """
    Helper function that converts a string into an
    array of integer line numbers
    
    Input:
        idstr: an string with a single integer or a
               set of ints separated by + and/or -
    
    Output:
        ids:   an array of integers
    
    Examples:
        '1+2' -> [1,2]
        '1-5' -> [1,2,3,4,5]
        '1-3+5+8-10' -> [1,2,3,5,8,9,10]
    """
    # String includes a range
    if '-' in idstr:
        # Range and an add!
        if '+' in idstr:
            # Split by adds
            chunks = idstr.split('+')
            ids = []
            # Check each chunk
            for chunk in chunks:
                if '-' in chunk:
                    splitstr = chunk.split('-')
                    theseIds = np.arange(int(splitstr[0]), int(splitstr[1])+1,1, dtype=int)
                    ids.extend(theseIds)
                else:
                    try:
                        myId = int(chunk)
                        ids.extend(chunk)
                    except:
                        sys.exit('Error processing id string')
        # Just a range
        else:
            splitstr = idstr.split('-')
            if len(splitstr) > 2:
                sys.exit('Cannot process ids with multiple -')
            ids = np.arange(int(splitstr[0]), int(splitstr[1])+1,1, dtype=int)
    # Just an add
    elif '+' in idstr:
        splitstr = idstr.split('+')
        ids = []
        for aStr in splitstr:
            try:
                ids.append(int(aStr))
            except:
                print ('Error in converting id string to individual ids. Error at', aStr)
                sys.exit()                
    # Just an int (or fail for a letter)
    else:
        try:
            ids = [int(idstr)]
        except:
            print ('Error in converting id string to individual ids. Error from', idstr)
            sys.exit()
    return ids

#|-------------------------------|
#|--- WF points to projection ---|
#|-------------------------------|
def getScatterPoints(mySatStuff, myPoints):
    """
    Helper function largely lifted from pieces of 
    WOMBAT. Takes the satellite pointing information
    and converts the wireframe points into the 
    projected pixels
    
    Inputs:
        mySatStuff - a satStuff (wombat header dictionary)
                     for the instrument of interest
    
        myPoints - an array of wireframe points from wf.points
    
    Output:
        plotPoints - an array of pixel locations [x,y] for the points
                     given in myPoints
    
    """
    #|--- Get parameters needed for projection ---|
    # Get satellite position
    obs = mySatStuff['POS']
     # Scale btwn pix and arcsec
    obsScl = [mySatStuff['SCALE'], mySatStuff['SCALE']]
    if mySatStuff['OBSTYPE'] == 'HI':
        obsScl = [mySatStuff['SCALE'] * 3600, mySatStuff['SCALE'] * 3600]
    # Occulter info
    if 'OCCRARC' in mySatStuff:
        occultR = mySatStuff['OCCRARC']
    else:
        occultR = None
    # WCS info    
    mywcs  = mySatStuff['WCS']
    
    allxs = []
    allys = []  
    #|--- Loop through WF points ---|
    for jj in range(len(myPoints[:,0])):
        # Convert Cart to Sph
        pt = myPoints[jj,:]
        r = np.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2)
        if r != 0:
            lat = np.arcsin(pt[2]/r) * 180/np.pi
        else:
            lat = 0
        lon = np.arctan2(pt[1],pt[0]) * 180 / np.pi
        pt = [lat, lon, r*7e8]
         
        # WISPR outer (at least) has issues in pts projection when CME
        # is behind the satellite. The projection code matches IDL so unclear
        # what the original issue is but not a porting issue. Work around
        # by just checking if a point is behind the sat lon   
        if 'WISPR' in mySatStuff['MYTAG']:              
            dLon = lon - obs[1]
            if dLon < -180:
                dLon +=360
            if dLon > 0:
                myPt = pts2proj(pt, obs, obsScl, mywcs,  occultR=occultR)
            else:
                myPt = []
                
        # Just calc all non wispr cases        
        else:
            myPt = pts2proj(pt, obs, obsScl, mywcs, occultR=occultR)
                
        # If the point is in the FoV add it to draw    
        if len(myPt) > 0:   
            allxs.append(myPt[0][0])      
            allys.append(myPt[0][1])
    plotPts = np.transpose(np.array([allxs,allys]))    
    return plotPts

#|--------------------|
#|--- Figure Setup ---|
#|--------------------|
def setupFigure(nInsts, nHoriz, doClean, ovw, bigOVW=True):
    """
    Helper function to set up the figure and axes. Hiding this
    away in a function because a little bit of effort sorting 
    out sizes and whatnot based on user specifications.
    
    Inputs:
        nInsts - the number of instruments/panels to include 
        
        nHoriz - the number of columns in the figure. if nInsts is
                 greater than nHoriz it will make multiple rows
    
        doClean - flag to include 'clean' versions of the obs where
                  there are no wf points plotted on top. these will
                  be shown directly under each unclean panel
    
        ovw -  flag to include the overview window showing the sat
               locations, FoVs, and projections of the wfs in the 
               equatorial plane
        
    Optional Inputs:
        bigOVW - flag to allow for a large ovw that takes up 2x2
                 panels instead of 1x1 like the observations. it's
                 only shown large if the layout is such to have a 
                 2x2 gap available
    
    Outputs:
        fig - the figure object
    
        allAx - an array of 1d axes arrays in the form [axesI, axesC, axesO]
                where axesI are the main instrument panels (with wf proj)
                axesC are the clean panels (None if not used) and axesO is
                the overview window panel (None if not used)
    """
    
    # |---------------------------|
    # |--- Sort out grid sizes ---|
    # |---------------------------|
    # Check that nHoriz is < 5
    if nHoriz >5:
        sys.exit('Max of 5 for nHoriz, currently set at '+str(nHoriz))
    # Number of insts
    nInst = len(allInsts)
    # Set nHoriz at nInst if shorter to make life easy
    nHoriz = np.min([nInst,nHoriz])
    # Number of rows needed for insts
    nBot = nInst % nHoriz
    isFull = False
    if nBot == 0:
        isFull = True
        nVertI = int(nInst / nHoriz)
        nFull = nVertI
    else:
        nVertI = int(nInst / nHoriz) + 1
        nFull = nVertI - 1    
    
    # Number of rows in the grid
    if doClean:
        nVertG = 2 * nVertI
    else:
        nVertG = nVertI
        
    # |------------------------------|    
    # |--- Figure out size of ovw ---|
    # |------------------------------|    
    nHorizG = nHoriz # actual grid size, might add for ovw
    if ovw:
        ovwsize = 1 # number of cells for gridspec. will be square
        # Single row -> add to end
        if bigOVW:
            if nVertI == 1:
                # Actually two rows to incl clean plot
                if doClean & (nHoriz <= 4):
                    ovwsize = 2
                # Otherwise is 1x1
            # Multiple rows of insts
            else:
                # Bottom row is full
                if isFull:
                    # Has at least two rows so don't need to check doClean
                    if nHoriz <= 4:
                        ovwsize = 2
                # Partial bottom row
                else:
                    # Need clean row for potential 2x2
                    if doClean:
                        nEmpty = nHoriz - nFull
                        if nEmpty > 1:
                            ovwsize = 2
        # Expand grid if needed
        if isFull:
            nHorizG += ovwsize
    
    # |-----------------------|    
    # |--- Make the figure ---| 
    # |-----------------------|    
    sclFig = {1:5, 2:4, 3:3.5, 4:2.5, 5:2, 6:2 }
    myFigScl = sclFig[nHorizG]
    
    fig = plt.figure(figsize = (nHorizG*myFigScl,nVertG*myFigScl))
    gs = gridspec.GridSpec(nrows=nVertG, ncols=nHorizG, figure=fig)
    
    axesI = []
    axesC = []
    axesO = None # grammatically incorrect but CK less likely to typo axis
    # integer shift to account for clean rows
    clShift = 0
    if doClean: clShift = 1
    
    # Make the full rows
    for i in range(nFull):
        for j in range(nHoriz):
            ax = fig.add_subplot(gs[i*(1+clShift),j])
            axesI.append(ax)
            if doClean:
                ax2 = fig.add_subplot(gs[1+i*(1+clShift),j])
                axesC.append(ax2)
    # Add a bottom partial row
    if not isFull:
        for j in range(nBot):
            ax = fig.add_subplot(gs[nFull*(1+clShift),j])
            axesI.append(ax)
            if doClean:
                ax2 = fig.add_subplot(gs[1+nFull*(1+clShift),j])
                axesC.append(ax2)
    # Add ovw
    if ovw:
        if isFull:
            axesO = fig.add_subplot(gs[:ovwsize,nHoriz:nHoriz+ovwsize])
        else:
            axesO = fig.add_subplot(gs[-ovwsize:,-ovwsize:])
        axesO.set_axis_off()
        axesO.set_aspect('equal')
    
    # Turn off axes and set equal aspect
    for axSet in [axesI,axesC]:
        for ax in axSet: 
            ax.set_axis_off()
            ax.set_aspect('equal')
    
    fig.set_facecolor('k')   
    plt.subplots_adjust(wspace=0.0, hspace=0.0,left=0,right=1,bottom=0,top=1)
         
    allAx = [axesI, axesC, axesO]
    
    return fig, allAx
            
#|-------------------|
#|--- Color Setup ---|
#|-------------------|
def setUpCMaps(allInsts):
    """
    Helper function that calls check4CT from wombatLoadCTs
    and converts the results into color maps for the figure
    to use
    
    Input:
        allInsts - an array of the instrument names
    
    Output:
        instCMaps - a dictionary with the color map for each inst
    
        instMinMax - a dictionary with the [min, max] contour values
    """
    # Set up color maps and min/max for each inst
    instCMaps = {}
    instMinMax = {}
    for key in allInsts:
        # Get the color table
        fakeIt = {}
        fakeIt['OBS'] = wp.inst2sat[key.upper()]
        fakeIt['INST'] = key
        hasCT = check4CT(fakeIt)
        if type(hasCT) == type(None):
            hasCT = 'gray'
        else:
            normalized_colors = [(r/255, g/255, b/255) for r, g, b in hasCT]        
            hasCT = LinearSegmentedColormap.from_list('mygrad',normalized_colors)
        instCMaps[key] = hasCT
        myMMs = MMdict[key.upper()]
        instMinMax[key] = [myMMs[0][sclidx], myMMs[1][sclidx]]
        # Set to log if EUV
        if ('AIA' in key.upper()) or ('EUI' in key.upper()) or ('EUVI' in key.upper()):
            instMinMax[key] = [myMMs[0][1], myMMs[1][1]]
    return instCMaps, instMinMax
        
def setupWFcolors(theWFs, customColors=None):
    """
    Helper function to get the wireframe colors
    
    Input:
        theWFs - a list of wf names
    
    Optional Input:
        customColors - a list of custom colors to use instead
                       of using the wombat defaults by wf type
                       (defaults to None -> wombat colors)
    
    Output:
        wfColorDict - a dictionary mapping wf name to a color
    
    """
    # Set up colors
    wfColorDict = {}
    nWFs = len(theWFs)
    if type(customColors) != type(None):
        if len(customColors) < nWFs:
            sys.exit('Given fewer custom colors than number of wireframes. Fix at top of movie script.')
        for i in range(nWFs):
            wfColorDict[theWFs[i]] = customColors[i]
    else:
        for i in range(nWFs):
            myType = fullWF2type[theWFs[i]] # take off last character -> should be good for most cases
            wfColorDict[theWFs[i]] = wf.colorDict[myType]
    return wfColorDict
    
#|--------------------|
#|--- Time Mapping ---|
#|--------------------|
def setupTimes(lines, logFile, proIms):
    """
    Function that sorts out everything related to time for
    the movie. It determines the time range, the individual 
    time steps, and maps from observation time to movie time
    
    Input:
        lines - the integer ids of the lines in the logFile that the movie
                will use
    
        logFile - the contents of the wombat log file opened using genfromtxt
    
        proIms - the process images structure from the wombat pickle
    
    Output:
        timesByInst - a dictionary with an array containing the observation times f
                      for each instrument
                      e.g. timesByInst[inst][time1, time2 ] where times are datetimes
    
        pidx2params - a dictionary for each instrument that maps each pickle index
                      to the corresponding wf parameters at that time
                      e.g. pidx2params[inst][pickleIdx][wfType] = [params]
    
        tMovie - an array of datetime objects corresponding to the movie frames
    
        movie2inst - a dictionary mapping from movie index to pickle index for each inst
    
    """
    # Package by inst, figure out min/max time
    timesByInst  = {}
    pidx2params   = {}
    minTime = datetime.datetime(3000,1,1)
    maxTime = datetime.datetime(1000,1,1)
    nLines = len(lines)
    for j in range(nLines):
        i = lines[j]
        myInst = logFile[i,1]
        myTime = datetime.datetime.strptime(logFile[i,2], "%Y-%m-%dT%H:%M:%S" )
        myWFtype = logFile[i,3]
        myParams = logFile[i,4:13]
        myParams = myParams[myParams != 'None'].astype(float)
        mypidx   = int(logFile[i,14])
        if myInst not in timesByInst:
            timesByInst[myInst]  = []
            pidx2params[myInst]   = {}
        if mypidx not in pidx2params[myInst]:
            pidx2params[myInst][mypidx] = {}
        timesByInst[myInst].append(myTime)
        pidx2params[myInst][mypidx][myWFtype] = myParams
    
        # Track early/late time
        if myTime < minTime: minTime = myTime
        if myTime > maxTime: maxTime = myTime

    # Start at rounded minTime
    startTime = datetime.datetime(minTime.year, minTime.month, minTime.day, minTime.hour)     
    fracUnder = (minTime-startTime).total_seconds()/60 / tRes
    if fracUnder > 1:
        startTime = startTime + datetime.timedelta(seconds=int(fracUnder)*tRes*60)

    # Figure out number of movie steps
    fracNtimes = (maxTime - startTime).total_seconds()/60/tRes
    nTimes = int(fracNtimes) + 1 # add one bc edges = intervals +1
    # Figure out if need to add one for last time (took int but might be over half dt away)
    if fracNtimes - nTimes > 0.5:
        nTimes += 1
    tMovie = []
    for i in range(nTimes):
        tMovie.append(startTime + datetime.timedelta(seconds=i*tRes*60))
    dtMovie = np.array([i*tRes for i in range(nTimes)])    
    allPtimes = {}
    allPdts = {}
    for key in timesByInst:
        allPtimes[key] = []
        allPdts[key] = []
        for i in range(len(proIms[key][0])):
            allPtimes[key].append(proIms[key][0][i].date.datetime)
            allPdts[key].append((allPtimes[key][-1] - allPtimes[key][0]).total_seconds()/60. )
            
    # Map movie to instrument idx and pidx
    movie2inst  = {}
    for key in timesByInst:
        movie2inst[key]  = []
        my0diff = (allPtimes[key][0] - tMovie[0]).total_seconds()/60.
        shiftDiff = np.array(allPdts[key]) + my0diff
        for i in range(nTimes):
            tdiff = np.abs(shiftDiff - dtMovie[i])         
            idx = np.where(tdiff == np.min(tdiff))[0][0]
            movie2inst[key].append(idx)
    
    return timesByInst, pidx2params, tMovie, movie2inst
    
#|---------------------|
#|--- General Setup ---|
#|---------------------|
def checkSetup(lines, logFile):
    """
    Function to check that everything is correct in the input
    parameters and returns some basic settings
    
    Input:
        lines - the integer ids of the lines in the logFile that the movie
                will use
    
        logFile - the contents of the wombat log file opened using genfromtxt
    
    Output:
        allInsts - a list of all the instruments to include. It will be in the
                   custom order if that is provided in the user settings
    
        allPickles - a list of all the pickles to open. will be a single pickle
                     right now because that is all that is supported
    
        theWFs - a list of all the wireframes to add in the figure. this will 
                 be the tags as provided in logFile which are a combination of
                 the wfType and an identifying tag (e.g. GCS1)
    
        fullWF2type - a dictionary mapping the full name in theWFs to the wf type
                      that wombat needs to create a wf structure
    
    
    """
    allInsts = np.unique(logFile[lines,1])
    nInsts = len(allInsts)
    
    # Check if given custom order
    if customOrder:
        if np.array_equal(np.sort(instOrder), np.sort(allInsts)):
            allInsts = instOrder
        else:
            print ('Cannot match custom instrument order:')
            print ('   ', instOrder)
            print ('To instruments from log file: ')
            print ('   ', allInsts)
            sys.exit('Exiting movie script')
            
    allPickles = np.unique(logFile[lines,13])
    # Check if need to combine pickles
    if len(allPickles) != 1:
        pickleDict = {}
        # Check if each inst only has one pickle
        for inst in allInsts:
            myIds = np.where(logFile[lines,1] == inst)[0]
            myPickles = np.unique(logFile[lines[myIds],13])
            if len(myPickles) > 1:
                sys.exit('Cannot combine multiple pickles for the same instrument. Error for '+inst)
            else:
                if myPickles[0] in pickleDict:
                    pickleDict[myPickles[0]].append(inst)
                else:
                    pickleDict[myPickles[0]] = [inst]
        
        # Combine them if didn't hit exit        
        bkgData = {}
        bkgData['proImMaps'] = {}
        bkgData['scaledIms'] = {}
        bkgData['satStuff'] = {}
        for aPick in pickleDict:
            with open(aPick, 'rb') as file:
                thisData = pickle.load(file)
            for aInst in pickleDict[aPick]:
                bkgData['proImMaps'][aInst] = {}
                bkgData['scaledIms'][aInst] = {}
                bkgData['satStuff'][aInst]  = {}
                bkgData['proImMaps'][aInst] = thisData['proImMaps'][aInst]
                bkgData['scaledIms'][aInst] = thisData['scaledIms'][aInst]
                bkgData['satStuff'][aInst]  = thisData['satStuff'][aInst]
      
    else:
        with open(allPickles[0], 'rb') as file:
            bkgData = pickle.load(file)   
        
            
        
    nLines = len(logFile[lines,0])
    theWFs = np.unique(logFile[lines,3])
    nWFs = len(theWFs)
    
    # Check WF types 
    fullWF2type = {}
    for i in range(nWFs):
        myType = theWFs[i][:-1].replace("Half", 'Half ') # take off last character -> should be good for most cases
        if myType in wf.colorDict:
            fullWF2type[theWFs[i]] = myType
        elif myType[:-1] in wf.colorDict: # try one more in case has two digit number
            fullWF2type[theWFs[i]] = myType[:-1]
        else:
            print ('Unknown WF type:', theWFs[i])
            print('Needs to be an existing WOMBAT WF type with no more than two additional')
            print('identifying characters at the end (e.g. typeX or typeXX)')
            sys.exit()
    
    
    return allInsts, allPickles, bkgData, theWFs, fullWF2type

#|-------------------------|
#|--- Plot Object Setup ---|
#|-------------------------|
def setupImgObj():
    """
    Function to initialize all the objects in the figure. It set things
    at the earliest time step values. The objects will be used in the
    update function for animation. There are no direct inputs but it does
    make use of all the global variables set before calling it
    
    Outputs:
        imObjs - an array of all the imshow object for each main instrument panel
    
        imObjsC - an array of all the imshow object for each clean instrument panel
    
        textObjs - an array of the text objects that display the inst name/obs time
    
        scatObjs - a dictionary with entries for each instrument. the entry is an
                   array with one scatter object for each wf that appears in the movie
    
    """
    movt = 0 # set time at 0
    # Setup holders
    imObjs = []
    imObjsC = []
    textObjs = []
    scatObjs = {}
    
    # Loop through instruments
    for i in range(nInsts):
        myInst = allInsts[i]
        instIdx = movie2inst[myInst][movt]
        mm = instMinMax[myInst]
        mySclIm = sclIms[myInst][didx][instIdx][sclidx]
        # Force log scale if euv
        if (satStuff[myInst][0][0]['OBSTYPE'] == 'EUV'):
            mySclIm = sclIms[myInst][didx][instIdx][1]
        #|--- Set up main image objects ---|
        imObj = allAx[0][i].imshow(mySclIm, cmap=instCMaps[myInst], vmin=mm[0], vmax=mm[1], origin='lower')
        imObjs.append(imObj)
        mydate =  proIms[myInst][0][instIdx].date.datetime.strftime("%Y-%m-%dT%H:%M")
        panelLabel = myInst + ' ' + mydate

        #|--- Set up clean img objects and text label ---|
        if doClean:
            imObjC = allAx[1][i].imshow(mySclIm, cmap=instCMaps[myInst], vmin=mm[0], vmax=mm[1], origin='lower')
            imObjsC.append(imObjC)
            textObj = allAx[1][i].text(0.5, 0, panelLabel, c='w', bbox=dict(facecolor='black', alpha=0.5), horizontalalignment='center',verticalalignment='bottom', transform = allAx[1][i].transAxes)
        else:
            textObj = allAx[0][i].text(0.5, 0, panelLabel, c='w', bbox=dict(facecolor='black', alpha=0.5), horizontalalignment='center',verticalalignment='bottom', transform = allAx[0][i].transAxes)
        textObjs.append(textObj)

        # Set up dummy scatter objects. One for each wf for each inst
        scatObjs[myInst] = []
        for j in range(nWFs):
            scatObj = allAx[0][i].scatter(0,0, c=wfColorDict[theWFs[j]], s=0, zorder=20)
            scatObjs[myInst].append(scatObj)
        
        #|--- Get WF scatter points in pixels ---|
        if instIdx in pidx2params[myInst]:        
            #|--- Convert WF points to proj ---|
            for j in range(nWFs):
                if theWFs[j] in pidx2params[myInst][instIdx]:
                    nowPs = pidx2params[myInst][instIdx][theWFs[j]]
                    awf = wf.wireframe(fullWF2type[theWFs[j]])
                    awf.params = nowPs
                    awf.getPoints()
                    plotPts = getScatterPoints(satStuff[myInst][didx][instIdx], awf.points)
                    scatObjs[myInst][j].set_offsets(plotPts)
                    scatObjs[myInst][j].set_sizes(wfSize*np.ones(plotPts.shape[0]))   
                     
    return imObjs, imObjsC, textObjs, scatObjs

def setupOVW(): 
    """
    Function to initialize all the objects in the figure. It set things
    at the earliest time step values. The objects will be used in the
    update function for animation. There are no direct inputs but it does
    make use of all the global variables set before calling it
    
    Outputs:
        ovwScats - an array of scatter objects for the wireframes
        
        satScats - an array of arrays of plot objects related to the satellites
                   it contains [scatterPoint, FoVline1, FoVline2, FoVobject, text]
                   where each item is an array for all satellites
    
        timeItem - the time label object
    
    """    
    movt = 0
    ovwScats = []
    satScats = [[], [], [], [], [], []] # [line1, line2, fill, text] for each sat
    satStr   = [] # temp holder to make sure no duplicate sat names
    L1counter = 0
    
    #|---- Create the sun and earth (nbd) ----|
    # These are const, don't need to save plot objects
    twopi = np.linspace(0, 2.01*np.pi, 200)
    x_data = np.cos(twopi)
    y_data = np.sin(twopi)
    allAx[2].plot(x_data, y_data, 'w', lw=1, zorder=0)    
    allAx[2].scatter([0],[0], c='y', s=50, zorder=10)
    allAx[2].scatter([0],[-1], c='DeepSkyBlue', s=50, zorder=2)
    
    # Add the general time on the ovw
    theTime = tMovie[movt].strftime("%Y-%m-%dT%H:%M")
    timeItem = allAx[2].text(0.02, 0.98, theTime, c='w', horizontalalignment='left',verticalalignment='top', transform = allAx[2].transAxes)  
    
    #|---- Create wireframe scatters ----|
    for j in range(nWFs):
        scatObj = allAx[2].scatter(0,0, c=wfColorDict[theWFs[j]], s=0, zorder=10)
        ovwScats.append(scatObj)  
    
    #|---- Create/plot satellite scatters ----|
    for i in range(nInsts):
        myInst = allInsts[i]
        instIdx = movie2inst[myInst][movt]
        
        # Sat scatter dots
        myPos  = satStuff[myInst][0][instIdx]['POS']
        myR = myPos[2] / 1.496e+11 
        myLon = myPos[1] * np.pi / 180.
        y = - myR * np.cos(myLon)
        x = myR * np.sin(myLon)
        satScat = allAx[2].scatter(x,y, c='w', s=20, zorder=3)
        satScats[0].append(satScat)
        
        # Sat FoVs
        # Line 1
        myPoint = satStuff[myInst][0][instIdx]['POINTING'][1]
        xPt1 = myPoint[1] 
        yPt1 = -myPoint[0]
        curve1, = allAx[2].plot([x, xPt1], [y, yPt1], 'w', lw=0.5, zorder=2)
        satScats[1].append(curve1)
        # Line 2
        myPoint = satStuff[myInst][0][instIdx]['POINTING'][2]
        xPt2 = myPoint[1] 
        yPt2 = -myPoint[0]
        curve2, = allAx[2].plot([x, xPt2], [y, yPt2], 'w', lw=0.5, zorder=2)
        satScats[2].append(curve2)
        # Fill
        xA, yA = np.array([x, xPt1]), np.array([y, yPt1])
        xB, yB = np.array([x, xPt2]), np.array([y, yPt2])
        xClosed = np.append(xA, xB[::-1])
        yClosed = np.append(yA, yB[::-1])
        fillIt = allAx[2].fill(xClosed, yClosed, color='blue', alpha=0.25, zorder=0)[0]
        satScats[3].append(fillIt)
        
        # Sat Name (not inst)
        myName = satStuff[myInst][0][instIdx]['SHORTNAME']
        if myName not in satStr:
            satStr.append(myName)
            # Figure out where to place text
            # inner cases
            if myR < 0.8:
                if x < 0:
                    xsat = x - 0.07
                else:
                    xsat = x + 0.07
                if y < 0:
                    ysat =  y - 0.1
                else:
                    ysat = y +0.05
            # L1 cases
            elif np.abs(myLon) < np.pi / 18:
                xsat = myR * np.sin(myLon)
                if L1counter == 0:
                    ysat = -1.05
                else:
                    ysat = -0.95 + L1counter * 0.1
                L1counter +=1
            # Other cases prob ok with this?               
            else:
                if x < 0:
                    xsat = x - 0.07
                else:
                    xsat = x + 0.07
                if y < 0:
                    ysat =  y - 0.1
                else:
                    ysat = y +0.05
            textItem = allAx[2].text(xsat, ysat, myName, c='w')    
            myLat = '{:.1f}'.format(myPos[0])
            ywid = 0.03
            if nInsts <= nHoriz: ywid = 0.05
            latItem = allAx[2].text(0.98, 0.98 - ywid*(len(satStr)-1), myName +': '+myLat+'$^{\\circ}$', c='w', horizontalalignment='right',verticalalignment='top', transform = allAx[2].transAxes)  
            
        else:
            textItem = None
            latItem = None
        satScats[4].append(textItem)
        satScats[5].append(latItem)
                    
    
    for j in range(nWFs):
        # Figure out if any inst has a set of params for this time
        # This will overwrite vals if diff insts have diff values 
        # but the small diffs shouldn't matter on ovw plot scale
        myParams = None
        for i in range(nInsts):
            myInst = allInsts[i]
            instIdx = movie2inst[myInst][movt] 
            if instIdx in pidx2params[myInst]: 
                if theWFs[j] in pidx2params[myInst][instIdx]:
                    myParams = pidx2params[myInst][instIdx][theWFs[j]]
        if type(myParams) != type(None):
            awf = wf.wireframe(fullWF2type[theWFs[j]])
            awf.params = myParams
            awf.getPoints()
            # Downselect project, or not based on type
            if awf.WFtype not in ['Sphere', 'Half Sphere', 'Ellipse', 'Half Ellipse']:
                myxs = -awf.points[::2,0] / 215
                myys = awf.points[::2,1]  / 215
            else:
                myxs = -awf.points[:,0] / 215
                myys = awf.points[:,1]  / 215
            ovwScats[j].set_offsets(np.transpose(np.array([myys,myxs])))
            ovwScats[j].set_sizes(int(0.5*wfSize)*np.ones(len(myxs)))
            
    return ovwScats, satScats, timeItem
    
#|------------------------|
#|--- Animation Update ---|
#|------------------------|
def update(movt):
    """
    Function passed to the animator which will update the plot based on time index. 
    
    This is essentially a copy of the setupImgObj and setupOVW functions with all
    the object saving portions cut out
    """
    for i in range(nInsts):
        myInst = allInsts[i]
        instIdx = movie2inst[myInst][movt]    
        didx = 0
        sclidx = 0
        mySclIm = sclIms[myInst][didx][instIdx][sclidx]
        # Force log scale if euv
        if (satStuff[myInst][0][0]['OBSTYPE'] == 'EUV'):
            mySclIm = sclIms[myInst][didx][instIdx][1]
        mydate =  proIms[myInst][0][instIdx].date.datetime.strftime("%Y-%m-%dT%H:%M")
        panelLabel = myInst + ' ' + mydate
        
        imObjs[i].set_data(mySclIm)
        textObjs[i].set_text(panelLabel)
        if doClean:
            imObjsC[i].set_data(mySclIm)
            
        #|--- Get WF scatter points in pixels ---|
        if instIdx in pidx2params[myInst]:     
            #|--- Convert WF points to proj ---|
            for j in range(nWFs):
                if theWFs[j] in pidx2params[myInst][instIdx]:
                    nowPs = pidx2params[myInst][instIdx][theWFs[j]]
                    awf = wf.wireframe(fullWF2type[theWFs[j]])
                    awf.params = nowPs
                    awf.getPoints()
                    plotPts = getScatterPoints(satStuff[myInst][didx][instIdx], awf.points)
                    scatObjs[myInst][j].set_offsets(plotPts)
                    scatObjs[myInst][j].set_sizes(wfSize*np.ones(plotPts.shape[0]))
                else:
                    scatObjs[myInst][j].set_offsets([0,0])
        else:
            # Clean it out if we don't have a fit
            for j in range(nWFs):
                scatObjs[myInst][j].set_offsets([0,0])
    if ovw:
        theTime = tMovie[movt].strftime("%Y-%m-%dT%H:%M")
        timeItem.set_text(theTime)  
                
        # |--- Update satellites ---|
        satStr = []
        L1counter = 0
        for i in range(nInsts):
            myInst = allInsts[i]
            instIdx = movie2inst[myInst][movt]
        
            myPos  = satStuff[myInst][0][instIdx]['POS']
            myR = myPos[2] / 1.496e+11 
            myLon = myPos[1] * np.pi / 180.
            y = - myR * np.cos(myLon)
            x = myR * np.sin(myLon)
            satScats[0][i].set_offsets([x,y])
            
            # Line 1
            myPoint = satStuff[myInst][0][instIdx]['POINTING'][1]
            xPt1 = myPoint[1] 
            yPt1 = -myPoint[0]
            satScats[1][i].set_data([x, xPt1], [y, yPt1])
            # Line 2
            myPoint = satStuff[myInst][0][instIdx]['POINTING'][2]
            xPt2 = myPoint[1] 
            yPt2 = -myPoint[0]
            satScats[2][i].set_data([x, xPt2], [y, yPt2])
            # Fill
            xA, yA = np.array([x, xPt1]), np.array([y, yPt1])
            xB, yB = np.array([x, xPt2]), np.array([y, yPt2])
            xClosed = np.append(xA, xB[::-1])
            yClosed = np.append(yA, yB[::-1])
            
            new_vertices = np.column_stack((xClosed, yClosed))
            satScats[3][i].set_xy(new_vertices)
            
            myName = satStuff[myInst][0][instIdx]['SHORTNAME']
            if myName not in satStr:
                satStr.append(myName)
                # inner cases
                if myR < 0.8:
                    if x < 0:
                        xsat = x - 0.07
                    else:
                        xsat = x + 0.07
                    if y < 0:
                        ysat =  y - 0.1
                    else:
                        ysat = y +0.05
                # L1 cases
                elif np.abs(myLon) < np.pi / 18:
                    xsat = myR * np.sin(myLon)
                    if L1counter == 0:
                        ysat = -1.05
                    else:
                        ysat = -0.95 + L1counter * 0.1
                    L1counter +=1
                # Other cases prob ok with this?               
                else:
                    if x < 0:
                        xsat = x - 0.07
                    else:
                        xsat = x + 0.07
                    if y < 0:
                        ysat =  y - 0.1
                    else:
                        ysat = y +0.05
                satScats[4][i].set_position((xsat, ysat))    
                
                myLat = '{:.1f}'.format(myPos[0])
                satScats[5][i].set_text(myName +': '+myLat+'$^{\\circ}$')
            
        # |--- Update wireframes ---|
        for j in range(nWFs):
            # Figure out if any inst has a set of params for this time
            # This will overwrite vals if diff insts have diff values 
            # but the small diffs shouldn't matter on ovw plot scale
            myParams = None
            for i in range(nInsts):
                myInst = allInsts[i]
                instIdx = movie2inst[myInst][movt] 
                if instIdx in pidx2params[myInst]: 
                    if theWFs[j] in pidx2params[myInst][instIdx]:
                        myParams = pidx2params[myInst][instIdx][theWFs[j]]
            if type(myParams) != type(None):
                awf = wf.wireframe(fullWF2type[theWFs[j]])
                awf.params = myParams
                awf.getPoints()
                if awf.WFtype not in ['Sphere', 'Half Sphere', 'Ellipse', 'Half Ellipse']:
                    myxs = -awf.points[::2,0] / 215
                    myys = awf.points[::2,1]  / 215
                else:
                    myxs = -awf.points[:,0] / 215
                    myys = awf.points[:,1]  / 215
                ovwScats[j].set_offsets(np.transpose(np.array([myys,myxs])))
                ovwScats[j].set_sizes(int(0.5*wfSize)*np.ones(len(myxs)))
            else:
                ovwScats[j].set_offsets([0,0])
                #ovwScats[j].set_sizes(wfSize*np.ones(len(myxs)))


#|----------------------|
#|----------------------|
#|--- Main Procedure ---|
#|----------------------|
#|----------------------|
 
#|---------------------|
#|--- General Setup ---|
#|---------------------|
ids = processIDstring(idstr)
lines = ids - 1 # txt file numbering one more than array index
logFile = np.genfromtxt(logFilePath, dtype=str)

# |--- Check the settings ---|    
global allInsts, allPickles, theWFs, fullWF2type, nInsts, nWFs
allInsts, allPickles, bkgData, theWFs, fullWF2type = checkSetup(lines, logFile)
nInsts = len(allInsts)
nWFs = len(theWFs)

# |--- Set up figure ---|
global fig, allAx
fig, allAx = setupFigure(len(allInsts), nHoriz, doClean, ovw)    

    
# |--- Open/unpackage the pickle ---|
global proIms, sclIms, satStuff
proIms = bkgData['proImMaps']
sclIms = bkgData['scaledIms']
satStuff = bkgData['satStuff']

#|--- Map out times ---|
global timesByInst, pidx2params, tMovie, movie2inst, nTimes
timesByInst, pidx2params, tMovie, movie2inst = setupTimes(lines, logFile, proIms)
nTimes = len(tMovie)
    
#|--- Set up color maps ---|
global instCMaps, instMinMax
instCMaps, instMinMax = setUpCMaps(allInsts) 

#|--- Set up wf colors ---|
global wfColorDict
if not doCustomColors:
    customColor = None
wfColorDict = setupWFcolors(theWFs, customColors=customColors)

#|--------------------|
#|--- Object Setup ---|
#|--------------------|
global imObjs, imObjsC, textObjs, scatObjs
imObjs, imObjsC, textObjs, scatObjs = setupImgObj()
if ovw:
    global ovwScats, satScats, timeItem 
    ovwScats, satScats, timeItem = setupOVW()
                
#|------------------|
#|--- Animate it ---|
#|------------------|
intv = 1 / fps * 1000 # interval in milliseconds
ani = animation.FuncAnimation(fig=fig, func=update, frames=nTimes, interval=intv)
#plt.show()
ani.save(movieName, writer='ffmpeg')

