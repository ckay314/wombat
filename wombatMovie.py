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

# Python needs ffmpeg installed via brew, just using pip does
# not work for some random reason


#|-------------------------|
#|--- Configure here!!! ---|
#|-------------------------|
# Name of wombat log file
#logFilePath = 'wbOutputs/2303full_CME1b.txt'
logFilePath = 'tempLog.txt'
logFilePath = 'wbOutputs/201207pretty.txt'

# Movie save name
#movieName = '2023full.mp4'
movieName = '2012full.mp4'

# Lines to do
#lines = range(41) # Replace with string reading code from other files
#lines = range(410)
lines=np.arange(40, 160)-1
# Time Resolution (in minutes)
tRes = 30 
# Plot shape (max of 5 horiz)
nHoriz = 2
# Include clean imgs without wf proj
doClean = False
# Instrument order
customOrder = False
instOrder = ['C2', 'COR2A', 'C3', 'WISPRI', 'SoloHI', 'HI1A_SR']
    
# Running (0) or base diff (1)
didx = 0
# Scaling mode linear(0), log(1), or sqrt(2)
sclidx = 0 
# Wireframe scatter point size
wfSize = 3

# Option to include overview plot
ovw = True

# Option to set custom colors, otherwise will use standard
# wombat GUI colors based on WF type. Custom will cycle 
# through the array in alphabetical order 
doCustomColors = False
customColors = ['#9AE630', 'cyan', 'DeepPink', 'PeachPuff', 'Gold', 'BlueViolet', 'LimeGreen']

    
# Dictionary for min/max values based on inst/scale mode
aiamm = [[0,0,0], [191, 191, 191]]
euimm = [[63,67,32], [150, 230, 191]]
cormm = [[63,0,21], [191, 191, 191]]
c2mm  = [[0,0,21], [191,191,191]]
c3mm  = [[37,0,37], [191,191,191]]
himm  = [[63,0,21], [128,191,191]]
wilomm = [[0,0,21], [128,191,191]]

MMdict = {'AIA94':aiamm, 'AIA131':aiamm, 'AIA171':aiamm,'AIA193':aiamm,'AIA211':aiamm,'AIA304':aiamm,'AIA335':aiamm,'AIA1600':aiamm,'AIA1700':aiamm, 'C2':c2mm, 'C3':c3mm, 'COR1':cormm, 'COR2':cormm, 'COR1A':cormm, 'COR2A':cormm, 'COR1B':cormm, 'COR2B':cormm, 'EUI174':aiamm, 'EUI304':aiamm, 'EUVI171':aiamm, 'EUVI195':aiamm, 'EUVI284':aiamm, 'EUVI304':aiamm, 'EUVI171A':aiamm, 'EUVI195A':aiamm, 'EUVI284A':aiamm, 'EUVI304A':aiamm, 'EUVI171B':aiamm, 'EUVI195B':aiamm, 'EUVI284B':aiamm, 'EUVI304B':aiamm, 'HI1':himm, 'HI2':himm, 'HI1A':himm, 'HI2A':himm, 'HI1B':himm, 'HI2B':himm, 'HI1A_SR':himm, 'HI1B_SR':himm, 'HI2A_SR':himm, 'HI2B_SR':himm, 'SOLOHI':wilomm, 'SOLOHI1':wilomm, 'SOLOHI2':wilomm, 'SOLOHI3':wilomm, 'SOLOHI4':wilomm, 'WISPR':wilomm, 'WISPRI':wilomm, 'WISPRO':wilomm, 'WISPR_LW':wilomm, 'WISPRI_LW':wilomm, 'WISPRO_LW':wilomm, 'WISPR_L3':wilomm, 'WISPRI_L3':wilomm, 'WISPRO_L3':wilomm}


def getScatterPoints(mySatStuff, myPoints):
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

 


def setupFigure(allInsts, nHoriz, doClean, ovw, bigOVW=True):
    # |---------------------------|
    # |--- Sort out grid sizes ---|
    # |---------------------------|
    # Check that nHoriz is < 5
    if nHoriz >5:
        sys.exit('Max of 5 for nHoriz, currently set at '+str(nHoriz))
    # Number of insts
    nInst = len(allInsts)
    # Set nHoriz at nInst if shorter
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
    axesO = None # grammatically incorrect but CK less likely to typo it matching
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
    
    
    for axSet in [axesI,axesC]:
        for ax in axSet: 
            ax.set_axis_off()
            ax.set_aspect('equal')
    
    fig.set_facecolor('k')   
    plt.subplots_adjust(wspace=0.0, hspace=0.0,left=0,right=1,bottom=0,top=1)
         
    allAx = [axesI, axesC, axesO]
    
    return fig, allAx
            
                
        
        
        
        
    
        
        
    
 
#|------------------------------|
#|--- Read in fits, organize ---|
#|------------------------------|
logFile = np.genfromtxt(logFilePath, dtype=str)
allInsts = np.unique(logFile[lines,1])
nInsts = len(allInsts)

# Calc num of rows bases on nHoriz and nInsts
nVert = int(nInsts / nHoriz)+1 # need to calc this later
if doClean:
    nVert *= 2

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

# |--- Set up figure ---|
fig, allAx = setupFigure(allInsts, nHoriz, doClean, ovw)
    

allPickles = np.unique(logFile[lines,13])
if len(allPickles) != 1:
    sys.exit('Cannot combine multiple pickles (yet)')
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
        
# Set up colors
wfColorDict = {}
if doCustomColors:
    if len(customColors) < nWFs:
        sys.exit('Given fewer custom colors than number of wireframes. Fix at top of movie script.')
    for i in range(nWFs):
        wfColorDict[theWFs[i]] = customColors[i]
else:
    for i in range(nWFs):
        myType = fullWF2type[theWFs[i]] # take off last character -> should be good for most cases
        wfColorDict[theWFs[i]] = wf.colorDict[myType]


# Package by inst, figure out min/max time
timesByInst  = {}
pidx2params   = {}
minTime = datetime.datetime(3000,1,1)
maxTime = datetime.datetime(1000,1,1)
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
nTimes = int((maxTime - startTime).total_seconds()/60/tRes)
tMovie = []
for i in range(nTimes+2):
    tMovie.append(startTime + datetime.timedelta(seconds=i*tRes*60))
dtMovie = np.array([i*tRes for i in range(nTimes+2)])


# Want to get all avail imgs in the pickle, including if beyond the latest
# fit for that inst -> need to open pickles
# Open the pickle
with open(allPickles[0], 'rb') as file:
    bkgData = pickle.load(file)   
WBinfo = bkgData['WBinfo']
proIms = bkgData['proImMaps']
sclIms = bkgData['scaledIms']
satStuff = bkgData['satStuff']

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
movie2instP = {}


for key in timesByInst:
    movie2inst[key]  = []
    my0diff = (allPtimes[key][0] - tMovie[0]).total_seconds()/60.
    shiftDiff = np.array(allPdts[key]) + my0diff
    for i in range(nTimes+2):
        tdiff = np.abs(shiftDiff - dtMovie[i])         
        idx = np.where(tdiff == np.min(tdiff))[0][0]
        movie2inst[key].append(idx)

# Set up color maps and min/max for each inst
instCMaps = {}
instMinMax = {}
for key in timesByInst:
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




# Plot time zero and set up all the img object

movt = 0
imObjs = []
imObjsC = []
textObjs = []
scatObjs = {}
for i in range(nInsts):
    myInst = allInsts[i]
    instIdx = movie2inst[myInst][movt]
    mm = instMinMax[myInst]
    mySclIm = sclIms[myInst][didx][instIdx][sclidx]
    
    imObj = allAx[0][i].imshow(mySclIm, cmap=instCMaps[myInst], vmin=mm[0], vmax=mm[1], origin='lower')
    imObjs.append(imObj)
    mydate =  proIms[myInst][0][instIdx].date.datetime.strftime("%Y-%m-%dT%H:%M")
    panelLabel = myInst + ' ' + mydate

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
# Set up ovw
if ovw:
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
            if awf.WFtype not in ['Sphere', 'Half Sphere', 'Ellipse', 'Half Ellipse']:
                myxs = -awf.points[::2,0] / 215
                myys = awf.points[::2,1]  / 215
            else:
                myxs = -awf.points[:,0] / 215
                myys = awf.points[:,1]  / 215
            ovwScats[j].set_offsets(np.transpose(np.array([myys,myxs])))
            ovwScats[j].set_sizes(int(0.5*wfSize)*np.ones(len(myxs)))
            

#plt.show()
#print (sd)
    
    
def update(movt):
    for i in range(nInsts):
        myInst = allInsts[i]
        instIdx = movie2inst[myInst][movt]    
        didx = 0
        sclidx = 0
        mySclIm = sclIms[myInst][didx][instIdx][sclidx]
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
           
        
    
ani = animation.FuncAnimation(fig=fig, func=update, frames=nTimes+2, interval=150)
#plt.show()
ani.save(movieName, writer='ffmpeg')

