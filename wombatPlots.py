import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys, os
import datetime
from scipy.interpolate import UnivariateSpline


sys.path.append('wombatCode/') 
import wombatWF as wf


''' 
syntax python wPs.py type logFile ids [ min#, max#, outName, pictype]

types:
    Line profiles 
    HT - height - time (basic evol of WF params)
    Kin - height time but adding in vel/acc
    En  - add in mass + energetics
    
    Figures
    linimg
    logimg
    sqimg
    
'''

#|--- Standard input error message ---|
global errorStrings, labSwap, labelMatch, dubColor
errorStrings = [' ', '  python3 wombatPlots.py logFile id(s) type otherParams', '          where:', '          - logFile is a wombat log file', '          - ids is an integer or int+int or int-int', '          - type sets the plot type from [ht, kin, en, linimg, logimg, sqimg]', '          - otherParams includes min#, max#, outName, and picType']

# Dictionary to swap from the GUI names that dont like latex to
# nicer things for this plot
deg = '($^{\\circ}$)'
labSwap = {'Height (Rs)':'Height (R$_S$)', 'Lon (deg)':'Lon'+deg , 'Lat (deg)': 'Lat'+deg, 'Tilt (deg)':'Tilt'+deg, 'AW (deg)':'AW'+deg, 'kappa':'$\\kappa$', 'AW_FO (deg)':'AW$_{FO}$'+deg, 'AW_EO (deg)':'AW$_{EO}$'+deg, 'deltaAx':'$\\delta_{Ax}$', 'deltaCS':'$\\delta_{CS}$', 'ecc1':'$\\epsilon_1$', 'ecc2':'$\\epsilon_2$', 'Roll (deg)':'Roll'+deg, 'Yaw (deg)':'Yaw'+deg, 'Pitch (deg)':'Pitch'+deg, 'Lx (Rs)':'L$_x$ (R$_S$)', 'Ly (Rs)':'L$_y$ (R$_S$)', 'Lz (Rs)':'L$_z$ (R$_S$)', 'HeightO (Rs)':'Height$_O$ (R$_S$)', 'LonO (deg)':'Lon$_O$'+deg, 'LatO (deg)':'Lat$_O$'+deg}


# Who to pair things with, GCS* is longest, followed by Slab
# Only match nice pairs, otherwise let it dump them in wherever
labelMatch = {'Tilt (deg)':['Roll (deg)'], 'AW (deg)': ['AW_FO (deg)', 'Lx (Rs)'], 'AW_FO (deg)':['AW (deg)', 'Lx (Rs)'],  'AW_EO (deg)':['AW (deg)', 'Ly (Rs)'], 'kappa':['ecc1', 'deltaAx',  'Yaw (deg)'], 'ecc1':['kappa', 'deltaAx', 'Yaw (deg)'],  'ecc2': ['deltaCS', 'Pitch (deg)'], 'deltaAx':['kappa', 'ecc1', 'Yaw (deg)'], 'deltaCS':['ecc2', 'Pitch (deg)']}

#dubColor = '#762b99'  # color for right labels used by two+ wfs

# |-----------------------------------|
# |--- Get kinematics from results ---|
# |-----------------------------------|
def getKinematics(wombatRes, wfTypes, polyDeg=3, r1=5, r2=10):
    # |--- Make holders ---|
    times = {}
    dts   = {}
    heights = {}
    for awf in wfTypes:
        times[awf]   = []
        heights[awf] = []
        dts[awf] = []
    
    # |--- Collect things ---|
    for aInst in wombatRes.keys():
        subRes = wombatRes[aInst]
        for awf in wfTypes:
            if awf in subRes.keys():
                myRes = subRes[awf]
                myTimes = myRes['times']
                for i in range(len(myTimes)):
                    roundTime = myTimes[i].replace(second=0)
                    if roundTime not in times[awf]:
                        times[awf].append(roundTime)
                        heights[awf].append(myRes['paramGrid'][0,i])
                        
    # |--- Sort things ---|
    for awf in wfTypes:
        times[awf] = np.array(times[awf])
        heights[awf] = np.array(heights[awf])
        idxs  = np.argsort(times[awf])
        times[awf] = times[awf][idxs]
        heights[awf] = heights[awf][idxs]
        
    # |--- Get seconds from earliest time ---|  
    # also get max time while here   
    earlyT = datetime.datetime(3000,1,1)
    lateT  = datetime.datetime(1000,1,1)
    for awf in wfTypes:
        if times[awf][0] < earlyT:
            earlyT = times[awf][0]
        if times[awf][-1] > lateT:
            lateT = times[awf][-1]
    for awf in wfTypes:
        for i in range(len(times[awf])):
            dts[awf].append((times[awf][i]-earlyT).total_seconds())
        dts[awf] = np.array(dts[awf])
   
   
    # |--- Fit spline to low part of h(t) ---| 
    Hsplines = {}
    vsplines = {}
    asplines = {}
    for awf in wfTypes:
        myIdx = np.where(heights[awf] <= r2)[0]
        if len(myIdx) > 4:
            Hsplines[awf] = UnivariateSpline(dts[awf][myIdx], heights[awf][myIdx])
            vsplines[awf] = Hsplines[awf].derivative(n=1) 
            asplines[awf] = Hsplines[awf].derivative(n=2) 
        else:
            Hsplines[awf] = None
            vsplines[awf] = None
            asplines[awf] = None
            
    # |--- Fit poly to high part of h(t) ---| 
    Hpolys = {}
    vpolys = {}
    apolys = {}
    for awf in wfTypes:
        myIdx = np.where(heights[awf] >= r1)[0]
        if len(myIdx) > 4:
            Hpolys[awf] = np.polyfit(dts[awf][myIdx], heights[awf][myIdx], deg=polyDeg)
            vpolys[awf] = np.polyder(Hpolys[awf])
            apolys[awf] = np.polyder(vpolys[awf])
        else:
            Hpolys[awf] = None
            vpolys[awf] = None
            apolys[awf] = None    
    
        
    # |--- Stitch together fits for v/a ---| 
    # Get max height
    # Make fake time array
    rng = (lateT-earlyT).total_seconds() 
    nT = 250
    fakeT = np.linspace(0, rng, nT)
    
    # Get time cutoffs of stitching region
    stitchIdx = {}
    for awf in wfTypes:
        idx1, idx2, idx3 = None, None, None
        if type(Hsplines[awf]) != type(None):
            allHs1 = Hsplines[awf](fakeT)
            idx1 = np.min(np.where(allHs1 >=r1)[0]) # need to check from low side bc spline goes wonky
        if type(Hpolys[awf]) != type(None):
            Hpoly = np.poly1d(Hpolys[awf])
            allHs2 = Hpoly(fakeT)
            idx2 = np.max(np.where(allHs2 <=r2)[0])
            idx3 = np.where(allHs2 == np.max(allHs2))[0]
            if idx3 != idx2:
                print ("!!! Warning -- spline height fit for " +awf+"turns over before final time")
        stitchIdx[awf] = [idx1, idx2, idx3]            
    
    stitchTs = {}
    stitchVs = {}
    stitchAs = {}
    for awf in wfTypes:
        print (awf)
        idx1, idx2, idx3 = stitchIdx[awf]
        
        # Set up lo/hi generators
        if type(idx1) != type(None):
            mySpline = Hsplines[awf]
        else:
            mySpline = lambda x: 9999
        if type(idx2) != type(None):
            myPoly = np.poly1d(Hpolys[awf])            
        else:
            myPoly = lambda x: -9999
        myMaxH = myPoly(fakeT[idx3])
            
            
        for i in range(nT):
            myT = fakeT[i]
            print (i, myT/3600., mySpline(myT), myPoly(myT))
            
            # ^ sort out if reg1, smush, reg2 and pick
            # do for vels/acc at same time
            
            
    
    
    # Testing plot script
    fig2 = plt.figure()
    for awf in wfTypes:
        myIdx = np.where(heights[awf] <= 10)[0]
        if len(myIdx) > 4:
            plt.plot(dts[awf][myIdx]/3600, heights[awf][myIdx], 'o')
            plt.plot(dts[awf][myIdx]/3600, Hsplines[awf](dts[awf][myIdx]), '--')
        myIdx2 = np.where(heights[awf] >= 5)[0]  
        if len(myIdx2) >4:
            plt.plot(dts[awf][myIdx2]/3600, heights[awf][myIdx2], 'o')  
            Hpoly = np.poly1d(Hpolys[awf])
            plt.plot(dts[awf][myIdx2]/3600, Hpoly(dts[awf][myIdx2]), '--')
            
        #plt.plot(dts[awf]/3600, vsplines[awf](dts[awf])*7e5, '--')
    #plt.ylim(0,2500)
    plt.show()
    print (sd)

# |-----------------------------|
# |--- Process Required Args ---|
# |-----------------------------|
def processArgs(args):
    '''
    Helper script to check that all the required inputs
    have been included and that they have reasonable values.
    The required inputs and checks are:
        log file - an existing wombat log file
        
        log ids  - the id or ids of lines to process. it can a single wf,
                   a list of integers separated by +s, or a range given by
                   two integers separated by -. The ids do not need to use
                   the same background pickle 
    
        type - ht#, kin#, en#, linimg, logimg, sqimg. The first three are line 
               profiles and the last three are 2D images with the projected
               WFs.
                    ht#  - basic fit parameters versus height
                    kin# - basic + derived velocity/acceleration
                    en#  - kin + energetics from mass calc
                    
                    the # modifies which of the basic fit params are shown
                    1 - just height
                    2 - height + aw(s)
                    3 - everything
                
    
    Inputs:
        args: the results from sys.argv or from input2args
    
    Outputs:
        miniLog - logfile, but only the lines selected by the ids
    
        wombatRes - a dictionary filled by info pulled from miniLog
                    wombatRes[instName][wfType] = {}
                    keys - ids, params, times, pickles -> arrays of all the
                           values for each time matching that inst/type
    
        mode - the string tag for type of plot to make

        uniqTs - an array of all times with a recon
    
        wfType = the types of WFs
    
        allInsts  = the inst names
    
        
    '''
    #|--------------------------|
    #|--- Check the log file ---|     
    #|--------------------------|
    if not os.path.exists(args[0]):
        print ('Cannot find log file. Check location and/or call syntax')
        for astr in errorStrings:
            print (astr)
        sys.exit()
    else:
        try:
            logFile = np.genfromtxt(args[0], dtype=str)
        except:
            sys.exit('Error opening logFile, check that it is a WOMBAT log file')
            
    #|-------------------------|
    #|--- Check the log ids ---|     
    #|-------------------------|
    idstr = args[1]
    nplus = idstr.count('+')
    singleWF = True # will overwrite later
    
    # Range in ids
    if '-' in idstr:
        if '+' in idstr:
            sys.exit('Cannot process ids with both + and -')
        splitstr = idstr.split('-')
        if len(splitstr) > 2:
            sys.exit('Cannot process ids with multiple -')
        ids = np.arange(int(splitstr[0]), int(splitstr[1])+1,1, dtype=int)
    # Series of specific ids     
    elif '+' in idstr:
        splitstr = idstr.split('+')
        ids = []
        for aStr in splitstr:
            try:
                ids.append(int(aStr))
            except:
                print ('Error in converting id string to individual ids. Error at', aStr)
                print('Full command line syntax is')
                for astr in errorStrings:
                    print (astr)
                sys.exit()                
    else:
        try:
            ids = [int(idstr)]
        except:
            print ('Error in converting id string to individual ids. Error from', idstr)
            print('Full command line syntax is')
            for astr in errorStrings:
                print (astr)
            sys.exit()
 
    #|----------------------|
    #|--- Package Things ---|     
    #|----------------------|
    txtIds = np.array(ids) - 1 # indexing from 0 in python
    miniLog = logFile[txtIds,:]
    
    allInsts = np.unique(miniLog[:,1])
    wfTypes  = []
    
    wombatRes = {}
    for aInst in allInsts:
        someIdx = np.where(miniLog[:,1] == aInst)[0]
        myWFs = np.unique(miniLog[someIdx,3])
        wombatRes[aInst] = {}
        for aWF in myWFs:
            if aWF not in wfTypes:
                wfTypes.append(str(aWF))
            # Check that it is a single pickle for this inst/wf shape
            myIds = someIdx[np.where(miniLog[someIdx,3] == aWF)[0]]
            # Collect the parameters
            wombatRes[aInst][aWF] = {}
            wombatRes[aInst][aWF]['ids'] = myIds
            wombatRes[aInst][aWF]['params'] = {}
            wombatRes[aInst][aWF]['times'] = []
            wombatRes[aInst][aWF]['pickles'] = []
            for aIdx in myIds:
                wombatRes[aInst][aWF]['times'].append(datetime.datetime.strptime(miniLog[aIdx,2], "%Y-%m-%dT%H:%M:%S" ))
                wombatRes[aInst][aWF]['pickles'].append(miniLog[aIdx,13])
                myParams = miniLog[aIdx,4:13]
                myParams = myParams[myParams != 'None'].astype(float)
                wombatRes[aInst][aWF]['params'][miniLog[aIdx,2]] = myParams
    
            # Make time series versions for each param
            nPs = len(myParams)
            nTs = len(myIds)
            paramGrid = np.zeros([nPs, nTs])
            for i in range(nTs):
                paramGrid[:,i] = wombatRes[aInst][aWF]['params'][miniLog[myIds[i],2]]
            wombatRes[aInst][aWF]['paramGrid'] = paramGrid
    
    #|-------------------------------|
    #|--- Check the dimension tag ---|     
    #|-------------------------------|
    mode = args[2].lower()
    if mode not in ['ht1', 'kin1', 'en1', 'ht2', 'kin2', 'en2', 'ht3', 'kin3', 'en3', 'linimg', 'logimg', 'sqimg']:
        print ('Error in reading mode '+mode)
        print ('Pick from [ht#, kin#, en#, linimg, logimg, sqimg]')
        print('Full command line syntax is')
        for astr in errorStrings:
            print (astr)
        sys.exit()
        
    #|----------------------------------|
    #|--- Get some useful quantities ---|     
    #|----------------------------------|
    #uniqTs   nTimes 
    nInsts = len(allInsts)
    nWFs   = len(wfTypes)
    uniqTs = np.unique(miniLog[:,2])
    nTimes = len(uniqTs)
    
    
    return miniLog, wombatRes, mode, uniqTs, wfTypes, allInsts

# |-------------------------|
# |--- Line profile plot ---|
# |-------------------------|
def profilePlot(mode, wombatRes, wfTypes, logH=False, wfColors=False):
    ''' 
    Options:
        ht1, ht2, ht3 - just height/time, height+wid, all params
        kin1, kin2, kin3 - same as ht but adds vel + accel
        en1, en2, en3   - same as kin but add mass + KE
    '''
    #|---------------------|
    #|--- Set up colors ---|     
    #|---------------------|
    pltColors = {}
    if wfColors:
        for wft in wfTypes:
            pltColors[wft] = wf.colorDict[wft]
    else:
        counter = 0
        cols = ['#888888','#882255', '#332288', '#661100', '#6699CC']
        for wft in wfTypes:
            pltColors[wft] = cols[counter]
            counter += 1

    #|----------------------------|
    #|--- Set up based on mode ---|     
    #|----------------------------|
    # Figure out what we are plotting and sort things
    # by collecting into a list of ylabels to show on the
    # left and additional right ylabels (if needed)
    
    yLabelsL = []
    id2id    = {} # axis number for each param i2i[param] = ax
    yLabelsR = []
    id2idR   = {}
    rAxes    = []
    rColors  = {} 
    lColors   = {}
    lColors[0] = 'k' # everyone has a height
    
    # |--- Case 1 ---|
    # Only height 
    if '1' in mode:
        nParams = 1
        tempWF =  wf.wireframe(wfTypes[0].replace('Half', 'Half '))
        yLabelsL.append(tempWF.labels[0])
        for wft in wfTypes:
            id2id[wft] = [0]
        
    # |--- Case 2 ---|
    # Only height and ang width
    elif '2' in mode:
        nParams = 1
        # Loop to see what combo of AWs we have
        myAWs = []
        for wft in wfTypes:
            tempWF = wf.wireframe(wft.replace('Half', 'Half '))
            if 'AW (deg)' in tempWF.labels:
                lColor = pltColors[wft]
                myAWs.append('AW (deg)')
            if 'AW_FO (deg)' in tempWF.labels:
                myAWs.append('AW_FO (deg)')
            if 'AW_EO (deg)' in tempWF.labels:
                myAWs.append('AW_EO (deg)')
            # set up the index mapping length, set -1 as default    
            id2id[wft] = np.zeros(len(tempWF.labels), dtype=int) - 1
            id2idR[wft] = np.zeros(len(tempWF.labels), dtype=int) - 1
        
        # Add the height for everyone        
        yLabelsL.append('Height (Rs)')
        for wft in wfTypes:
            id2id[wft][0] = 0
                
        # Process based on number of AW tags        
        nAWs = len(np.unique(myAWs))
        # 1 aw - simple case (test with wbOutputs/201207.txt 95-109 ht2 )
        if nAWs == 1:
            nParams = 2
            yLabelsL.append('AW (deg)')
            lColors[1] = 'k'
            for wft in wfTypes:
                tempWF = wf.wireframe(wft.replace('Half', 'Half '))
                if 'AW (deg)' in tempWF.labels:
                    idx = np.where(tempWF.labels == 'AW (deg)')[0]
                    id2id[wft][idx[0]] = 1
        
        # 2 aw - only have torus fo/eo but not aw (wbOutputs/WomBlog.txt 95+98+101 ht2)
        if nAWs == 2:
            nParams = 3
            yLabelsL.append('AW_FO (deg)')
            yLabelsL.append('AW_EO (deg)')
            
            for wft in wfTypes:
                tempWF = wf.wireframe(wft.replace('Half', 'Half '))
                if 'AW_FO (deg)' in tempWF.labels:
                    idx = np.where(tempWF.labels == 'AW_FO (deg)')[0]
                    id2id[wft][idx[0]] = 1
                    if len(wfTypes) > 1:
                        lColors[1] = pltColors[wft]
                    else:
                        lColors[1] = 'k'
                if 'AW_EO (deg)' in tempWF.labels:
                    idx = np.where(tempWF.labels == 'AW_EO (deg)')[0]
                    id2id[wft][idx[0]] = 2
                    if len(wfTypes) > 1:
                        lColors[2] = pltColors[wft]
                    else:
                        lColors[2] = 'k'
            
        # 3 aw - have both aw and fo/eo (wbOutputs/WomBlog.txt 95-109 ht2)
        if nAWs == 3:
            nParams = 3
            yLabelsL.append('AW (deg)')
            yLabelsL.append('')
            yLabelsR.append('')
            yLabelsR.append('AW_FO (deg)')
            yLabelsR.append('AW_EO (deg)')
            for wft in wfTypes:
                tempWF = wf.wireframe(wft.replace('Half', 'Half '))
                if 'AW (deg)' in tempWF.labels:
                    idx = np.where(tempWF.labels == 'AW (deg)')[0]
                    id2id[wft][idx[0]] = 1
                    lColors[1] = pltColors[wft]
                if 'AW_FO (deg)' in tempWF.labels:
                    idx = np.where(tempWF.labels == 'AW_FO (deg)')[0]
                    id2idR[wft][idx[0]] = 1
                    rColors[1] = pltColors[wft]
                if 'AW_EO (deg)' in tempWF.labels:
                    idx = np.where(tempWF.labels == 'AW_EO (deg)')[0]
                    id2idR[wft][idx[0]] = 2
                    rColors[2] = pltColors[wft]
                
        
    # |--- Case 3 ---|
    # Plot it all!
    else:
        # Figure out who has the most params and
        # what that number is. Collect the labels
        # at the same time
        nParams = 0
        maxType = None
        ylabs = {}
        for wft in wfTypes:
            if wf.npDict[wft.replace('Half', 'Half ')] > nParams:
                nParams = wf.npDict[wft.replace('Half', 'Half ')]
                maxType = wft
            tempWF = wf.wireframe(wft.replace('Half', 'Half '))
            ylabs[wft] = tempWF.labels
        id2id[maxType] = np.arange(0,nParams)
        yLabelsL = np.array(ylabs[maxType])
        # set right labels as long enough flag it string
        yLabelsR = np.array(['nullnullnullnull' for i in range(nParams)])
        lColors = {}
        for i in range(nParams):
            lColors[i] = pltColors[maxType]
        doubleLabels = {}
        doubleColors = {}
        
        # Loop through the other WFs and assign to spots
        for wft in wfTypes:
            if wft != maxType:
                myn = len(ylabs[wft])
                id2id[wft] = []
                for i in range(myn):
                    if (ylabs[wft][i] in yLabelsL):
                        idx = np.where(yLabelsL ==  ylabs[wft][i])[0]
                        id2id[wft].append(idx[0])
                    else:
                        id2id[wft].append(-1) # flag as not on left
                        # Figure out if we already have rights
                        if wft not in id2idR.keys():
                            id2idR[wft] = np.zeros(myn) - 1 # set -1 for empty
                        # Add to right if label already exits on right    
                        if (ylabs[wft][i] in yLabelsR):
                            idx = np.where(yLabelsR ==  ylabs[wft][i])[0]
                            id2idR[wft][i] = int(idx[0])
                            
                        # Otherwise add to right    
                        if ylabs[wft][i] in labelMatch.keys():
                            myFriends = labelMatch[ylabs[wft][i]]
                            idL = -1 # subtle shiver at this var name
                            lonely = True
                            while lonely:
                                for aFriend in myFriends:
                                    if aFriend in yLabelsL:
                                        idL = np.where(yLabelsL == aFriend)[0] 
                                        id2idR[wft][i] = int(idL[0])
                                        if yLabelsR[idL] == 'nullnullnullnull':
                                            yLabelsR[idL] = ylabs[wft][i]
                                            rColors[idL[0]] = pltColors[wft]   
                                        elif yLabelsR[idL] != ylabs[wft][i]:
                                        #else:
                                            doubleLabels[idL[0]] = [yLabelsR[idL[0]], ylabs[wft][i]]
                                            yLabelsR[idL] = 'double'
                                            doubleColors[idL[0]] = [rColors[idL[0]], pltColors[wft]]
                                        lonely = False
                                if lonely:
                                        sys.exit(ylabs[wft][i] + ' has no friend, find somewhere to add in labelMatch dictionary')
                                        
        # Clean up left labels - set at black if no right axis label
        for i in range(nParams):
            if yLabelsR[i] == 'nullnullnullnull':
                lColors[i] = 'k'
                               
    #|-------------------------|
    #|--- Add in kinematics ---|
    #|-------------------------|
    if ('kin' in mode) or ('en' in mode):
        velRes, accRes = getKinematics(wombatRes, wfTypes)
    
    
    #|-------------------------|
    #|--- Add in energetics ---|
    #|-------------------------|
    

        
    #|-------------------------|
    #|--- Set up the figure ---|     
    #|-------------------------|
    np2sz = {1:3, 2:5, 3:6, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:11, 11:12}
    fig, ax = plt.subplots(nParams, 1, figsize=(7,np2sz[nParams]), layout='constrained')
    
    if nParams == 1:
        ax = [ax]
        
    axR = {}
    
    #|--- Set up the left labels ---|
    for i in range(nParams):
        if yLabelsL[i] != '': 
            ax[i].set_ylabel(labSwap[yLabelsL[i]], color=lColors[i])

    #|--- Set up the right labels ---|
    for i in range(len(yLabelsR)):
        # Single right label
        if yLabelsR[i] not in ['nullnullnullnull', 'double', '']:
            nowC = rColors[i]
            axR[i] = ax[i].twinx() 
            axR[i].set_ylabel(labSwap[yLabelsR[i]], color=nowC)
        # Double right label (or more?)
        elif yLabelsR[i] == 'double':
            axR[i] = ax[i].twinx() 
            axR[i].set_ylabel('')
            for j in range(len(doubleLabels[i])):            
                axR[i].text(0.98,0.95-0.23*j, labSwap[doubleLabels[i][j]], transform=axR[i].transAxes, horizontalalignment='right', verticalalignment='top', color=doubleColors[i][j] )
    
            
    #|--------------------------|
    #|--- Fill in the figure ---|     
    #|--------------------------| 
    hasLabel = []   
    for aInst in wombatRes.keys():
        subRes = wombatRes[aInst]
        for aType in wfTypes:
            if aType in subRes.keys():
                myRes = subRes[aType]
                myTimes = myRes['times']
                myC = pltColors[aType]
                
                for i in range(len(id2id[aType])):
                    myax = None
                    if id2id[aType][i] != -1:
                        myax = ax[id2id[aType][i]]
                    elif aType in id2idR:
                        if id2idR[aType][i] != -1:
                            myax = axR[id2idR[aType][i]]
                                              
                    if type(myax) != type(None):    
                        if aType not in hasLabel:
                            myax.plot(myTimes, myRes['paramGrid'][i,:], 'o', c=myC, label=aType)
                            hasLabel.append(aType)
                        else:
                            myax.plot(myTimes, myRes['paramGrid'][i,:], 'o', c=myC)
    if logH:
        ax[0].set_yscale('log')
                        
    ax[0].legend(loc='lower right', bbox_to_anchor=(1., 1.), ncols=len(hasLabel))
    
        
        
    #print (id2id)
    #print (id2idR)
    #print (yLabelsL)
    #print (yLabelsR)
    
    plt.show()

    

def wombatPlotWrapper(args):
    #|----------------------------------|
    #|--- Check the number of inputs ---|     
    #|----------------------------------|
    nArgs = len(args)
    if (nArgs <3) or (len(args) > 6):
        print ('Incorrect number of parameters provided. Syntax is')
        for astr in errorStrings:
            print (astr)
        sys.exit()
        
    
    #|-------------------------------------|
    #|--- Check the critical parameters ---|     
    #|-------------------------------------|
    miniLog, wombatRes, mode, uniqTs, wfTypes, allInsts = processArgs(args)
    
    #|-------------------------|
    #|---Run line plot mode ---|     
    #|-------------------------|
    if mode in ['ht1', 'ht2','ht3', 'kin1', 'kin2', 'kin3']:
        profilePlot(mode, wombatRes, wfTypes)
        


# |-----------------------|
# |--- Text line input ---|
# |-----------------------|
if __name__ == '__main__':
    wombatPlotWrapper(sys.argv[1:])
