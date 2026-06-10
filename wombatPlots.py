import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys, os
import datetime
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline

from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning

from dingo import dingoWrapper
# Ignore all OptimizeWarnings globally
warnings.filterwarnings("ignore", category=OptimizeWarning)
np.seterr(invalid='ignore', divide='ignore')

sys.path.append('wombatCode/') 
import wombatWF as wf

import pickle


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

global vdragScalers 
vdragScalers = [600e5, 350e5, 1e-12]

def vdrag(t_in, vCME0_in, vSW_in, C_in): 
    # Inputs are normalized to near 1 to make the 
    # curve_fit happy, convert back to physical units 
    # using vdragScalers (cm, cm, 1/cm)
    C = C_in * vdragScalers[2]
    t = t_in 
    vCME0 = vCME0_in * vdragScalers[0]
    vSW   = vSW_in * vdragScalers[1]
    vout = (vCME0 - vSW) / (1 + C * (vCME0 - vSW) * t) + vSW 
    return vout

# |-----------------------------------|
# |--- Get kinematics from results ---|
# |-----------------------------------|
def getKinematics(wombatRes, wfTypes, dragHeights=[5,21.5]):
    # |---------------------|
    # |--- Preprocessing ---|
    # |---------------------|
    # Convert drag Heights to cm
    dragHeights = np.array(dragHeights)*7e10
    
    # |--- Make holders ---|
    times = {}
    dts   = {}
    heights = {}
    newtVs  = {} # newtonian derivs
    newtAs  = {} # newtonian derivs
    for awf in wfTypes:
        times[awf]   = []
        heights[awf] = []
        dts[awf] = []
        newtVs[awf] = []
        newtAs[awf] = []
    
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
        heights[awf] = np.array(heights[awf])  * 7e10
        idxs  = np.argsort(times[awf])
        times[awf] = times[awf][idxs]
        heights[awf] = heights[awf][idxs]
        
    # |-----------------|
    # |--- Calc v/a  ---|
    # |-----------------|
    earlyT = datetime.datetime(3000,1,1)
    lateT  = datetime.datetime(1000,1,1)
    for awf in wfTypes:
        if times[awf][0] < earlyT:
            earlyT = times[awf][0]
        if times[awf][-1] > lateT:
            lateT = times[awf][-1]
    for awf in wfTypes:
        print ('Calculating two-point derivatives for', awf)
        for i in range(len(times[awf])):
            dts[awf].append((times[awf][i]-earlyT).total_seconds())
            if i != 0:
                newtVs[awf].append((heights[awf][i]-heights[awf][i-1])/(times[awf][i]-times[awf][i-1]).total_seconds())
            if i > 1:
                j = i -1
                newtAs[awf].append((newtVs[awf][j] - newtVs[awf][j-1])/(dts[awf][j] - dts[awf][j-1]))
                
        dts[awf] = np.array(dts[awf])
        newtVs[awf] = np.array(newtVs[awf])
        newtAs[awf] = np.array(newtAs[awf]) # matches dts[awf][1:-1]
        
        # |--- Print some basic things ---|
        print ('    Max v (km/s):', '{:.1f}'.format(np.max(newtVs[awf])/1e5))
        ipPoints = np.where(heights[awf] >= dragHeights[0])[0]
        if len(ipPoints) > 3:
            print (' avg IP v (km/s):', '{:.1f}'.format(np.mean(newtVs[awf][ipPoints[0]+1:])/1e5))
            print (' med IP v (km/s):', '{:.1f}'.format(np.median(newtVs[awf][ipPoints[0]+1:])/1e5))        
        print ('')
             
       
            
    # |--------------------|
    # |--- Fit Drag Eq  ---|
    # |--------------------|
    # Fit to velocity seems to work best
    splitIds =  {}
    dragFits = {}
    for awf in wfTypes:
        midH = 0.5*(heights[awf][1:] + heights[awf][:-1])
        midT = 0.5*(dts[awf][1:] + dts[awf][:-1])
        vSmooth = gaussian_filter1d(newtVs[awf], sigma=1)       
        
        # Consider all starting points in between heights given
        # by dragHeights
        aboveDH = np.where(midH >= dragHeights[0])[0]
        bestVal = 9e20 # arbitrary large
        bestId  = -1
        bestPs = None
        if len(aboveDH) > 3:
            for i in range(len(aboveDH)-3):
                if midH[aboveDH[i]] <= dragHeights[1]:
                    x, y = midT[aboveDH[i:]] - midT[aboveDH[i]], vSmooth[aboveDH[i:]]
                    # Might not converge so put in try/except
                    # (coincidently ck was listening to Converge when writing this)
                    try:
                        goodIdx = np.where(y > 0)[0]
                        popt, pcov = curve_fit(vdrag, x[goodIdx], y[goodIdx], p0=[1,1, 1])
                        errs = np.sqrt(np.diag(pcov)) # 1 stddev errors
                        v1, v2, v3 = popt
                        err1, err2, err3 = errs
                        totErr = np.abs(err1/v1) + np.abs(err2/v2) + np.abs(err3/v3)
                        if totErr < bestVal:
                            bestId = aboveDH[i]
                            bestVal = totErr
                            bestPs = [popt, errs]                       
                    except:
                        pass
        
        # |--- Package and print info to terminal ---|
        if (bestId != -1) and (bestVal <=10):
            v1, v2, v3 = bestPs[0]
            e1, e2, e3 = bestPs[1]
            if bestId == 0:
                hunc = midH[1]-midH[0]
            else:
                hunc = 0.5*(midH[bestId+1]-midH[bestId-1])
            print ('Starting ' +awf +' drag fit at index', bestId)
            print (' Total error: ', '{:.3f}'.format(bestVal), '(sum of the 3 fractional errors)')
            print ('  vCME_0 (km/s): ', '{:.1f}'.format(v1*vdragScalers[0]/1e5), '+/-', '{:.1f}'.format(e1* vdragScalers[0]/1e5))
            print ('   vSW_0 (km/s): ', '{:.1f}'.format(v2*vdragScalers[1]/1e5), '+/-', '{:.1f}'.format(e2*vdragScalers[1]/1e5))
            print ('       C (1/cm): ', '{:.2e}'.format(v3 * vdragScalers[2]), '+/-', '{:.2e}'.format(e3 * vdragScalers[2]))
            print ('starting H (Rs): ', '{:.2f}'.format(midH[bestId]/7e10), '+/-', '{:.2f}'.format(hunc/7e10), '(unc from h resolution)')
            bestPs[0] = np.append(bestPs[0], midH[bestId])
            bestPs[1] = np.append(bestPs[1], hunc)
            dragFits[awf] = bestPs
            
        else:
            print ('Cannot fit drag eq to '+awf)
            bestId = -1
        print ('')
        splitIds[awf] = bestId
    
    
              
    # |----------------------------|
    # |--- Secret testing plot  ---|
    # |----------------------------|
    if False:
        fig = plt.figure()
        for awf in wfTypes:
            midH = 0.5*(heights[awf][1:] + heights[awf][:-1])
            midT = 0.5*(dts[awf][1:] + dts[awf][:-1])
            vSmooth = gaussian_filter1d(newtVs[awf], sigma=1)
        
            plt.plot(midT/3600, newtVs[awf]/1e5, 'co')
            plt.plot(midT/3600, vSmooth/1e5, 'ko')
    
            myIdx = splitIds[awf]
            if myIdx != -1:
                myParams = dragFits[awf]
                v1, v2, v3, v4 = myParams[0]
                x = midT[myIdx:] - midT[myIdx]
                plt.plot((x+midT[myIdx])/3600, vdrag(x,v1, v2, v3)/1e5, '--')
        
        plt.show()
    
    # |-----------------------|
    # |--- Package Output  ---|
    # |-----------------------|
    outRes = {}
    outRes['times'] = times
    outRes['dts'] = dts
    outRes['heights'] = heights
    outRes['newtVs'] = newtVs
    outRes['newtAs'] = newtAs
    outRes['splitIds'] = splitIds
    outRes['dragFits'] = dragFits

    return outRes

# |-----------------------------------|
# |--- Get energetics from results ---|
# |-----------------------------------|
def getEnergetics(args, wombatRes, wfTypes, kinRes):
    
    bigMassRes = {} # index by shape then inst
    for awf in wfTypes:
        bigMassRes[awf] = {}
        for aInst in wombatRes.keys():
            if awf in wombatRes[aInst]:
                bigMassRes[awf][aInst] = {}
                bigMassRes[awf][aInst]['times'] = []
                bigMassRes[awf][aInst]['masses'] = []

    # Have to sort into chunks for each pickle for 
    # the dingo mass calc. Can do two WFs at same time
    
    
    # |--- Collect things ---|
    for aInst in wombatRes.keys():
        
        if ('EUV' not in aInst.upper()) & ('AIA' not in aInst.upper()):
            print ('Calculating masses for', aInst)
            # Check if one or two res, can process together
            if len(wombatRes[aInst].keys()) < 3:
                idsbyPickle = {}
                # Collect all the ids for each pickle
                for awf in wombatRes[aInst].keys():
                    for i in range(len(wombatRes[aInst][awf]['pickles'])):
                        if wombatRes[aInst][awf]['pickles'][i] in idsbyPickle.keys():
                            idsbyPickle[wombatRes[aInst][awf]['pickles'][i]].append(wombatRes[aInst][awf]['ids'][i]+1)
                        else:
                            idsbyPickle[wombatRes[aInst][awf]['pickles'][i]] = [wombatRes[aInst][awf]['ids'][i]+1]
                            
                # Convert the array of ints to a string for dingo (stringo)
                for key in idsbyPickle:
                    myids = idsbyPickle[key]
                    nids = len(myids)
                    
                    if nids == 1:
                        strids = str(myids[0])
                    else:
                        strids = ''
                        for i in range(nids-1):
                            strids = strids + str(myids[i]) + '+'
                        strids = strids + str(myids[-1])
                    idsbyPickle[key] = strids
                    
                    # Pass to dingo
                    dargs = [args[0], strids, '0d']
                    if aInst == 'HI1A_SR':
                    #if True:
                        massRes, aboutMe = dingoWrapper(dargs, pullMass=True)
                        print (nids)
                        print (strids)
                        
                        for i in range(len(aboutMe)):
                            mydeets = aboutMe[i].split()
                            for j in range(len(massRes[i])):
                                bigMassRes[mydeets[2+j]][mydeets[1]]['times'].append(mydeets[0])
                                bigMassRes[mydeets[2+j]][mydeets[1]]['masses'].append(massRes[i][j])
                        
                        
    with open("bigMassRes.pkl", "wb") as file:
        pickle.dump(bigMassRes, file)

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
    #kinRes = None
    
    
    
    #|-------------------------|
    #|--- Add in energetics ---|
    #|-------------------------|
    #enRes = None
    #print (mode)
    
        
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
    
    #|--------------------------|
    #|--- Process kinematics ---|     
    #|--------------------------|
    if ('kin' in mode) or ('en' in mode):
        kinRes = getKinematics(wombatRes, wfTypes)
    
    #|--------------------------|
    #|--- Process energetics ---|     
    #|--------------------------|    
    if ('en' in mode):
        enRes = getEnergetics(args, wombatRes, wfTypes, kinRes)
        
        
    #|-------------------------|
    #|---Run line plot mode ---|     
    #|-------------------------|
    if mode in ['ht1', 'ht2','ht3', 'kin1', 'kin2', 'kin3', 'en1', 'en2', 'en3']:
        profilePlot(mode, wombatRes, wfTypes)
        


# |-----------------------|
# |--- Text line input ---|
# |-----------------------|
if __name__ == '__main__':
    wombatPlotWrapper(sys.argv[1:])
