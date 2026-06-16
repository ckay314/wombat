import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys, os
import datetime
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
from matplotlib.ticker import ScalarFormatter

from dingo import dingoWrapper
# Ignore all OptimizeWarning/UserWarning
warnings.filterwarnings("ignore", category=OptimizeWarning)
np.seterr(invalid='ignore', divide='ignore')
warnings.filterwarnings("ignore", category=UserWarning)

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
global errorStrings, labSwap,labErrs, labelMatch, inst2sat, labErrs, hErrs
errorStrings = [' ', '  python3 wombatPlots.py logFile id(s) type otherParams', '          where:', '          - logFile is a wombat log file', '          - ids is an integer or int+int or int-int', '          - type sets the plot type from [ht, kin, en, linimg, logimg, sqimg]', '          - otherParams includes min#, max#, outName, logIt, GUIcolors, and picType']

# Dictionary to swap from the GUI names that dont like latex to
# nicer things for this plot
deg = '($^{\\circ}$)'
labSwap = {'Height (Rs)':'Height (R$_S$)', 'Lon (deg)':'Lon'+deg , 'Lat (deg)': 'Lat'+deg, 'Tilt (deg)':'Tilt'+deg, 'AW (deg)':'AW'+deg, 'kappa':'$\\kappa$', 'AW_FO (deg)':'AW$_{FO}$'+deg, 'AW_EO (deg)':'AW$_{EO}$'+deg, 'deltaAx':'$\\delta_{Ax}$', 'deltaCS':'$\\delta_{CS}$', 'ecc1':'$\\epsilon_1$', 'ecc2':'$\\epsilon_2$', 'Roll (deg)':'Roll'+deg, 'Yaw (deg)':'Yaw'+deg, 'Pitch (deg)':'Pitch'+deg, 'Lx (Rs)':'L$_x$ (R$_S$)', 'Ly (Rs)':'L$_y$ (R$_S$)', 'Lz (Rs)':'L$_z$ (R$_S$)', 'HeightO (Rs)':'Height$_O$ (R$_S$)', 'LonO (deg)':'Lon$_O$'+deg, 'LatO (deg)':'Lat$_O$'+deg}

# Set up general errors for each param
# Assume heights depend on which inst
hErrs = {'EUV':0.05, 'COR':0.1, 'HI':0.5}
labErrs = {'Height (Rs)':hErrs, 'Lon (deg)':10, 'Lat (deg)': 5, 'Tilt (deg)':15, 'AW (deg)':5, 'kappa':0.05, 'AW_FO (deg)':5, 'AW_EO (deg)':5, 'deltaAx':0.1, 'deltaCS':0.1, 'ecc1':0.1, 'ecc2':0.1, 'Roll (deg)':15, 'Yaw (deg)':15, 'Pitch (deg)':15, 'Lx (Rs)':hErrs, 'Ly (Rs)':hErrs, 'Lz (Rs)':hErrs, 'HeightO (Rs)':0.1, 'LonO (deg)':5, 'LatO (deg)':5}


# Who to pair things with, GCS* is longest, followed by Slab
# Only match nice pairs, otherwise let it dump them in wherever
labelMatch = {'Tilt (deg)':['Roll (deg)'], 'AW (deg)': ['AW_FO (deg)', 'Lx (Rs)'], 'AW_FO (deg)':['AW (deg)', 'Lx (Rs)'],  'AW_EO (deg)':['AW (deg)', 'Ly (Rs)'], 'kappa':['ecc1', 'deltaAx',  'Yaw (deg)'], 'ecc1':['kappa', 'deltaAx', 'Yaw (deg)'],  'ecc2': ['deltaCS', 'Pitch (deg)'], 'deltaAx':['kappa', 'ecc1', 'Yaw (deg)'], 'deltaCS':['ecc2', 'Pitch (deg)']}

# Dict to sort instruments by sat 
inst2sat = {'AIA94':'SDO', 'AIA131':'SDO', 'AIA171':'SDO','AIA193':'SDO','AIA211':'SDO','AIA304':'SDO','AIA335':'SDO','AIA1600':'SDO','AIA1700':'SDO', 'C2':'SOHO', 'C3':'SOHO', 'COR1':'STEREO', 'COR2':'STEREO', 'COR1A':'STEREOA', 'COR2A':'STEREOA', 'COR1B':'STEREOB', 'COR2B':'STEREOB', 'EUVI171':'STEREO', 'EUVI195':'STEREO', 'EUVI284':'STEREO', 'EUVI304':'STEREO', 'EUVI171A':'STEREOA', 'EUVI195A':'STEREOA', 'EUVI284A':'STEREOA', 'EUVI304A':'STEREOA', 'EUVI171B':'STEREOB', 'EUVI195B':'STEREOB', 'EUVI284B':'STEREOB', 'EUVI304B':'STEREOB', 'HI1':'STEREO', 'HI2':'STEREO', 'HI1A':'STEREOA', 'HI2A':'STEREOA', 'HI1B':'STEREOB', 'HI2B':'STEREOB', 'HI1A_SR':'STEREOA', 'HI1B_SR':'STEREOB', 'HI2A_SR':'STEREOA', 'HI2B_SR':'STEREOB', 'SOLOHI':'SOLO', 'SOLOHI1':'SOLO', 'SOLOHI2':'SOLO', 'SOLOHI3':'SOLO', 'SOLOHI4':'SOLO', 'WISPR':'PSP', 'WISPRI':'PSP', 'WISPRO':'PSP', 'WISPR_LW':'PSP', 'WISPRI_LW':'PSP', 'WISPRO_LW':'PSP', 'WISPR_L3':'PSP', 'WISPRI_L3':'PSP', 'WISPRO_L3':'PSP'}

# Dict to convertt instrument to type 
inst2type = {'AIA94':'EUV', 'AIA131':'EUV', 'AIA171':'EUV','AIA193':'EUV','AIA211':'EUV','AIA304':'EUV','AIA335':'EUV','AIA1600':'EUV','AIA1700':'EUV', 'C2':'COR', 'C3':'COR', 'COR1':'COR', 'COR2':'COR', 'COR1A':'COR', 'COR2A':'COR', 'COR1B':'COR', 'COR2B':'COR', 'EUVI171':'EUV', 'EUVI195':'EUV', 'EUVI284':'EUV', 'EUVI304':'EUV', 'EUVI171A':'EUV', 'EUVI195A':'EUV', 'EUVI284A':'EUV', 'EUVI304A':'EUV', 'EUVI171B':'EUV', 'EUVI195B':'EUV', 'EUVI284B':'EUV', 'EUVI304B':'EUV', 'HI1':'HI', 'HI2':'HI', 'HI1A':'HI', 'HI2A':'HI', 'HI1B':'HI', 'HI2B':'HI', 'HI1A_SR':'HI', 'HI1B_SR':'HI', 'HI2A_SR':'HI', 'HI2B_SR':'HI', 'SOLOHI':'HI', 'SOLOHI1':'HI', 'SOLOHI2':'HI', 'SOLOHI3':'HI', 'SOLOHI4':'HI', 'WISPR':'HI', 'WISPRI':'HI', 'WISPRO':'HI', 'WISPR_LW':'HI', 'WISPRI_LW':'HI', 'WISPRO_LW':'HI', 'WISPR_L3':'HI', 'WISPRI_L3':'HI', 'WISPRO_L3':'HI'}


global vdragScalers 
vdragScalers = [600e5, 350e5, 1e-12]

global picType
picType = '.png' # png or pdf, gets overwritten if set in input 


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
def getKinematics(wombatRes, wfTypes, dragHeights=[5,21.5], incDrag=False):
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
    uncHs   = {}
    uncVs   = {}
    uncAs   = {}
    for awf in wfTypes:
        times[awf]   = []
        heights[awf] = []
        dts[awf] = []
        newtVs[awf] = []
        newtAs[awf] = []
        uncHs[awf] = []
        uncVs[awf] = []
        uncAs[awf] = []
        
    # |--- Collect things ---|
    for aInst in wombatRes.keys():
        subRes = wombatRes[aInst]
        hunc = hErrs[inst2type[aInst]]
        for awf in wfTypes:
            if awf in subRes.keys():
                myRes = subRes[awf]
                myTimes = myRes['times']
                for i in range(len(myTimes)):
                    roundTime = myTimes[i].replace(second=0)
                    if roundTime not in times[awf]:
                        times[awf].append(roundTime)
                        heights[awf].append(myRes['paramGrid'][0,i])
                        uncHs[awf].append(hunc)
                        
    # |--- Sort things ---|
    for awf in wfTypes:
        times[awf] = np.array(times[awf])
        heights[awf] = np.array(heights[awf])  * 7e10
        uncHs[awf] = np.array(uncHs[awf]) * 7e10
        idxs  = np.argsort(times[awf])
        times[awf] = times[awf][idxs]
        heights[awf] = heights[awf][idxs]
        uncHs[awf] = uncHs[awf][idxs]
    
    # |----------------------------|
    # |--- Secret testing plot  ---|
    # |----------------------------|
    if False:
        fig = plt.figure()
        plt.plot(times['GCS'], heights['GCS']/7e10, 'ko')
        plt.plot(times['Sphere'], heights['Sphere']/7e10, 'bo')
        plt.show()
        print(sd)
       
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
        #hSmooth = gaussian_filter1d(heights[awf], sigma=1) 
        hSmooth = heights[awf]
        myuncH  = uncHs[awf]
        for i in range(len(times[awf])):
            dts[awf].append((times[awf][i]-earlyT).total_seconds())
            if i != 0:
                newtVs[awf].append((hSmooth[i]-hSmooth[i-1])/(times[awf][i]-times[awf][i-1]).total_seconds())
                uncVs[awf].append(np.sqrt(myuncH[i]**2 + myuncH[i-1]**2)/(times[awf][i]-times[awf][i-1]).total_seconds()) # cm/s
                #print ((np.sqrt(myuncH[i]**2 + myuncH[i-1]**2))/7e10, (times[awf][i]-times[awf][i-1]).total_seconds()/60, uncVs[awf][-1]/1e5)
                
            if i > 1:
                j = i -1
                newtAs[awf].append((newtVs[awf][j] - newtVs[awf][j-1])/(dts[awf][j] - dts[awf][j-1]))
                uncAs[awf].append(np.sqrt(uncVs[awf][j]**2 + uncVs[awf][j-1]**2)/(dts[awf][i]-dts[awf][i-1])) # cm/s
                
        dts[awf] = np.array(dts[awf])
        newtVs[awf] = np.array(newtVs[awf])
        newtAs[awf] = np.array(newtAs[awf]) # matches dts[awf][1:-1]
        uncVs[awf]  = np.array(uncVs[awf])
        uncAs[awf]  = np.array(uncAs[awf])
        
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
    if incDrag:
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
                    tunc = (midT[1]-midT[0])
                else:
                    hunc = 0.5*(midH[bestId+1]-midH[bestId-1])
                    tunc = 0.5*(midT[bestId+1]-midT[bestId-1])
                print ('Starting ' +awf +' drag fit at index', bestId)
                print (' Total error: ', '{:.3f}'.format(bestVal), '(sum of the 3 fractional errors)')
                print ('  vCME_0 (km/s): ', '{:.1f}'.format(v1*vdragScalers[0]/1e5), '+/-', '{:.1f}'.format(e1* vdragScalers[0]/1e5))
                print ('   vSW_0 (km/s): ', '{:.1f}'.format(v2*vdragScalers[1]/1e5), '+/-', '{:.1f}'.format(e2*vdragScalers[1]/1e5))
                print ('       C (1/cm): ', '{:.2e}'.format(v3 * vdragScalers[2]), '+/-', '{:.2e}'.format(e3 * vdragScalers[2]))
                print ('starting H (Rs): ', '{:.2f}'.format(midH[bestId]/7e10), '+/-', '{:.2f}'.format(hunc/7e10), '(unc from h resolution)')
                bestPs[0] = np.append(bestPs[0], [midH[bestId], earlyT + datetime.timedelta(seconds=midT[bestId])])
                bestPs[1] = np.append(bestPs[1], [hunc, tunc])
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
        
            #plt.plot(midT/3600, newtVs[awf]/1e5, 'co')
            #plt.plot(midT/3600, vSmooth/1e5, 'ko')
            #plt.plot(midH/7e10, newtVs[awf]/1e5, 'co')
            cDict = {'GCS':'k', 'Sphere':'b'}
            plt.plot(midH/7e10, vSmooth/1e5, 'o', c=cDict[awf])
            
            myIdx = splitIds[awf]
            if myIdx != -1:
                myParams = dragFits[awf]
                v1, v2, v3, v4, v5 = myParams[0]
                x = midT[myIdx:] - midT[myIdx]
                #plt.plot((x+midT[myIdx])/3600, vdrag(x,v1, v2, v3)/1e5, '--')
                plt.plot(midH[myIdx:]/7e10, vdrag(x,v1, v2, v3)/1e5, '--', c=cDict[awf])
                
            ats = dts[awf][1:-1]   
            ahts = heights[awf][1:-1] / 7e10
            #plt.plot(ats/3600, newtAs[awf]/1e2, 'c+')
            plt.plot(ahts, newtAs[awf]/1e2, '+', c=cDict[awf])
            myIdx = splitIds[awf]
            if myIdx != -1:
                myParams = dragFits[awf]
                v1, v2, v3, v4, v5 = myParams[0]
                x = midT[myIdx:] - midT[myIdx]
                print ('x', x)
                print (v1, v2, v3)
                print (vdrag(x,v1, v2, v3))
                #plt.plot((x+midT[myIdx])/3600, -v3*vdragScalers[2]*vdrag(x,v1, v2, v3)**2 / 1e2, '--')
                #plt.plot((x+midT[myIdx])/3600, 0*(x+midT[myIdx])/3600, 'r--')
                plt.plot(midH[myIdx:]/7e10, -v3*vdragScalers[2]*vdrag(x,v1, v2, v3)**2 / 1e2, '--', c=cDict[awf])
                plt.plot(midH[myIdx:]/7e10, 0*(x+midT[myIdx])/3600, 'r--')
                
        
        plt.show()
        #print (sd)
    
    # |-----------------------|
    # |--- Package Output  ---|
    # |-----------------------|
    outResK = {}
    outResK['times'] = times
    outResK['dts'] = dts
    outResK['heights'] = heights    
    outResK['vels'] = newtVs
    outResK['accs'] = newtAs 
    outResK['errs'] = [uncHs, uncVs, uncAs]
    if incDrag:
        outResK['splitIds'] = splitIds
        outResK['dragFits'] = dragFits

    return outResK

# |-----------------------------------|
# |--- Get energetics from results ---|
# |-----------------------------------|
def getEnergetics(args, wombatRes, wfTypes, kinRes, reloadIt=None):
        
    # |--------------------------------|
    # |--- Use Dingo to calc masses ---|
    # |--------------------------------|
    # Check if passed reloadIt, could be an existing pkl or
    # a name to save the output for future use
    saveName = 'bigMassRes.pkl'
    if type(reloadIt) == type(None):
        if not os.path.isfile(reloadIt):
            saveName = np.copy(reloadIt)
            reloadIt = None
            
    # Pull it if actually exists    
    if type(reloadIt) == type(None):
        bigMassRes = {} # index by shape then inst
        for awf in wfTypes:
            bigMassRes[awf] = {}
            for aInst in wombatRes.keys():
                if awf in wombatRes[aInst]:
                    bigMassRes[awf][aInst] = {}
                    bigMassRes[awf][aInst]['times'] = []
                    bigMassRes[awf][aInst]['masses'] = []
                    bigMassRes[awf][aInst]['heights'] = []

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
                            myh = wombatRes[aInst][awf]['params'][wombatRes[aInst][awf]['timesSTR'][i]][0]
                            if wombatRes[aInst][awf]['pickles'][i] in idsbyPickle.keys():
                                idsbyPickle[wombatRes[aInst][awf]['pickles'][i]].append(wombatRes[aInst][awf]['ids'][i]+1)
                                bigMassRes[awf][aInst]['heights'].append(myh)
                            else:
                                idsbyPickle[wombatRes[aInst][awf]['pickles'][i]] = [wombatRes[aInst][awf]['ids'][i]+1]
                                bigMassRes[awf][aInst]['heights'] = [myh]
         
                            
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
                        #print (dargs)
                        #if aInst == 'HI2A_SR':
                        if True:
                            massRes, aboutMe = dingoWrapper(dargs, pullMass=True, silent=True)     
                            for i in range(len(aboutMe)):
                                mydeets = aboutMe[i].split()
                                for j in range(len(massRes[i])):
                                    bigMassRes[mydeets[2+j]][mydeets[1]]['times'].append(mydeets[0])
                                    bigMassRes[mydeets[2+j]][mydeets[1]]['masses'].append(massRes[i][j])
                        
        with open(saveName, 'wb') as file:
            pickle.dump(bigMassRes, file)

            
    # |-----------------------------------|
    # |--- Alternatively reload masses ---|
    # |-----------------------------------|        
    else:
        with open(reloadIt, 'rb') as file:
            bigMassRes = pickle.load(file)
    
    # |-------------------------------|
    # |--- Package masses by shape ---|
    # |-------------------------------|
    times = {}
    masses = {}
    mheights = {}
    for awf in wfTypes:
        allts = {}
        allMs = {}
        allhs = {}
        for aInst in bigMassRes[awf]:
            mySat = inst2sat[aInst]
            if mySat not in allts.keys():
                allts[mySat] = []
                allMs[mySat] = []
                allhs[mySat] = []
            for i in range(len(bigMassRes[awf][aInst]['times'])):
                mytime = datetime.datetime.strptime(bigMassRes[awf][aInst]['times'][i], "%Y-%m-%dT%H:%M:%S")
                allts[mySat].append(mytime.replace(second=0))
                allMs[mySat].append(bigMassRes[awf][aInst]['masses'][i]*1e15)
                allhs[mySat].append(bigMassRes[awf][aInst]['heights'][i])

        for aSat in allts.keys():
            allts[aSat] = np.array(allts[aSat])
            allMs[aSat] = np.array(allMs[aSat])
            allhs[aSat] = np.array(allhs[aSat])
            idxs  = np.argsort(allts[aSat])
            allts[aSat] = allts[aSat][idxs]
            allMs[aSat] = allMs[aSat][idxs]
            allhs[aSat] = allhs[aSat][idxs]
        
        times[awf] = allts
        masses[awf] = allMs
        mheights[awf] = allhs
        
    # Get a dt analogous to kin calc, might be useful
    earlyT = datetime.datetime(3000,1,1)
    for awf in wfTypes:
        for aSat in times[awf].keys():
            if times[awf][aSat][0] < earlyT:
                earlyT = times[awf][aSat][0]   
    dts = {}
    for awf in wfTypes:
        dts[awf] = {}
        for aSat in times[awf].keys():
            dts[awf][aSat] = []
            for i in range(len(times[awf][aSat])):
                dts[awf][aSat].append((times[awf][aSat][i]-earlyT).total_seconds())
            
    
    # |---------------------------|
    # |--- Match to velocities ---|
    # |---------------------------|
    # Velocities are at mid times bc derivatives
    # but want to match to masses from the img times
    # Interp vel to mass time bc vel is probably less
    # uncertain/wildly varying
    vels  = {}
    KEs   = {}
    errs  = {}

    for awf in wfTypes:
        vels[awf]    = {}
        KEs[awf]     = {}
        errs[awf]    = {}
        for aSat in times[awf].keys():
            Mtimes = times[awf][aSat]
            Ms = masses[awf][aSat]
            vtimes = kinRes['times'][awf]
            vs = kinRes['vels'][awf]
            verrs = kinRes['errs'][1][awf]
            matchvs = []
            nowKEs = []
            MuncsL = []
            MuncsH = []
            KEuncsH = []
            KEuncsL = []
                        
            for i in range(len(Mtimes)):
                if Mtimes[i] in vtimes:
                    idx = np.where(vtimes == Mtimes[i])[0][0]
                    if idx == 0:
                        myv = vs[0]
                        myverr = verrs[0]
                    elif idx == len(vs):
                        myv = vs[-1]
                        myverr = verrs[-1]
                    else:
                        myv = 0.5*(vs[idx] + vs[idx-1])
                        myverr = 0.5*(verrs[idx] + verrs[idx-1])
                    
                    matchvs.append(myv)
                    nowKEs.append(0.5*Ms[i] * myv**2)
                    #print (awf, aSat, Mtimes[i], idx, len(vs), myv/1e5, Ms[i], nowKEs[i], prevhs[i])
                    
                    # add in errors
                    MuncsL.append(0.5*Ms[i])
                    MuncsH.append(2*Ms[i])
                    KEuncsL.append(np.sqrt((MuncsL[-1] * 0.5*myv**2)**2 + (myverr * Ms[i]*myv)**2))
                    KEuncsH.append(np.sqrt((MuncsH[-1] * 0.5*myv**2)**2 + (myverr * Ms[i]*myv)**2))
                    
                else:
                    matchvs.append(None)
                    nowKEs.append(None)
                    MuncsL.append(None)
                    MuncsH.append(None)
                    KEuncsL.append(None)
                    KEuncsH.append(None)
                    
            vels[awf][aSat] = np.array(matchvs)
            KEs[awf][aSat]  = np.array(nowKEs)
            errs[awf][aSat] = np.array([np.array(MuncsL), np.array(MuncsH), np.array(KEuncsL), np.array(KEuncsH)])
        
    # |-----------------------|
    # |--- Package Output  ---|
    # |-----------------------|
    outRes = {}
    outRes['times'] = times
    outRes['dts'] = dts
    outRes['heights'] = mheights
    outRes['masses'] = masses    
    outRes['vels'] = vels
    outRes['KEs'] = KEs
    outRes['errs'] = errs
    
    return outRes

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
        myType = inst2type[aInst]
        
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
            wombatRes[aInst][aWF]['type'] = myType
            wombatRes[aInst][aWF]['ids'] = myIds
            wombatRes[aInst][aWF]['params'] = {}
            wombatRes[aInst][aWF]['times'] = []
            wombatRes[aInst][aWF]['timesSTR'] = []
            wombatRes[aInst][aWF]['pickles'] = []
            for aIdx in myIds:
                wombatRes[aInst][aWF]['times'].append(datetime.datetime.strptime(miniLog[aIdx,2], "%Y-%m-%dT%H:%M:%S" ))
                wombatRes[aInst][aWF]['pickles'].append(miniLog[aIdx,13])
                myParams = miniLog[aIdx,4:13]
                myParams = myParams[myParams != 'None'].astype(float)
                wombatRes[aInst][aWF]['params'][miniLog[aIdx,2]] = myParams
                wombatRes[aInst][aWF]['timesSTR'].append(miniLog[aIdx,2])
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

# |--------------------------|
# |--- Process Bonus Args ---|
# |--------------------------|
def processBonusArgs(args, mode):
    # Set defaults
    dragHeights = [5,21.5]
    errorbars = False
    incDrag   = False
    logIt     = False
    minVal    = None
    maxVal    = None
    massPkl   = None
    outName   = None
    overlap   = 1
    reloadIt  = None
    versusH   = False
    wfColors  = False 
    
    for val in args:
        lval = val.lower()
        # |--- Contour limits ---|        
        if 'min' in lval:
            try:
                minVal = int(lval.replace('min',''))
            except:
                print ('Cannot convert', lval, 'into integer minimum value')

        elif 'max' in lval:
            try:
                maxVal = int(lval.replace('max',''))
            except:
                print ('Cannot convert', lval, 'into integer maximum value')
                
        # |--- Overlap info for splitting wfs ---|        
        elif 'ovl' in lval:
            try:
                overlap = float(lval.replace('ovl',''))
            except:
                print ('Cannot convert', lval, 'into overlap value')
        elif 'overlap' in lval:
            try:
                overlap = float(lval.replace('overlap',''))
            except:
                print ('Cannot convert', lval, 'into overlap value')
                
        # |--- Drag heights ---|        
        elif 'dh1_' in lval:
            try:
                dragHeights[0] = float(lval.replace('dh1_',''))
            except:
                print ('Cannot convert', lval, 'into lower drag height')
        elif 'dh2_' in lval:
            try:
                dragHeights[1] = float(lval.replace('dh2_',''))
            except:
                print ('Cannot convert', lval, 'into upper drag height')
        
        # |--- Other flags ---|
        elif lval in ['eb', 'ebs', 'errors', 'errorbars', 'unc', 'uncs']:
            errorbars = True        

        elif lval in ['drag', 'incdrag', 'fitdrag', 'incldrag']:
            incDrag = True        

        elif lval in ['log', 'logit', 'logh', 'logR', 'logheight', 'logd', 'logdist', 'logradius', 'logdistance']:
            logIt = True        

        elif lval in ['versush', 'versusr', 'versusd', 'vsh', 'vsr', 'vsd',]:
            versusH = True        

        elif lval in ['wfcolors', 'guicolors']:
            wfColors = True        
        
        elif lval in ['png', 'pdf', '.png', '.pdf']:
            if lval[0] != '.':
                lval = '.' + lval
            picType = lval # set as global, don't need to return it
        
        # |--- Check if a pickle to save mass calc ---|
        elif '.pkl' in lval:
            if type(massPkl) != type(None):
                print('Have at least two inputs with .pkl')
                print('Cannot assign both to massPkl so doublecheck: ')
                print('   ', massPkl, val)
            else:
                massPkl = val
        # |--- Assume is output name if not a match to above ---|
        else:
            if type(outName) != type(None):
                print('Have at least two inputs that cannot be matched to specific keywords')
                print('Cannot assign both to outputName so doublecheck: ')
                print('   ', outName, val)
                sys.exit()
            else:
                outName = val

    return dragHeights, errorbars, incDrag, logIt, minVal, maxVal, massPkl, outName, overlap, reloadIt, versusH, wfColors
# |-------------------------|
# |--- Line profile plot ---|
# |-------------------------|
def profilePlot(mode, wombatRes, wfTypes, logH=False, wfColors=False, enRes=None, kinRes=None, versusH=False, errorbars=False, incDrag=False, outName='wombatProfile'):
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
        kinAxId = len(yLabelsL)
        yLabelsL = np.append(yLabelsL, ['v (km/s)', 'a (km/s$^2$)'])
        lColors[nParams+1] = 'k'
        lColors[nParams] = 'k'
        nParams += 2
    
    #|-------------------------|
    #|--- Add in energetics ---|
    #|-------------------------|
    if ('en' in mode):
        yLabelsL = np.append(yLabelsL, ['M (g)', 'KE (erg)'])
        lColors[nParams+1] = 'k'
        lColors[nParams] = 'k'
        nParams += 2
    
        
    #|-------------------------|
    #|--- Set up the figure ---|     
    #|-------------------------|
    if versusH:
        nParams = nParams  - 1
        hlab = yLabelsL[0]
        yLabelsL = yLabelsL[1:]
        yLabelsR = yLabelsR[1:]
        kinAxId -= 1
        for awf in id2id:
            myids = id2id[awf]
            newids = np.copy(myids)
            newids[0] = -1 # height go bye bye
            newids[1:] = newids[1:] - 1
            id2id[awf] = newids

    
    np2sz = {1:3, 2:5, 3:6, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:10, 11:10}
    fig, ax = plt.subplots(nParams, 1, figsize=(7,np2sz[nParams]), layout='constrained', sharex=True)
    if versusH:
        ax[-1].set_xlabel(hlab)  
    if nParams == 1:
        ax = [ax]
        
    axR = {}
    
    #|--- Set up the left labels ---|
    for i in range(nParams):
        if yLabelsL[i] != '': 
            thisLab = yLabelsL[i]
            if thisLab in labSwap.keys():
                thisLab = labSwap[thisLab] 
            ax[i].set_ylabel(thisLab, color=lColors[i])

    #|--- Set up the right labels ---|
    for i in range(len(yLabelsR)):
        # Single right label
        if yLabelsR[i] not in ['nullnullnullnull', 'double', '']:
            nowC = rColors[i]
            axR[i] = ax[i].twinx() 
            thisLab = yLabelsR[i]
            if thisLab in labSwap.keys():
                thisLab = labSwap[thisLab]
            axR[i].set_ylabel(thisLab, color=nowC)
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
                obsType = wombatRes[aInst][aType]['type']
                myRes = subRes[aType]
                myTimes = myRes['times']
                myC = pltColors[aType]
                for i in range(len(id2id[aType])):
                    myax = None
                    if id2id[aType][i] > -1:
                        myax = ax[id2id[aType][i]]
                        myErr = labErrs[yLabelsL[id2id[aType][i]]]
                        # Check if height dictionary or float/int
                        if not isinstance(myErr, (int, float)):
                            myErr = labErrs[yLabelsL[id2id[aType][i]]][obsType]
                    elif aType in id2idR:
                        if id2idR[aType][i] > -1:
                            myax = axR[id2idR[aType][i]]
                            # Height should never be on right
                            myErr = labErrs[yLabelsL[id2id[aType][i]]]
                                              
                    if type(myax) != type(None):    
                        xdata = myTimes
                        ydata = myRes['paramGrid'][i,:]
                        if versusH:
                            xdata = myRes['paramGrid'][0,:]
                            #ydata = myRes['paramGrid'][i+1,:]
                                                                           
                        if aType not in hasLabel:
                            myax.plot(xdata, ydata, 'o', c=myC, label=aType)
                            hasLabel.append(aType)
                        else:
                            myax.plot(xdata, ydata, 'o', c=myC)
                        if errorbars:
                            myax.errorbar(xdata, ydata, yerr=myErr, c=myC, capsize=3, fmt='none')
    if logH:
        if versusH:
            ax[0].set_xscale('log')
        else:
            ax[0].set_yscale('log')
                        
    ax[0].legend(loc='lower right', bbox_to_anchor=(1., 1.), ncols=len(hasLabel))
    
    #|-------------------------|
    #|--- Add in kinematics ---|     
    #|-------------------------|
    if ('kin' in mode) or ('en' in mode):     
        for awf in kinRes['times']:
            myC = pltColors[awf]
            myTimes = kinRes['times'][awf]
            myHeights = kinRes['heights'][awf]
            # Get midpoint times for v
            midTimes = []
            midHs    = []
            for i in range(len(myTimes)-1):
                delt_time = (myTimes[i+1] - myTimes[i]).total_seconds()
                midtime = myTimes[i] + datetime.timedelta(seconds=0.5*delt_time)
                midTimes.append(midtime)
                midHs.append(0.5 * (myHeights[i+1] + myHeights[i])/7e10)
                
            # Plot vels     
            myVels =  kinRes['vels'][awf]
            if versusH:
                xdata = midHs
            else:
                xdata = midTimes
            ax[kinAxId].plot(xdata, myVels/1e5, 'o', c=myC)
            if errorbars:
                ax[kinAxId].errorbar(xdata, myVels/1e5, yerr=kinRes['errs'][1][awf]/1e5, c=myC, capsize=3, fmt='none')
                
            # Plot accels
            myAccs =  kinRes['accs'][awf]
            accts  = myTimes[:-2]
            if versusH:
                xdata = kinRes['heights'][awf][1:-1] / 7e10
            else:
                xdata = myTimes[1:-1]
            ax[kinAxId+1].plot(xdata, myAccs/1e5, 'o', c=myC)
            if errorbars:
                ax[kinAxId+1].errorbar(xdata, myAccs/1e5, yerr=kinRes['errs'][2][awf]/1e5, c=myC, capsize=3, fmt='none')
            
            # Drag fit
            if incDrag:
                try:
                    myParams = kinRes['dragFits'][awf]
                    v1, v2, v3, v4, td0 = myParams[0]
                    delt_tds = []
                    tds = []
                    hds = []
                    for i in range(len(myTimes)):
                        if myTimes[i] > td0:
                            delt_dtime = (myTimes[i] - td0).total_seconds()
                            delt_tds.append(delt_dtime)
                            tds.append(myTimes[i])
                            hds.append(myHeights[i]/7e10)
                    delt_tds = np.array(delt_tds)
                    tds = np.array(tds)
                    mydragv = vdrag(delt_tds,v1, v2, v3)
                    mydraga = -v3*vdragScalers[2]*mydragv**2
                    if versusH:
                        xdata = hds
                    else:
                        xdata = tds
                    
                    ax[kinAxId].plot(xdata, mydragv/1e5, 'w',  zorder=20)
                    ax[kinAxId+1].plot(xdata, mydraga/1e5, 'w',  zorder=20)
                    ax[kinAxId].plot(xdata, mydragv/1e5, '--', c=myC, zorder=21)
                    ax[kinAxId+1].plot(xdata, mydraga/1e5, '--', c=myC, zorder=21)
                except:
                    print('Cannot add drag fit for ', awf)
        
        # Add dashed line for zero accel
        xlims = np.array(ax[kinAxId+1].get_xlim())
        if versusH and (xlims[0] < 1):
            xlims[0] = 1
        ax[kinAxId+1].plot(xlims, [0,0], 'k--', zorder=0)
        ax[kinAxId+1].set_xlim(xlims)
        #ax[kinAxId+1].set_ylim([-750, 750])
        enAxId = kinAxId+2
    
    #|-------------------------|
    #|--- Add in energetics ---|     
    #|-------------------------| 
    satsyms = ['*', '^', '+']
    if ('en' in mode): 
        for awf in kinRes['times']:
            myC = pltColors[awf]
            myTimes = enRes['times'][awf]
            myMass  = enRes['masses'][awf]
            myHs    = enRes['heights'][awf]
            theseSats = np.array(list(myMass.keys()))
            for i in range(len(theseSats)):
                aSat = theseSats[i]
                goodIdx = np.where(enRes['masses'][awf][aSat] > 0)
                goodts = enRes['times'][awf][aSat][goodIdx]
                goodHs = enRes['heights'][awf][aSat][goodIdx]
                goodMs = enRes['masses'][awf][aSat][goodIdx]
                goodKEs = enRes['KEs'][awf][aSat][goodIdx]
                
                if versusH:
                    xdata = goodHs
                else:
                    xdata = goodts
                
                if aSat not in hasLabel:
                    ax[enAxId].plot(xdata, goodMs, c='k', marker=satsyms[i], lw=0, label=str(aSat))
                    hasLabel.append(aSat)

                ax[enAxId].plot(xdata, goodMs, c=myC, marker=satsyms[i], lw=0)
                ax[enAxId+1].plot(xdata, goodKEs, c=myC, marker=satsyms[i], lw=0)
                
                if errorbars:
                    ml, mh = enRes['errs'][awf][aSat][0][goodIdx], enRes['errs'][awf][aSat][1][goodIdx]
                    kel, keh = enRes['errs'][awf][aSat][2][goodIdx], enRes['errs'][awf][aSat][3][goodIdx]
                    ax[enAxId].errorbar(xdata, goodMs, yerr=[ml,mh], c=myC, marker=satsyms[i], capsize=3, fmt='none')
                    ax[enAxId+1].errorbar(xdata, goodKEs, yerr=[kel,keh], c=myC, marker=satsyms[i], capsize=3, fmt='none')
                
        
                    
        ax[enAxId].legend(loc='lower right', bbox_to_anchor=(1., 1.), ncols=len(hasLabel))
        ax[enAxId].set_yscale('log')
        ax[enAxId+1].set_yscale('log')   
        # Add dashed lines at int powers
        for i in range(2):
            ylims = ax[enAxId+i].get_ylim()
            xlims = ax[enAxId+i].get_xlim()
            toadd = np.log10(np.array(ylims)).astype(int)
            
            for j in np.arange(toadd[0], toadd[1]+1):
                myval = np.power(10,j,dtype=float)
                ax[enAxId+i].plot(xlims, [myval,myval], 'k:')      
            
    
    if not versusH:
        date_form = mdates.DateFormatter("%m-%d %H:%M")
        ax[-1].xaxis.set_major_formatter(date_form)
        fig.autofmt_xdate()
    else:
        if logH:
            ax[-1].xaxis.set_major_formatter(ScalarFormatter())

    
    #plt.show()
    print ('Saving figure as', outName+picType)
    plt.savefig(outName+picType)

    

def wombatPlotWrapper(args):
    #|----------------------------------|
    #|--- Check the number of inputs ---|     
    #|----------------------------------|
    nArgs = len(args)
    if (nArgs <3):
        print ('Incorrect number of parameters provided. Syntax is')
        for astr in errorStrings:
            print (astr)
        sys.exit()
        
    
    #|-------------------------------------|
    #|--- Check the critical parameters ---|     
    #|-------------------------------------|
    miniLog, wombatRes, mode, uniqTs, wfTypes, allInsts = processArgs(args)
    
    #|----------------------------------|
    #|--- Check any bonus parameters ---|     
    #|----------------------------------|
    # Set defaults
    if len(args) > 4:
       allBonus = args[3:]
       dragHeights, errorbars, incDrag, logIt, minVal, maxVal, massPkl, outName, overlap, reloadIt, versusH, wfColors  = processBonusArgs(allBonus, mode)
    
    
    #|--------------------------|
    #|--- Process kinematics ---|     
    #|--------------------------|
    kinRes = None
    if ('kin' in mode) or ('en' in mode):
        kinRes = getKinematics(wombatRes, wfTypes, incDrag=incDrag)

    #|--------------------------|
    #|--- Process energetics ---|     
    #|--------------------------| 
    enRes = None   
    if ('en' in mode):
        enRes = getEnergetics(args, wombatRes, wfTypes, kinRes, reloadIt=massPkl)
        #enRes = getEnergetics(args, wombatRes, wfTypes, kinRes)

        
    #|--------------------------|
    #|--- Run line plot mode ---|     
    #|--------------------------|
    if mode in ['ht1', 'ht2','ht3', 'kin1', 'kin2', 'kin3', 'en1', 'en2', 'en3']:
        profilePlot(mode, wombatRes, wfTypes, logH=logIt, wfColors=wfColors, kinRes=kinRes, enRes=enRes, errorbars=errorbars, versusH=versusH, incDrag=incDrag)
        


# |-----------------------|
# |--- Text line input ---|
# |-----------------------|
if __name__ == '__main__':
    wombatPlotWrapper(sys.argv[1:])
