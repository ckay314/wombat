"""
Set of functions for making standardized plots from WOMBAT log lines and
a wrapper to call them from the command line. The script will use DINGO
to calculate masses as needed, which can take some time. It does include
the option to save the DINGO results in a pkl so they can be reloaded
after the first calcuation to allow for quick manipulation of plot 
aesthetics.


The command line syntax is

    python3 wombatPlots.py logFile lineIDs mode [optional parameters]

where 

    logFile:  points to a log file created by WOMBAT

    lineIDs:  can be a single integer (e.g. 4)
              a range of integers (e.g. 4-10)
              or a series of integers connected by + (e.g. 4+8+12)

    mode:    select from one of the following plot mode tags

             Line plot variants:
                ht# - just height (# is parameter config)
                kin# - height, velocity, accceleration
                en# - height, vel, acc, mass, kinetic energy

                # sets which wireframe params are also show
                1 - just height
                2 - height + angular width(s)
                3  - all wf params

                e.g. kin2 would show height, AW, vel, and acc
            
             pts - print the Cartesian Stonyhust location all of the wf 
                   points to a file

             vmap - determine the velocity from two timesteps for the same wf
                    (using their diff heights and delta t). Makes a 3d scatter
                    plot with each wf point colored by its velocity and prints
                    the cartesian locations and total velocity to a file

and the optional arguments fall into the following types
Direct flags (written as is):
        eb:         flag to include error bars/uncertainties in figures
                    (defaults to False)

        drag:       flag to fit the drag equation to the reconstructions
                    (defaults to False)

        vsh:        flag to plot versus height instead of versus time
                    (defaults to False)
        
        log:        flag to plot x-axis heights on a log scale
                    (defaults to False)
        
        
        wfcolors:   flag to use the same colors as the wombat GUI instead of the
                    standard plot colors that are more suited for line plots on a
                    white backgrounds
                    (defaults to False)

        png/pdf:    flag to save figures as png or pdf
                    (defaults to png)

        1au/L1:     flag to get a predicted arrival time at either L1 or 1 AU
                    (defaults to doing neither)


Keys with numbers (# replaced by float or time)
        densratio_#: the ratio between the inner and outer wireframe densities (n1/n2).
                     the densities vary from pixel to pixel but the ratio between the 
                     two remains the same in any overlapping regions. can be a decimal
                     (defaults to 1.)

        dh1_#/dh2_#: the min/max heights to set the range over which the drag calculation
                     looks for an optimal starting height (in Rs)
                     (defaults to 5 Rs/21.5 Rs)

        newbase_#:  a new time to switch to the base time for the mass calculation. the
                    # should be replaced by a time stamp that parse_time can process
        

    Other:
        pickleName - the name of a pickle with the saved results of previous DINGO
                     calculation. These are automatically saved by getEnergetics in
                     wbPlotPickles/ using saveName or defaulting to bigMassRes.pkl
                     It automatically searches this directory so it should just be
                     the file name.

        saveName - if an arugment is passed that does not fit any of the other tag types
                   then it is assumed to be a save name for any output images/files
"""


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

#|----------------------|
#|--- Set up globals ---|
#|----------------------|
global errorStrings, labSwap,labErrs, labelMatch, inst2sat, labErrs, hErrs

#|--- Standard input error message ---|
errorStrings = [' ', '  python3 wombatPlots.py logFile id(s) type otherParams', '          where:', '          - logFile is a wombat log file', '          - ids is an integer or int+int or int-int', '          - type sets the plot type from [ht, kin, en, linimg, logimg, sqimg]', '          - otherParams includes min#, max#, outName, logIt, GUIcolors, and picType']


# |--- Reconstruction errors ---|
# Set up generic height values based on inst type
# Other parameters are const
hErrs = {'EUV':0.05, 'COR':0.1, 'HI':0.5}
labErrs = {'Height (Rs)':hErrs, 'Lon (deg)':10, 'Lat (deg)': 5, 'Tilt (deg)':15, 'AW (deg)':5, 'kappa':0.05, 'AW_FO (deg)':5, 'AW_EO (deg)':5, 'deltaAx':0.1, 'deltaCS':0.1, 'ecc1':0.1, 'ecc2':0.1, 'Roll (deg)':15, 'Yaw (deg)':15, 'Pitch (deg)':15, 'Lx (Rs)':hErrs, 'Ly (Rs)':hErrs, 'Lz (Rs)':hErrs, 'HeightO (Rs)':0.1, 'LonO (deg)':5, 'LatO (deg)':5}


# |--- GUI labels to plot labels ---|
# GUI doesn't like latex but plots do
deg = '($^{\\circ}$)'
labSwap = {'Height (Rs)':'Height (R$_S$)', 'Lon (deg)':'Lon'+deg , 'Lat (deg)': 'Lat'+deg, 'Tilt (deg)':'Tilt'+deg, 'AW (deg)':'AW'+deg, 'kappa':'$\\kappa$', 'AW_FO (deg)':'AW$_{FO}$'+deg, 'AW_EO (deg)':'AW$_{EO}$'+deg, 'deltaAx':'$\\delta_{Ax}$', 'deltaCS':'$\\delta_{CS}$', 'ecc1':'$\\epsilon_1$', 'ecc2':'$\\epsilon_2$', 'Roll (deg)':'Roll'+deg, 'Yaw (deg)':'Yaw'+deg, 'Pitch (deg)':'Pitch'+deg, 'Lx (Rs)':'L$_x$ (R$_S$)', 'Ly (Rs)':'L$_y$ (R$_S$)', 'Lz (Rs)':'L$_z$ (R$_S$)', 'HeightO (Rs)':'Height$_O$ (R$_S$)', 'LonO (deg)':'Lon$_O$'+deg, 'LatO (deg)':'Lat$_O$'+deg}


# |--- Dictionary for label pairing ---|
# Who to pair things with, GCS* is longest, followed by Slab
# Only match nice pairs, otherwise let it dump them in wherever
labelMatch = {'Tilt (deg)':['Roll (deg)'], 'AW (deg)': ['AW_FO (deg)', 'Lx (Rs)'], 'AW_FO (deg)':['AW (deg)', 'Lx (Rs)'],  'AW_EO (deg)':['AW (deg)', 'Ly (Rs)'], 'kappa':['ecc1', 'deltaAx',  'Yaw (deg)'], 'ecc1':['kappa', 'deltaAx', 'Yaw (deg)'],  'ecc2': ['deltaCS', 'Pitch (deg)'], 'deltaAx':['kappa', 'ecc1', 'Yaw (deg)'], 'deltaCS':['ecc2', 'Pitch (deg)']}

# |--- Dictionary of instruments by sat ---|
inst2sat = {'AIA94':'SDO', 'AIA131':'SDO', 'AIA171':'SDO','AIA193':'SDO','AIA211':'SDO','AIA304':'SDO','AIA335':'SDO','AIA1600':'SDO','AIA1700':'SDO', 'C2':'SOHO', 'C3':'SOHO', 'COR1':'STEREO', 'COR2':'STEREO', 'COR1A':'STEREOA', 'COR2A':'STEREOA', 'COR1B':'STEREOB', 'COR2B':'STEREOB', 'EUVI171':'STEREO', 'EUVI195':'STEREO', 'EUVI284':'STEREO', 'EUVI304':'STEREO', 'EUVI171A':'STEREOA', 'EUVI195A':'STEREOA', 'EUVI284A':'STEREOA', 'EUVI304A':'STEREOA', 'EUVI171B':'STEREOB', 'EUVI195B':'STEREOB', 'EUVI284B':'STEREOB', 'EUVI304B':'STEREOB', 'HI1':'STEREO', 'HI2':'STEREO', 'HI1A':'STEREOA', 'HI2A':'STEREOA', 'HI1B':'STEREOB', 'HI2B':'STEREOB', 'HI1A_SR':'STEREOA', 'HI1B_SR':'STEREOB', 'HI2A_SR':'STEREOA', 'HI2B_SR':'STEREOB', 'SOLOHI':'SOLO', 'SOLOHI1':'SOLO', 'SOLOHI2':'SOLO', 'SOLOHI3':'SOLO', 'SOLOHI4':'SOLO', 'WISPR':'PSP', 'WISPRI':'PSP', 'WISPRO':'PSP', 'WISPR_LW':'PSP', 'WISPRI_LW':'PSP', 'WISPRO_LW':'PSP', 'WISPR_L3':'PSP', 'WISPRI_L3':'PSP', 'WISPRO_L3':'PSP'}

# |--- Dictionary for inst type ---|
inst2type = {'AIA94':'EUV', 'AIA131':'EUV', 'AIA171':'EUV','AIA193':'EUV','AIA211':'EUV','AIA304':'EUV','AIA335':'EUV','AIA1600':'EUV','AIA1700':'EUV', 'C2':'COR', 'C3':'COR', 'COR1':'COR', 'COR2':'COR', 'COR1A':'COR', 'COR2A':'COR', 'COR1B':'COR', 'COR2B':'COR', 'EUVI171':'EUV', 'EUVI195':'EUV', 'EUVI284':'EUV', 'EUVI304':'EUV', 'EUVI171A':'EUV', 'EUVI195A':'EUV', 'EUVI284A':'EUV', 'EUVI304A':'EUV', 'EUVI171B':'EUV', 'EUVI195B':'EUV', 'EUVI284B':'EUV', 'EUVI304B':'EUV', 'HI1':'HI', 'HI2':'HI', 'HI1A':'HI', 'HI2A':'HI', 'HI1B':'HI', 'HI2B':'HI', 'HI1A_SR':'HI', 'HI1B_SR':'HI', 'HI2A_SR':'HI', 'HI2B_SR':'HI', 'SOLOHI':'HI', 'SOLOHI1':'HI', 'SOLOHI2':'HI', 'SOLOHI3':'HI', 'SOLOHI4':'HI', 'WISPR':'HI', 'WISPRI':'HI', 'WISPRO':'HI', 'WISPR_LW':'HI', 'WISPRI_LW':'HI', 'WISPRO_LW':'HI', 'WISPR_L3':'HI', 'WISPRI_L3':'HI', 'WISPRO_L3':'HI'}


global vdragScalers 
vdragScalers = [600e5, 350e5, 1e-12]

global picType
picType = '.png' # png or pdf, gets overwritten if set in input 

# |-----------------------------------------|
# |--- Normalized rag func (for fitting) ---|
# |-----------------------------------------|
def vdrag(t_in, vCME0_in, vSW_in, C_in): 
    # Inputs are normalized to near 1 to make the 
    # curve_fit happy, convert back to physical units 
    # using vdragScalers (cm, cm, 1/cm)
    # e.g. vCME0_in of 1 is actually 600 km/s
    C = C_in * vdragScalers[2]
    t = t_in 
    vCME0 = vCME0_in * vdragScalers[0]
    vSW   = vSW_in * vdragScalers[1]
    vout = (vCME0 - vSW) / (1 + C * (vCME0 - vSW) * t) + vSW 
    return vout

# |-----------------------------------|
# |--- Get kinematics from results ---|
# |-----------------------------------|
def getKinematics(wombatRes, wfTypes, dragHeights=[5,21.5], incDrag=False, predAT=None):
    """
    Function to derive the kinematics (velocity and acceleration) from a set
    of wombat results. It will also fit a basic drag equation to the reconstructions
    and can predict the arrival time at L1 or 1 AU if flagged to do so.
    
    Inputs:
        wombatRes: a dictionary filled by info pulled from miniLog in the format
                    wombatRes[instName][wfType] = {}
                    keys - ids, params, times, pickles -> arrays of all the
                           values for each time matching that inst/type
    
        wfTypes:   an array of the wf types to use
    
    Optional Inputs:
        dragHeights: the min/max heights to set the range over which the drag calculation
                     looks for an optimal starting height (in Rs)
                     (defaults to 5 Rs/21.5 Rs)
    
        incDrag:     a flag to fit the drag equation to the results
                     (defaults to False)
    
        predAT:      a flag to get a predicted arrival time at either L1 or 1 AU
                    (defaults to doing neither)
    
    Outputs: 
             the kinematic results are packaged into a dictionary outResK and returned. The 
             keys/entries are below. Each item is a dictionary with keys for each wf type
                times:    datetimes (direct from wombatRes)
    
                dt:       time difference in seconds from the earliest time for any wf
    
                heights:  the heights (direct from wombatRes) 
                          (in cm)
    
                vels:     the two-point derivatives calculated from the height/times 
                          this array is one shorter than heights/times and should be
                          compared to the midpoint values of the full h/t array
                          (in cm/s)

                accs:     the two-point derivatives calculated from vels/times
                          this array is two shorter than heights/times and should be
                          compared to idx [1:-1] of the full h/t array
                          (in cm/s^2)

                errs:     arrays for the uncertainty in [heights, vels, accs] over time
                          (units are [cm, cm/s, cm/s^2])
    
                splitIds: dictionary of the index where the drag mode starts
                          (only included if incDrag is flagged)
    
                dragFits: dictionary with [[[vCME, vSW, Cdrag], [h0, t0]], [errors]] where
                          the first array is the drag equation parameters + starting height/time from
                          the fit and the second array is the corresponding errors. The first three are
                          in normalized units and need to be scaled back to physical units using the scaler
                          and h is in cm and t is a time with the error in s
                          (only included if incDrag is flagged)
    
    
    """
    # |---------------------|
    # |--- Preprocessing ---|
    # |---------------------|
    # Convert drag Heights to cm
    dragHeights = np.array(dragHeights)*7e10
    
    # |--- Make holders ---|
    times = {}
    dts   = {}
    heights = {}
    newtVs  = {} # newtonian derivs (x2 - x1) / (t2 - t1)
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
    # |--- Figure out the earliest time ---|
    earlyT = datetime.datetime(3000,1,1)
    lateT  = datetime.datetime(1000,1,1)
    for awf in wfTypes:
        if times[awf][0] < earlyT:
            earlyT = times[awf][0]
        if times[awf][-1] > lateT:
            lateT = times[awf][-1]
            
    #|--- Loop through wf types ---|        
    for awf in wfTypes:
        print ('Calculating two-point derivatives for', awf)
        #hSmooth = gaussian_filter1d(heights[awf], sigma=1) 
        hSmooth = heights[awf] # Not smoothing anymore but leaving structure
        myuncH  = uncHs[awf]
        for i in range(len(times[awf])):
            dts[awf].append((times[awf][i]-earlyT).total_seconds())
            # Get v
            if i != 0:
                newtVs[awf].append((hSmooth[i]-hSmooth[i-1])/(times[awf][i]-times[awf][i-1]).total_seconds())
                uncVs[awf].append(np.sqrt(myuncH[i]**2 + myuncH[i-1]**2)/(times[awf][i]-times[awf][i-1]).total_seconds()) # cm/s
                #print ((np.sqrt(myuncH[i]**2 + myuncH[i-1]**2))/7e10, (times[awf][i]-times[awf][i-1]).total_seconds()/60, uncVs[awf][-1]/1e5)
            # Get a    
            if i > 1:
                j = i -1
                newtAs[awf].append((newtVs[awf][j] - newtVs[awf][j-1])/(dts[awf][j] - dts[awf][j-1]))
                uncAs[awf].append(np.sqrt(uncVs[awf][j]**2 + uncVs[awf][j-1]**2)/(dts[awf][i]-dts[awf][i-1])) # cm/s
        
        # Package as arrays       
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
            # Get the mid heights/times that match the vs
            midH = 0.5*(heights[awf][1:] + heights[awf][:-1])
            midT = 0.5*(dts[awf][1:] + dts[awf][:-1])
            vSmooth = gaussian_filter1d(newtVs[awf], sigma=1)       
        
            # Consider all starting points in between heights given
            # by dragHeights
            aboveDH = np.where(midH >= dragHeights[0])[0]
            bestVal = 9e20 # arbitrary large
            bestId  = -1
            bestPs = None
            # Try all heights, keep the best one
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
    
    
    # |---------------------------|
    # |--- Arr Time Prediction ---|
    # |---------------------------|
    if type(predAT) != type(None):
        if predAT.lower() == 'l1':
            critx = 0.99
        elif predAT.lower() in ['1au', '1 au']:
            critx = 1.
        else:
            sys.exit('Arrival time point ' + predAT + ' not understood, exiting.')
        
        didDrag = []
        
        # Use drag fit results
        if incDrag:
            print('Calculating drag model arrival time at ' + predAT)
            for awf in wfTypes:
                if awf in dragFits:
                    myPs   = dragFits[awf][0]
                    myErrs = dragFits[awf][1]
                    v0, vsw, C = myPs[0] * vdragScalers[0], myPs[1] * vdragScalers[1], myPs[2] * vdragScalers[2] # cm/s, 1/cm
                    h0, t0 = myPs[3], myPs[4] # cm, datetime
                    ev, eh = myErrs[0] * vdragScalers[0], myErrs[3] 
                    
                    fakets = np.linspace(0,5*24*3600, 200) # 5 days in seconds
                    modx = (1/C * np.log(1 + C * (v0 - vsw)*fakets) + vsw * fakets + h0) / 1.496e+13 
                    
                    # get index around crit x
                    if (np.max(modx) > critx) and (np.min(modx) < critx):
                        idx1 = np.max(np.where(modx <= critx)[0])
                        idx2 = np.min(np.where(modx >= critx)[0])
                        smallts = np.linspace(fakets[idx1],fakets[idx2], 100)
                        modx2 = (1/C * np.log(1 + C * (v0 - vsw)*smallts) + vsw * smallts + h0) / 1.496e+13 
                        hitIdx = np.where(np.abs(modx2 - critx) == np.min(np.abs(modx2 - critx)))[0]
                        tt = smallts[hitIdx[0]]
                        artime = t0 + datetime.timedelta(seconds=tt)
                        # Get error estimate - just using x = v/t not full drag eq for now
                        # because not easily separable to get t(x) instead of x(t)
                        tdist = critx* 1.496e+13 - h0
                        aterr = np.sqrt((ev * tdist/v0**2)**2 + (eh/ v0)**2)
                        print ('   ', awf, 'impact at ', artime.strftime("%Y-%m-%d %H:%M"), '+/-', '{:.1f}'.format(aterr/3600.), ' hr' )
                        
                        didDrag.append(awf)
            print ("")
            
        # Non drag cases
        for awf in wfTypes:
            if len(didDrag) != len(wfTypes):
                print('Calculating simple '+awf+ ' arrival time at ' + predAT)
            if awf not in didDrag:
                # Use max v to calc
                maxv  = np.max(newtVs[awf])
                maxId = np.where(newtVs[awf] == maxv)[0]
                maxh  = heights[awf][maxId[0]]
                t0    = times[awf][maxId[0]]
                dist  = critx* 1.496e+13 - maxh
                
                tt_maxv = dist / maxv
                artimeMax = t0 + datetime.timedelta(seconds=tt_maxv)
                print ('   max  v gives impact at ', artimeMax.strftime("%Y-%m-%d %H:%M") )
                
                
                ipPoints = np.where(heights[awf] >= dragHeights[0])[0]
                if len(ipPoints) > 3:
                    meanv = np.mean(newtVs[awf][ipPoints[0]+1:])
                    tt_meanv = dist / meanv
                    artimeMean = t0 + datetime.timedelta(seconds=tt_meanv)
                    print ('   mean v gives impact at ', artimeMean.strftime("%Y-%m-%d %H:%M") )
                    
                    medv  = np.median(newtVs[awf][ipPoints[0]+1:])
                    tt_medv = dist / medv
                    artimeMed = t0 + datetime.timedelta(seconds=tt_medv)
                    print ('   med  v gives impact at ', artimeMed.strftime("%Y-%m-%d %H:%M") )
                    
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
def getEnergetics(args, wombatRes, wfTypes, kinRes, reloadIt=None, overlap=1, rebase=False):
    """
    Function to derive the energetics (mass and kinetic energy) from a set
    of wombat results. It sends the inputs to DINGO to calculated masses then
    combines these with the velocities to create energies. The mass times are at obs
    times but velocities are at midpoint/half step times so we interpolate the v from 
    the half step to the expected value at the mass/integer time. There is a little bit of
    run time for the DINGO calculation so it will save the DINGO results in a pkl so
    that it can be called again to adjust plot aesthetics without redoing the full calc
    
    Inputs:
        args:      the full set of arguments from the command line (everything after 
                   the .py). These are passed to DINGO since it uses the same keywords
                   and its easier to fake a command line call than pass specific variables
    
        wombatRes: a dictionary filled by info pulled from miniLog in the format
                    wombatRes[instName][wfType] = {}
                    keys - ids, params, times, pickles -> arrays of all the
                           values for each time matching that inst/type
    
        wfTypes:   an array of the wf types to use
    
        kinRes:    the results from getKinematics. a dictionary with keys times, dts, 
                   heights, vels, acs, errs, splitIds, dragFits
    
    Optional Inputs:
        reloadIt:   a pickle to reload previous results instead of calling DINGO
                    (defaults to None/not reloading)
        
        overlap:    the ratio between the inner and outer wireframe densities (n1/n2).
                    the densities vary from pixel to pixel but the ratio between the 
                    two remains the same in any overlapping regions. can be a decimal
                    (defaults to 1.)
        
        rebase:     a string flaging to use a new base time for the mass calculation. this
                    mimics a dingo command line argument so it has the form 
                    densratio_DATESTRING. DINGO will use the closest time as the new base
                    and recalculate the masses.
        
    Outputs: 
             the energetic results are packaged into a dictionary outRes and returned. The 
             keys/entries are below. Each item is a dictionary with keys for each wf type
                masses:   the mass in the projected wf region for each time
                          (in g)

                times:    datetimes (direct from wombatRes)
    
                dt:       time difference in seconds from the earliest time for any wf
    
                heights:  the heights (direct from wombatRes) 
                          (in cm)
    

                vels:     the velocities at the times matching the masses 
                          (in cm/s)

    
                KEs:      the kinetic energy calculated as 1/2 mass * vel^2
                          (units are ergs)

                errs:     arrays for the uncertainty as [mass_lower, mass_upper, KE_lower
                          KE_upper] where each is an array over all times. The general mass
                          error is set at a factor of 2 so it is not symmetric in +/-
                          (units are [cm, cm/s, cm/s^2])
    
    """
    
        
    # |--------------------------------|
    # |--- Use Dingo to calc masses ---|
    # |--------------------------------|
    # Check if passed reloadIt, could be an existing pkl or
    # a name to save the output for future use
    bmrDir = 'wbPlotPickles/'
    if not os.path.exists(bmrDir):
        os.mkdir(bmrDir)
        print ('Created output folder', bmrDir)
    
    # Check if have name and it exists    
    saveName = 'bigMassRes.pkl'
    if type(reloadIt) != type(None):
        if not os.path.isfile(bmrDir+reloadIt):
            saveName = str(np.copy(reloadIt))
            reloadIt = None
            
    # Make the bigMassRes if doesnt exists    
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

        
        # Dingo needs things sorted by background pickle. It is 
        # fine with two wireframes though
    
        # |--- Collect things ---|
        for aInst in wombatRes.keys():
            # Make sure not EUV
            if ('EUV' not in aInst.upper()) & ('AIA' not in aInst.upper()):
                print ('Calculating masses for', aInst)
                # Check if one or two res, can process together
                if len(wombatRes[aInst].keys()) >= 3:
                    overlap = 0
                    print ('More than two wfs for', aInst, ' cannot process as overlaping so doing individually')
                    
                # Collect all the ids for each pickle
                idsbyPickle = {}
                wfsbyPickle = {}
                for awf in wombatRes[aInst].keys():
                    for i in range(len(wombatRes[aInst][awf]['pickles'])):
                        myh = wombatRes[aInst][awf]['params'][wombatRes[aInst][awf]['timesSTR'][i]][0]
                        if wombatRes[aInst][awf]['pickles'][i] in idsbyPickle.keys():
                            idsbyPickle[wombatRes[aInst][awf]['pickles'][i]].append(wombatRes[aInst][awf]['OGids'][i]+1)
                            wfsbyPickle[wombatRes[aInst][awf]['pickles'][i]].append(awf)
                            bigMassRes[awf][aInst]['heights'].append(myh)
                        else:
                            idsbyPickle[wombatRes[aInst][awf]['pickles'][i]] = [wombatRes[aInst][awf]['OGids'][i]+1]
                            bigMassRes[awf][aInst]['heights'] = [myh]
                            wfsbyPickle[wombatRes[aInst][awf]['pickles'][i]] = [awf]
         
                # Format the array of integer indices to the string format 
                # that dingo wants (stringo)
                for key in idsbyPickle:
                    myids = np.array(idsbyPickle[key])
                    mywfs = np.array(wfsbyPickle[key])
                    nids = len(myids)
                    
                    # Single Id
                    if nids == 1:
                        strids = [str(myids[0])]
                    # Multis
                    else:
                        # Single WF
                        if overlap != 0:
                            strids = ''
                            for i in range(nids-1):
                                strids = strids + str(myids[i]) + '+'
                            strids = [strids + str(myids[-1])]
                        # Multi WF
                        else:
                            strids = []
                            nowwfs = np.unique(mywfs)
                            for awf in wombatRes[aInst].keys():
                                nowids = myids[np.where(mywfs == awf)[0]]
                                strid = ''
                                for i in range(len(nowids)-1):
                                    strid = strid + str(nowids[i]) + '+'
                                strid = strid + str(nowids[-1])
                                strids.append(strid)
                    # Save the string                    
                    idsbyPickle[key] = strids
                    
                    #|-----------------|
                    #|--- Run DINGO ---|
                    #|-----------------|
                    for strid in strids:
                        dargs = [args[0], strid, '0d']
                        # Add dens ratio to args
                        if overlap != 0:
                            ovlstr = 'densratio_'+str(overlap)
                            dargs.append(ovlstr)
                        # Add rebase to args
                        if rebase:
                            dargs.append(rebase)
                        # if statement to turn off actual mass calc when testing things    
                        if True:
                            massRes, aboutMe = dingoWrapper(dargs, pullMass=True, silent=True)     
                            for i in range(len(aboutMe)):
                                mydeets = aboutMe[i].split()
                                for j in range(len(massRes[i])):
                                    bigMassRes[mydeets[2+j]][mydeets[1]]['times'].append(mydeets[0])
                                    bigMassRes[mydeets[2+j]][mydeets[1]]['masses'].append(massRes[i][j])
        # Save it    
        with open(bmrDir+saveName, 'wb') as file:
            pickle.dump(bigMassRes, file)
        
        # Reload will determine which index we actually want 
        # from an existing pickle. Sicne we just made this we
        # want everyone but make a fake subIdx for everyone
        subIdx = {}
        for aInst in wombatRes.keys():
            subIdx[aInst] = {}
            for awf in wfTypes:
                if awf in wombatRes[aInst]:
                    subIdx[aInst][awf] = range(len(bigMassRes[awf][aInst]['times']))

            
    # |-----------------------------------|
    # |--- Alternatively reload masses ---|
    # |-----------------------------------|        
    else:
        # Open it
        with open(bmrDir+reloadIt, 'rb') as file:
            bigMassRes = pickle.load(file)
            
        # Need to potentially downselect depending on what lines given
        subIdx = {}
        for aInst in wombatRes.keys():
            subIdx[aInst] = {}
            for awf in wfTypes:
                subIdx[aInst][awf] = []
                if awf in wombatRes[aInst]:
                    for i in range(len(wombatRes[aInst][awf]['timesSTR'])):
                        myBMR = np.array(bigMassRes[awf][aInst]['times'])
                        if wombatRes[aInst][awf]['timesSTR'][i] in myBMR:
                            thisidx = np.where( myBMR == wombatRes[aInst][awf]['timesSTR'][i])[0]
                            subIdx[aInst][awf].append(thisidx[0])    
                    
 
 
    # |-------------------------------|
    # |--- Package masses by shape ---|
    # |-------------------------------|
    times = {}
    masses = {}
    mheights = {}
    # Loop through wf types
    for awf in wfTypes:
        allts = {}
        allMs = {}
        allhs = {}
        # Look through insts
        for aInst in wombatRes.keys():
            mySat = inst2sat[aInst]
            # Create if don't already have
            if mySat not in allts.keys():
                allts[mySat] = []
                allMs[mySat] = []
                allhs[mySat] = []
            # Add all the datas 
            for i in subIdx[aInst][awf]:
                mytime = datetime.datetime.strptime(bigMassRes[awf][aInst]['times'][i], "%Y-%m-%dT%H:%M:%S")
                allts[mySat].append(mytime.replace(second=0))
                allMs[mySat].append(bigMassRes[awf][aInst]['masses'][i]*1e15)
                allhs[mySat].append(bigMassRes[awf][aInst]['heights'][i])
        
        # Repackage as sorted array
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
                # Find a matching time
                if Mtimes[i] in vtimes:
                    idx = np.where(vtimes == Mtimes[i])[0][0]
                    if idx == 0:
                        myv = vs[0]
                        myverr = verrs[0]
                    elif idx == len(vs):
                        myv = vs[-1]
                        myverr = verrs[-1]
                    else:
                        # this is assuming uniform time steps, might want to fix
                        myv = 0.5*(vs[idx] + vs[idx-1])
                        myverr = 0.5*(verrs[idx] + verrs[idx-1])
                    
                    matchvs.append(myv)
                    nowKEs.append(0.5*Ms[i] * myv**2)
                    
                    # add in errors, 
                    MuncsL.append(0.5*Ms[i])
                    MuncsH.append(2*Ms[i])
                    KEuncsL.append(np.sqrt((MuncsL[-1] * 0.5*myv**2)**2 + (myverr * Ms[i]*myv)**2))
                    KEuncsH.append(np.sqrt((MuncsH[-1] * 0.5*myv**2)**2 + (myverr * Ms[i]*myv)**2))
                # Reject other wise, probably should improve this instead of 
                # just tossing out    
                else:
                    print ('nomatch')
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


# |------------------------------|
# |--- Print wireframe points ---|
# |------------------------------|
def printPoints(wombatRes, wfTypes, outName=None, morePts=2):
    """
    Function to print the Cartesian Stonyhurst location of the wireframe
    points to a text file
    
    Inputs:
        wombatRes: a dictionary filled by info pulled from miniLog in the format
                    wombatRes[instName][wfType] = {}
                    keys - ids, params, times, pickles -> arrays of all the
                           values for each time matching that inst/type
    
        wfTypes:   an array of the wf types to use
        
    Optional Inputs:
        outName:    a name to use for the save file. It will be saved in the wbOutputs 
                    folder as outName + '.txt'
                    (defaults to wombatPoints.txt)
    
        morePoints: flag to scale up the number of grid points used in the wireframe. Each
                    dimension of the wf grid is scaled up by this factor. e.g. a wf with
                    standard grid dimensions of [10,10,10] and a setting of 2 would become
                    [20,20,20]
                    (defaults to 2)
    
    Outputs:    Nothing is returned but it generates a text file with the following columns:
                    WFtype
                    time of observation
                    pointID
                    StonyCart x (Rs)
                    StonyCart y (Rs)
                    StonyCart z (Rs)
    
    """
    # Collect things 
    toDo = {}
    for awf in wfTypes:
        toDo[awf] = {}
        myTimes = []
        myParams = []
        for aInst in wombatRes:
            if awf in wombatRes[aInst].keys():
                for i in range(len(wombatRes[str(aInst)][awf]['timesSTR'])):
                    # remove seconds
                    myTime = wombatRes[str(aInst)][awf]['timesSTR'][i][:-3] 
                    if myTime not in myTimes:
                        myTimes.append(myTime)
                        myParams.append(wombatRes[str(aInst)][awf]['paramGrid'][:,i])
        toDo[awf]['times'] = myTimes
        toDo[awf]['params'] = myParams
    
    # Set up output file    
    if type(outName) == type(None):
        outName = 'wombatPoints'
    f1 = open('wbOutputs/'+outName+'.txt', 'w')
    print ('Saving points in wbOutputs/'+outName+'.txt')
    
    # Get points and print
    for awf in toDo:
        times = toDo[awf]['times']
        for i in range(len(times)):
            mywf = wf.wireframe(awf[:-1].replace('Half', 'Half ')) 
            mywf.params = toDo[awf]['params'][i]
            
            # Rescale grid points (if needed)
            mywf.gPoints = [i * morePts for i in mywf.gPoints]
            mywf.getPoints()
            npts = mywf.points.shape[0]
            for j in range(npts):
                f1.write(awf + ' ' + times[i] + ' ' + str(j) + ' ' +  '{:.3f}'.format(mywf.points[j][0]) + ' ' + '{:.3f}'.format(mywf.points[j][1]) + ' ' + '{:.3f}'.format(mywf.points[j][2]) +'\n')
    f1.close()
    
# |------------------------------|
# |--- Make velocity map/file ---|
# |------------------------------|
def makeVmap(wombatRes, wfTypes, outName=None, morePts=2):
    """
    Function to calculate the velocity of the wireframe points between two
    time steps. These need not be adjacent time steps, the code will function
    with whatever it is given, but wildly separated times might give interesting
    results. A map of the speeds (shown as colored scatter points) with be shown
    or saved and a text file with the positions and speeds will be created.
    
    Inputs:
        wombatRes: a dictionary filled by info pulled from miniLog in the format
                    wombatRes[instName][wfType] = {}
                    keys - ids, params, times, pickles -> arrays of all the
                           values for each time matching that inst/type
    
        wfTypes:   an array of the wf types to use
        
    Optional Inputs:
        outName:    a name to use for the save files. It will be saved in the wbOutputs 
                    folder as outName + '.txt'. If the name is set to showit then it will
                    pop up the interactive window instead of saving a figure, which gives
                    one more freedom with the viewing angle
                    (defaults to wbVels.txt/png)
    
        morePoints: flag to scale up the number of grid points used in the wireframe. Each
                    dimension of the wf grid is scaled up by this factor. e.g. a wf with
                    standard grid dimensions of [10,10,10] and a setting of 2 would become
                    [20,20,20]
                    (defaults to 2)
    
    Outputs:    Nothing is returned but it generates a text file with the following columns:
                    WFtype
                    time of observation
                    pointID
                    StonyCart x (Rs)
                    StonyCart y (Rs)
                    StonyCart z (Rs)
                    speed (km/s)
    
                It will also save a png showing a 3d scatter plot with the average position
                between the two time steps and each point colored by the speed. If the save
                name is not set this will pop up an interactive window.
    
    """
    # Collect things
    toDo = {}
    if len(wfTypes) > 1:
        sys.exit('Can only make v map for single WF at a time')
    
    # Grab the wf(s) params
    for awf in wfTypes:
        myTimes = []
        myDts   = []
        myParams = []
        for aInst in wombatRes:
            if awf in wombatRes[aInst].keys():
                for i in range(len(wombatRes[str(aInst)][awf]['timesSTR'])):
                    # remove seconds
                    myTime = wombatRes[str(aInst)][awf]['timesSTR'][i][:-3] 
                    if myTime not in myTimes:
                        myTimes.append(myTime)
                        myParams.append(wombatRes[str(aInst)][awf]['paramGrid'][:,i])
                        myDts.append(wombatRes[str(aInst)][awf]['times'][i])

    if len(myTimes) > 2:
        sys.exit('Can only make v map using two times')
        
    # Get time differences
    dt = (myDts[1] - myDts[0]).total_seconds()
    midT =( myDts[0] + datetime.timedelta(seconds=0.5*dt)).strftime("%Y-%m-%d %H:%M:%S")
    
        
    # Make the wireframes
    wf1 = wf.wireframe(awf[:-1].replace('Half', 'Half '))  
    wf1.params = myParams[0] 
    wf1.gPoints = [i * morePts for i in wf1.gPoints]
    wf1.getPoints() 
    wf2 = wf.wireframe(awf[:-1].replace('Half', 'Half '))  
    wf2.params = myParams[1] 
    wf2.gPoints = [i * morePts for i in wf2.gPoints]
    wf2.getPoints() 
    
    # See if showing or saving
    if outName != 'showit':
        if type(outName) == type(None):
            fname = 'wbOutputs/wbVels.txt'
            figName = 'wbOutputs/wbVels'
        else:
            fname = 'wbOutputs/wbVels_'+outName+'.txt'
            figName = 'wbOutputs/wbVels_'+outName
        f1 = open(fname, 'w')
    
    # Caculate mid points and velocity
    npts = wf1.points.shape[0]
    xs, ys, zs, vs = [], [], [], []
    for j in range(npts):
        myx = 0.5 * (wf1.points[j][0] + wf2.points[j][0])
        myy = 0.5 * (wf1.points[j][1] + wf2.points[j][1])
        myz = 0.5 * (wf1.points[j][2] + wf2.points[j][2])
        
        dx = wf2.points[j][0] - wf1.points[j][0]
        dy = wf2.points[j][1] - wf1.points[j][1]
        dz = wf2.points[j][2] - wf1.points[j][2]
        dxyz = np.sqrt(dx**2 + dy**2 +dz**2)
        myv = dxyz * 7e5 / dt # km/s
        
        xs.append(myx)
        ys.append(myy)
        zs.append(myz)
        vs.append(myv)
        
        # Make text to save
        if outName != 'showit':
            f1.write(midT + ' '+ awf +' ' + str(j) + ' ' + '{:.3f}'.format(myx) + ' ' + '{:.3f}'.format(myy)+' '+'{:.3f}'.format(myz) +' '+'{:.3f}'.format(myv) +'\n')
    
    # Plot it
    fig = plt.figure(figsize=(6, 6), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('magma')
    im = ax.scatter(xs, ys, zs, c=vs, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.05, pad=0.02, shrink=0.5) 
    cbar.set_label('Speed (km/s)') 
    # Prettify
    ax.set_aspect('equal') 
    units = ' (Rs)'
    ax.set_xlabel('x'+units)
    ax.set_ylabel('y'+units)
    ax.set_zlabel('z'+units)
    
    
        
    if outName == 'showit':
        plt.show()
    else:
        f1.close()
        print ('Saving points as', fname)
        print ('Saving figure as', figName+picType)
        plt.savefig(figName+picType)

   
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
    
        type -  line profiles: ht#, kin#, en#
                    ht#  - basic fit parameters versus height
                    kin# - basic + derived velocity/acceleration
                    en#  - kin + energetics from mass calc
                    
                    the # modifies which of the basic fit params are shown
                    1 - just height
                    2 - height + aw(s)
                    3 - everything
    
                 list of points: points or pts (same thing, just allow both tags)

                 2 wf velocity: vmap 
                
    
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
            wombatRes[aInst][aWF]['OGids'] = txtIds[myIds]
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
    
    #|--------------------------|
    #|--- Check the mode tag ---|     
    #|--------------------------|
    mode = args[2].lower()
    if mode not in ['ht1', 'kin1', 'en1', 'ht2', 'kin2', 'en2', 'ht3', 'kin3', 'en3', 'linimg', 'logimg', 'sqimg', 'linim', 'logim', 'sqim', 'points', 'pts', 'vmap']:
        print ('Error in reading mode '+mode)
        print ('Pick from [ht#, kin#, en#, linimg, logimg, sqimg, pts, vmap]')
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
    """
    Helper function to process the bonus/non-critical arguments. Default values
    are hardcoded at the top and will be used if args does not include an
    alternate value to replace them
    
    The options in args are
    Direct tags:
    
            eb:         flag to include error bars/uncertainties in figures
                        (defaults to False)

            drag:       flag to fit the drag equation to the reconstructions
                        (defaults to False)

            vsh:        flag to plot versus height instead of versus time
                        (defaults to False)
        
            log:        flag to plot x-axis heights on a log scale
                        (defaults to False)
        
        
            wfcolors:   flag to use the same colors as the wombat GUI instead of the
                        standard plot colors that are more suited for line plots on a
                        white backgrounds
                        (defaults to False)

            png/pdf:    flag to save figures as png or pdf
                        (defaults to png)

            1au/L1:     flag to get a predicted arrival time at either L1 or 1 AU
                        (defaults to doing neither)


    Keys with numbers (# replaced by float or time)
            densratio_#: the ratio between the inner and outer wireframe densities (n1/n2).
                         the densities vary from pixel to pixel but the ratio between the 
                         two remains the same in any overlapping regions. can be a decimal
                         (defaults to 1.)

            dh1_#/dh2_#: the min/max heights to set the range over which the drag calculation
                         looks for an optimal starting height (in Rs)
                         (defaults to 5 Rs/21.5 Rs)

            newbase_#:  a new time to switch to the base time for the mass calculation. the
                        # should be replaced by a time stamp that parse_time can process
        

        Other:
            pickleName - the name of a pickle with the saved results of previous DINGO
                         calculation. These are automatically saved by getEnergetics in
                         wbPlotPickles/ using saveName or defaulting to bigMassRes.pkl
                         It automatically searches this directory so it should just be
                         the file name.

            saveName - if an arugment is passed that does not fit any of the other tag types
                       then it is assumed to be a save name for any output images/files
    
    Outputs:
        The function returns dragHeights, errorbars, incDrag, logIt, massPkl, outName, overlap, 
                             predAT, rebase, reloadIt, versusH, wfColors
    
    """
    
    # Set defaults
    dragHeights = [5,21.5]
    errorbars = False
    incDrag   = False
    logIt     = False
    massPkl   = None
    outName   = None
    overlap   = 1
    predAT    = None
    rebase    = False
    reloadIt  = None
    versusH   = False
    wfColors  = False 
    
    for val in args:
        lval = val.lower()
        # |--- Overlap info for splitting wfs ---|        
        if 'ovl_' in lval:
            try:
                overlap = float(lval.replace('ovl_',''))
            except:
                print ('Cannot convert', lval, 'into overlap value')
        elif 'overlap_' in lval:
            try:
                overlap = float(lval.replace('overlap_',''))
            except:
                print ('Cannot convert', lval, 'into overlap value')
        elif 'densratio_' in lval:
            try:
                overlap = float(lval.replace('densratio_',''))
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
            
        elif lval in ['1au', 'l1']:
            predAT = lval
            
        #|--- Rebase ---|
        elif 'newbase_' in lval:
            rebase  = lval
            
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

    return dragHeights, errorbars, incDrag, logIt, massPkl, outName, overlap, predAT, rebase, reloadIt, versusH, wfColors

# |-------------------------|
# |--- Line profile plot ---|
# |-------------------------|
def profilePlot(mode, wombatRes, wfTypes, logH=False, wfColors=False, enRes=None, kinRes=None, versusH=False, errorbars=False, incDrag=False, outName='wombatProfile'):
    ''' 
    Main plotting script for any form of profile/line plots. It will automatically
    adjust between the different mode options. None of the kinematic/energetic
    calculations are done here, they should be previously computed and passed through
    the appropriate optional inputs.
    
    Inputs:
        mode:      a combination of a string tag + integer tag. the options are
                        ht# - just height (# is parameter config)
                        kin# - height, velocity, accceleration
                        en# - height, vel, acc, mass, kinetic energy

                        # sets which wireframe params are also show
                        1 - just height
                        2 - height + angular width(s)
                        3  - all wf params
    
                    e.g. kin2 would show height, AW, vel, and acc                  
    
        wombatRes: a dictionary filled by info pulled from miniLog in the format
                    wombatRes[instName][wfType] = {}
                    keys - ids, params, times, pickles -> arrays of all the
                           values for each time matching that inst/type
    
        wfTypes:   an array of the wf types to use
        
    Optional Inputs:
        logH:       flag to show the height axis on a log scale
                    (defaults to False)
    
        wfColors:   flag to use the same colors as the wombat GUI instead of the
                    standard plot colors that are more suited for line plots on a
                    white backgrounds
                    (defaults to False)
    
        enRes:      the energetics calculation result. pass directly from getEnergetics
                    but not needed if not doing en mode.
                    (defaults to None)
    
        kinRes:     the kinematics calculation result. pass directly from getKinematics
                    but not needed if not doing kin or en mode.
                    (defaults to None)
    
        versusH:    flag to switch the x-axis to height instead of time. you cannot use
                    this for ht1 and plot only height versus height
                    (defaults to False)
    
        errorbars:  flag to include error bars/uncertainties in figures
                    (defaults to False)
    
        incDrag:    flag that drag fit was included in getKinematics and to include the
                    best fit profiles in the figure
                    (defaults to False)
    
        outName:    a name to use for the save file. It will be saved in the wbOutputs 
                    folder as outName + picType (global var). If the name is set to showit 
                    then it will pop up a window instead of saving a figure
                    (defaults to wombatProfile.png)
        
    Outputs: No direct outputs but saves a figure according to outName
    
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

    #|--------------------------------------------|
    #|--- Set up label structure based on mode ---|     
    #|--------------------------------------------|
    # Figure out what we are plotting and where we want to plot it
    # Labels can go on left or right, once we set this up can just
    # dump in the data where it wants to go
    
    yLabelsL = []
    id2id    = {} # axis number for each param i2i[param] = ax
    yLabelsR = []
    id2idR   = {}
    rAxes    = []
    rColors  = {} 
    lColors   = {}
    lColors[0] = 'k' # everyone has a height
    
    # |--------------|        
    # |--- Case 1 ---|
    # |--------------|        
    # Only height 
    if '1' in mode:
        nParams = 1
        tempWF =  wf.wireframe(wfTypes[0][:-1].replace('Half', 'Half '))
        yLabelsL.append(tempWF.labels[0])
        for wft in wfTypes:
            id2id[wft] = [0]
        
    # |--------------|        
    # |--- Case 2 ---|
    # |--------------|        
    # Only height and ang width
    elif '2' in mode:
        nParams = 1
        # Figure out how many AWs we actually have
        # Differs across diff WF type
        myAWs = []
        for wft in wfTypes:
            tempWF = wf.wireframe(wft[:-1].replace('Half', 'Half '))
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
                
        #|--- Make labels based on number of AW tags ---|       
        nAWs = len(np.unique(myAWs))
        # 1 aw - simple case (test with wbOutputs/201207.txt 95-109 ht2 )
        if nAWs == 1:
            nParams = 2
            yLabelsL.append('AW (deg)')
            lColors[1] = 'k'
            for wft in wfTypes:
                tempWF = wf.wireframe(wft[:-1].replace('Half', 'Half '))
                if 'AW (deg)' in tempWF.labels:
                    idx = np.where(tempWF.labels == 'AW (deg)')[0]
                    id2id[wft][idx[0]] = 1
        
        # 2 aw - only have torus fo/eo but not aw (wbOutputs/WomBlog.txt 95+98+101 ht2)
        if nAWs == 2:
            nParams = 3
            yLabelsL.append('AW_FO (deg)')
            yLabelsL.append('AW_EO (deg)')
            
            for wft in wfTypes:
                tempWF = wf.wireframe(wft[:-1].replace('Half', 'Half '))
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
                tempWF = wf.wireframe(wft[:-1].replace('Half', 'Half '))
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
                
        
    # |--------------|        
    # |--- Case 3 ---|
    # |--------------|        
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
            tempWF = wf.wireframe(wft[:-1].replace('Half', 'Half '))
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
    
    #|--- Add the left labels ---|
    for i in range(nParams):
        if yLabelsL[i] != '': 
            thisLab = yLabelsL[i]
            if thisLab in labSwap.keys():
                thisLab = labSwap[thisLab] 
            ax[i].set_ylabel(thisLab, color=lColors[i])

    #|--- Add the right labels ---|
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
            
            # Check if we need to add a label for wf type 
            # No params in first part for kin1/en3 cases versus height    
            if kinAxId == 0:
                ax[kinAxId].plot(xdata, myVels/1e5, 'o', c=myC, label=awf)
                hasLabel.append(awf)
            else:
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
    
    ax[0].legend(loc='lower right', bbox_to_anchor=(1., 1.), ncols=len(hasLabel))
    
    
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
    
    if outName == 'showit':
        plt.show()
    else:
        print ('Saving figure as', outName+picType)
        plt.savefig('wbOutputs/'+outName+picType)

    

def wombatPlotWrapper(args):
    """
    Main wrapper to call from the command line. The script will use DINGO
    to calculate masses as needed, which can take some time. It does include
    the option to save the DINGO results in a pkl so they can be reloaded
    after the first calcuation to allow for quick manipulation of plot 
    aesthetics.


    The command line syntax is

        python3 wombatPlots.py logFile lineIDs mode [optional parameters]

    where 

        logFile:  points to a log file created by WOMBAT

        lineIDs:  can be a single integer (e.g. 4)
                  a range of integers (e.g. 4-10)
                  or a series of integers connected by + (e.g. 4+8+12)

        mode:    select from one of the following plot mode tags

                 Line plot variants:
                    ht# - just height (# is parameter config)
                    kin# - height, velocity, accceleration
                    en# - height, vel, acc, mass, kinetic energy

                    # sets which wireframe params are also show
                    1 - just height
                    2 - height + angular width(s)
                    3  - all wf params

                    e.g. kin2 would show height, AW, vel, and acc
            
                 pts - print the Cartesian Stonyhust location all of the wf 
                       points to a file

                 vmap - determine the velocity from two timesteps for the same wf
                        (using their diff heights and delta t). Makes a 3d scatter
                        plot with each wf point colored by its velocity and prints
                        the cartesian locations and total velocity to a file

    and the optional arguments fall into the following types
    Direct flags (written as is):
            eb:         flag to include error bars/uncertainties in figures
                        (defaults to False)

            drag:       flag to fit the drag equation to the reconstructions
                        (defaults to False)

            vsh:        flag to plot versus height instead of versus time
                        (defaults to False)
        
            log:        flag to plot x-axis heights on a log scale
                        (defaults to False)
        
        
            wfcolors:   flag to use the same colors as the wombat GUI instead of the
                        standard plot colors that are more suited for line plots on a
                        white backgrounds
                        (defaults to False)

            png/pdf:    flag to save figures as png or pdf
                        (defaults to png)

            1au/L1:     flag to get a predicted arrival time at either L1 or 1 AU
                        (defaults to doing neither)


    Keys with numbers (# replaced by float or time)
            densratio_#: the ratio between the inner and outer wireframe densities (n1/n2).
                         the densities vary from pixel to pixel but the ratio between the 
                         two remains the same in any overlapping regions. can be a decimal
                         (defaults to 1.)

            dh1_#/dh2_#: the min/max heights to set the range over which the drag calculation
                         looks for an optimal starting height (in Rs)
                         (defaults to 5 Rs/21.5 Rs)

            newbase_#:  a new time to switch to the base time for the mass calculation. the
                        # should be replaced by a time stamp that parse_time can process
        

        Other:
            pickleName - the name of a pickle with the saved results of previous DINGO
                         calculation. These are automatically saved by getEnergetics in
                         wbPlotPickles/ using saveName or defaulting to bigMassRes.pkl
                         It automatically searches this directory so it should just be
                         the file name.

            saveName - if an arugment is passed that does not fit any of the other tag types
                       then it is assumed to be a save name for any output images/files
    """
    
    #|----------------------------------|
    #|--- Check the number of inputs ---|     
    #|----------------------------------|
    nArgs = len(args)
    if (nArgs <3):
        print ('Incorrect number of parameters provided. Syntax is')
        for astr in errorStrings:
            print (astr)
        sys.exit()
        
    #|-------------------------------|
    #|--- Check for output folder ---|     
    #|-------------------------------|
    if not os.path.exists('wbOutputs/'):
        os.mkdir('wbOutputs/')
        print ('Created output folder wbOutputs')
        
    #|-------------------------------------|
    #|--- Check the critical parameters ---|     
    #|-------------------------------------|
    miniLog, wombatRes, mode, uniqTs, wfTypes, allInsts = processArgs(args)
    
    #|----------------------------------|
    #|--- Check any bonus parameters ---|     
    #|----------------------------------|
    # Set defaults
    allBonus = args[3:]
    dragHeights, errorbars, incDrag, logIt, massPkl, outName, overlap, predAT, rebase, reloadIt, versusH, wfColors  = processBonusArgs(allBonus, mode)
    
    #|--------------------------|
    #|--- Process kinematics ---|     
    #|--------------------------|
    kinRes = None
    if ('kin' in mode) or ('en' in mode):
        kinRes = getKinematics(wombatRes, wfTypes, incDrag=incDrag, predAT=predAT)

    #|--------------------------|
    #|--- Process energetics ---|     
    #|--------------------------| 
    enRes = None   
    if ('en' in mode):
        enRes = getEnergetics(args, wombatRes, wfTypes, kinRes, reloadIt=massPkl, overlap=overlap, rebase=rebase)
        
    #|--------------------------|
    #|--- Run line plot mode ---|     
    #|--------------------------|
    if mode in ['ht1', 'ht2','ht3', 'kin1', 'kin2', 'kin3', 'en1', 'en2', 'en3']:
        if type(outName) == type(None):
            outName = 'showit'
        profilePlot(mode, wombatRes, wfTypes, logH=logIt, wfColors=wfColors, kinRes=kinRes, enRes=enRes, errorbars=errorbars, versusH=versusH, incDrag=incDrag, outName=outName)
    
    #|--------------------------|
    #|--- Basic print points ---|     
    #|--------------------------|    
    if mode in ['pts', 'points']:
        printPoints(wombatRes, wfTypes, outName=outName)
    
    #|--------------------|
    #|--- Velocity map ---|     
    #|--------------------|    
    if mode in ['vmap']:
        makeVmap(wombatRes, wfTypes, outName=outName)
    

# |-----------------------|
# |--- Text line input ---|
# |-----------------------|
if __name__ == '__main__':
    wombatPlotWrapper(sys.argv[1:])
