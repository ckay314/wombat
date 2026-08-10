import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
import pickle
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

sys.path.append('wombatCode/') 
from wombatLoadCTs import *
import wombatPlots as wp

# Python needs ffmpeg installed via brew, pip no happy for some reason


#|-------------------------|
#|--- Configure here!!! ---|
#|-------------------------|
# Name of wombat log file
logFilePath = 'wbOutputs/2303full_CME1b.txt'
# Lines to do
lines = range(41) # Replace with string reading code from other files
# Time Resolution (in minutes)
tRes = 30 
# Plot shape
nHoriz = 4
# Include clean imgs without wf proj
doClean = True
# Running (0) or base diff (1)
didx = 0
# Scaling mode linear(0), log(1), or sqrt(2)
sclidx = 0

# Calc num of rows bases on nHoriz and nInsts
nVert = 1 # need to calc this later
if doClean:
    nVert *= 2
    
# Dictionary for min/max values based on inst/scale mode
aiamm = [[0,0,0], [191, 191, 191]]
euimm = [[63,67,32], [150, 230, 191]]
cormm = [[63,0,21], [191, 191, 191]]
c2mm  = [[0,0,21], [191,191,191]]
c3mm  = [[37,0,37], [191,191,191]]
himm  = [[63,0,21], [128,191,191]]
wilomm = [[0,0,21], [128,191,191]]

MMdict = {'AIA94':aiamm, 'AIA131':aiamm, 'AIA171':aiamm,'AIA193':aiamm,'AIA211':aiamm,'AIA304':aiamm,'AIA335':aiamm,'AIA1600':aiamm,'AIA1700':aiamm, 'C2':c2mm, 'C3':c3mm, 'COR1':cormm, 'COR2':cormm, 'COR1A':cormm, 'COR2A':cormm, 'COR1B':cormm, 'COR2B':cormm, 'EUI174':aiamm, 'EUI304':aiamm, 'EUVI171':aiamm, 'EUVI195':aiamm, 'EUVI284':aiamm, 'EUVI304':aiamm, 'EUVI171A':aiamm, 'EUVI195A':aiamm, 'EUVI284A':aiamm, 'EUVI304A':aiamm, 'EUVI171B':aiamm, 'EUVI195B':aiamm, 'EUVI284B':aiamm, 'EUVI304B':aiamm, 'HI1':himm, 'HI2':himm, 'HI1A':himm, 'HI2A':himm, 'HI1B':himm, 'HI2B':himm, 'HI1A_SR':himm, 'HI1B_SR':himm, 'HI2A_SR':himm, 'HI2B_SR':himm, 'SOLOHI':wilomm, 'SOLOHI1':wilomm, 'SOLOHI2':wilomm, 'SOLOHI3':wilomm, 'SOLOHI4':wilomm, 'WISPR':wilomm, 'WISPRI':wilomm, 'WISPRO':wilomm, 'WISPR_LW':wilomm, 'WISPRI_LW':wilomm, 'WISPRO_LW':wilomm, 'WISPR_L3':wilomm, 'WISPRI_L3':wilomm, 'WISPRO_L3':wilomm}

 
    
#|------------------------------|
#|--- Read in fits, organize ---|
#|------------------------------|
logFile = np.genfromtxt(logFilePath, dtype=str)
allInsts = np.unique(logFile[lines,1])
nInsts = len(allInsts)
allPickles = np.unique(logFile[lines,13])
if len(allPickles) != 1:
    sys.exit('Cannot combine multiple pickles (yet)')
nLines = len(logFile[:,0])
theWFs = np.unique(logFile[lines,3])
nWFs = len(theWFs)

# Package by inst, figure out min/max time
timesByInst  = {}
pidx2params   = {}
minTime = datetime.datetime(3000,1,1)
maxTime = datetime.datetime(1000,1,1)
for i in range(nLines):
    myInst = logFile[i,1]
    myTime = datetime.datetime.strptime(logFile[i,2], "%Y-%m-%dT%H:%M:%S" )
    myWFtype = logFile[i,3]
    myParams = logFile[i,4:13]
    myParams = myParams[myParams != 'None'].astype(float)
    mypidx   = logFile[i,14]
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


sclFig = {1:5, 2:4, 3:3.5, 4:2.5, 5:2, 6:2 }
myFigScl = sclFig[nHoriz]
    
fig, ax0 = plt.subplots(nVert, nHoriz, figsize = (nHoriz*myFigScl,nVert*myFigScl))
axes = ax0.flatten()
fig.set_facecolor('k')
for ax in axes: 
    ax.set_axis_off()
    ax.set_aspect('equal')

movt = 0
nPanels = nInsts
if doClean: nPanels *= 2
imObjs = []
imObjsC = []
textObjs = []
scatObjs = {}
for i in range(nInsts):
    myInst = allInsts[i]
    instIdx = movie2inst[myInst][movt]
    mm = instMinMax[myInst]
    mySclIm = sclIms[myInst][didx][instIdx][sclidx]
    
    imObj = axes[i].imshow(mySclIm, cmap=instCMaps[myInst], vmin=mm[0], vmax=mm[1], origin='lower')
    imObjs.append(imObj)
    mydate =  proIms[myInst][0][instIdx].date.datetime.strftime("%Y-%m-%dT%H:%M")
    panelLabel = myInst + ' ' + mydate

    if doClean:
        imObjC = axes[i+nHoriz].imshow(mySclIm, cmap=instCMaps[myInst], vmin=mm[0], vmax=mm[1], origin='lower')
        imObjsC.append(imObjC)
        textObj = axes[i+nHoriz].text(0.5, 0, panelLabel, c='w', bbox=dict(facecolor='black', alpha=0.5), horizontalalignment='center',verticalalignment='bottom', transform = axes[i+nHoriz].transAxes)
    else:
        textObj = axes[i+nHoriz].text(0.5, 0, panelLabel, c='w', bbox=dict(facecolor='black', alpha=0.5), horizontalalignment='center',verticalalignment='bottom', transform = axes[i+nHoriz].transAxes)
    textObjs.append(textObj)

plt.subplots_adjust(wspace=0.0, hspace=0.0,left=0,right=1,bottom=0,top=1)
#plt.show()

def update(movt):
    for i in range(nInsts):
        myInst = allInsts[i]
        instIdx = movie2inst[myInst][movt]
        #instPidx = pidxByInst[myInst][instIdx]
        #instParams = paramsByInst[myInst][instIdx]
    
        didx = 0
        sclidx = 0
        mySclIm = sclIms[myInst][didx][instIdx][sclidx]
        mydate =  proIms[myInst][0][instIdx].date.datetime.strftime("%Y-%m-%dT%H:%M")
        panelLabel = myInst + ' ' + mydate
        
        imObjs[i].set_data(mySclIm)
        textObjs[i].set_text(panelLabel)
        if doClean:
            imObjsC[i].set_data(mySclIm)
    
#ani = animation.FuncAnimation(fig=fig, func=update, frames=nTimes+2, interval=150)
#plt.show()
#ani.save('test.mp4', writer='ffmpeg')


'''fig, ax = plt.subplots()
t = np.linspace(0, 3, 40)
g = -9.81
v0 = 12
z = g * t**2 / 2 + v0 * t

v02 = 5
z2 = g * t**2 / 2 + v02 * t

scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
ax.set(xlim=(0, 3), ylim=(-4, 10), xlabel='Time [s]', ylabel='Z [m]')
ax.legend()


def update(frame):
    # for each frame, update the data stored on each artist.
    x = t[:frame]
    y = z[:frame]
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # update the line plot:
    line2.set_xdata(t[:frame])
    line2.set_ydata(z2[:frame])
    return (scat, line2)


ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
#plt.show()
#ani.save(filename="pillow_example.gif", writer="pillow")
ani.save('test.mp4', writer='ffmpeg')'''
