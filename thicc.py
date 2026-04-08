import pickle
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.coordinates
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
from sunpy.coordinates import frames
from sunpy.coordinates import get_horizons_coord
import matplotlib.gridspec as gridspec
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator
import matplotlib.cm as cm

import sys, os
sys.path.append('prepCode/') 
sys.path.append('wombatCode/') 
import wombatWF as wf
from wombatLoadCTs import *

# |--------------------------------|
# |------- Suppress Warnings ------|
# |--------------------------------|
# Astropy/sunpy have a lot of thoughts about missing
# keywords so set the logger to only complain about
# actual errors not all warnings
import logging
logging.basicConfig(level='INFO')
slogger = logging.getLogger('sunpy')
slogger.setLevel(logging.ERROR)
alogger = logging.getLogger('astropy')
alogger.setLevel(logging.ERROR)
# Turn off divide warning globally
np.seterr(divide='ignore')

global picType
picType = '.png' # either '.png' or '.pdf'

def createGrid(FoV, nGridY):
    miny, maxy = FoV[0][0], FoV[0][1]
    minz, maxz = FoV[1][0], FoV[1][1]
    dy = (maxy - miny) / nGridY
    # Points on the grid
    ygs = np.array([miny + i*dy for i in range(nGridY+1)])
    # Midpoints
    yms = np.array([miny + (0.5+i)*dy for i in range(nGridY)])
    # Same for z, but need to get nGridZ
    nGridZ = int((maxz - minz) / dy)+1
    zgs = np.array([minz + i * dy  for i in range(nGridZ+1)])
    zms = np.array([minz + (0.5+i)*dy for i in range(nGridZ)])
    return dy, ygs, yms, nGridZ, zgs, zms

def getWidth(points, FoV=None, nGridY=100, fillAround=True):
    # Points are xyz where: 
    #   x = LoS
    #   y = Lon (perp to LoS)
    #   z = Lat
    # FoV should be [[miny, maxy], [minz, maxz]]
    #   in same units as points (e.g. Rsun)
    # nGridY is the number of grid cells in the
    #   y direction. This determines a physical
    #   grid cell size that will be the same in 
    #   the other dimensions
    # fillAround will smooth out the edges of the center
    #   position array (midx) which is needed if passing
    #   to an interpolation function after
    
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    
    # Get range of points in yz
    minyP, maxyP = np.min(y), np.max(y)
    minzP, maxzP = np.min(z), np.max(z)
    
    # Set up FoV (if we don't have)
    if type(FoV) == type(None):
        padY = 0.1*(maxyP - minyP)
        padZ = 0.1*(maxyP - minyP)
        FoV = [[minyP-padY, maxyP+padY], [minzP-padZ, maxzP+padZ]]
    miny, maxy = FoV[0][0], FoV[0][1]
    minz, maxz = FoV[1][0], FoV[1][1]
    
    # Set up a grid for the width contours
    dy, ygs, yms, nGridZ, zgs, zms = createGrid(FoV, nGridY)
        
    wids = np.zeros([nGridZ, nGridY])
    midx = np.zeros([nGridZ, nGridY]) - 9999
    
    pad = 1.4 # distance from midpoint to check (in dy)
    
    for i in range(nGridY):       
        if (yms[i] >= minyP) & (yms[i] <= maxyP):
            mypts = np.where(np.abs(y-yms[i]) <= pad*dy)[0]
            subx, suby, subz = x[mypts], y[mypts], z[mypts]
            if len(subz) > 0:
                myminz, mymaxz = np.min(subz), np.max(subz)
                if zms[0] > myminz:
                    zidx0 = 0
                else:
                    zidx0 = np.max(np.where(zms <= myminz))
                if zms[-1] < mymaxz:
                    zidx1 = len(zms) - 1
                else:
                    zidx1 = np.min(np.where(zms >= mymaxz))
                zinds = range(zidx0, zidx1)
        
                subx, suby, subz = x[mypts], y[mypts], z[mypts]
        
                for j in zinds:
                    mypts2 = np.where(np.abs(subz - zms[j]) < pad*dy)[0]
                    #mypts2 = np.where((subz >= z0-pad*dy) & (subz <= z1+pad*dy))[0]
                    sortx = np.sort(subx[mypts2])
                    if len(sortx) > 1:               
                        fullwid = sortx[-1] - sortx[0]
                        wids[j,i] = fullwid
                        midx[j,i] = 0.5*(sortx[-1] + sortx[0])
    # Certain shapes (e.g. half sphere) can see "into" and makes wid/xc wonky
    # when only have one side along LoS... just remove this part
    idx = np.where(midx != -9999)                    
    medxc, stdxc = np.median(midx[idx]), np.std(midx[idx])
    medwid, stdwid = np.median(wids[idx]), np.std(wids[idx])
    # Check for back only section
    midx[np.where((midx < (medxc-stdxc)) & (wids < 10*dy ) & (wids != -9999))] = -9999
    # Check for front only section
    midx[np.where((midx > (medxc+stdxc)) & (wids < 10*dy ) & (wids != -9999))] = -9999

                  
    # need to fill in the outer -9999 region of xc so interp is happy
    for i in range(midx.shape[1]):
        notOut = np.where(midx[:,i] !=-9999)[0]
        if len(notOut) >= 2:
            midx[:notOut[0], i]  = midx[notOut[0],i]
            midx[notOut[-1]:, i] = midx[notOut[-1],i]
    for i in range(midx.shape[0]):
        notOut = np.where(midx[i,:] !=-9999)[0]
        if len(notOut) >= 2:
            midx[i,:notOut[0]]  = midx[i, notOut[0]]
            midx[i,notOut[-1]:] = midx[i, notOut[-1]]
    
       
    return wids, midx, FoV, nGridY


def StonyCart2CartFoV(pts, satLat, satLon, roll):
    # assume is pts is [xs, ys, zs]
    # and first point is spacecraft
    # and second point is FoV center
    # satLat, satLon, roll all in degrees
    
    
    # Rotate so sc at x=0
    pts = wf.rotz(pts,-satLon)
    
    # Rotate so sc at z = 0
    pts = wf.roty(pts, satLat)
    
    # Move spacecraft to origin
    scx = pts[0][0]
    pts[0] -= scx
    
    # Want FoV in -x dir
    ang = np.arctan2(pts[1][1], pts[0][1]) + np.pi
    pts = wf.rotz(pts,-ang*180/np.pi)
    
    # Take out z component of FoV
    ang = np.arctan2(pts[2][1], np.abs(pts[0][1]))
    pts = wf.roty(pts,-ang*180/np.pi)
        
    # Account for roll
    #roll = myMap.meta['crota']
    pts = wf.rotx(pts, -roll)
    
    # Move FoV to x = 0
    FoVx =  pts[0][1]
    pts[0] -= FoVx 
    
    # Take off the first two points we added
    xOut = pts[0][2:]
    yOut = pts[1][2:]
    zOut = pts[2][2:]
    
    return [xOut, yOut, zOut]
    
    
def map2CartFoV(myMap, points, pixCent=None):
    # points should be [[pixss], [pixys]]
    # pix cent should be [pixx, pixy]
    
    # Get satellite position from the map
    satLonD = myMap.observer_coordinate.lon.degree
    satLatD = myMap.observer_coordinate.lat.degree
    satR = myMap.observer_coordinate.radius.au 
    satLonR = satLonD * np.pi / 180.
    satLatR = satLatD * np.pi / 180.
    
    
    # Get vector from sun to sat
    # (just the cartesian sat loc)
    satxyz = np.array([np.cos(satLatR)*np.cos(satLonR), np.cos(satLatR)*np.sin(satLonR), np.sin(satLatR)]) * satR
    
    # Get the direction the FoV is pointing in Stony frame
    # Use pix cent if provided, otherwise assume middle
    if type(pixCent) == type(None):
        pixCent = [myMap.data.shape[1]/2, myMap.data.shape[0]/2] # pix is xy but shape is yx

    # Get the heliprojective coord of the pixel
    coordM = myMap.pixel_to_world(pixCent[0] * u.pix, pixCent[1] * u.pix)
    # Get elongation angle -> distance to Thompson Sphere
    ell = np.sqrt(coordM.Tx.rad**2 + coordM.Ty.rad**2)
    dM = np.abs(satR * np.abs(np.cos(ell)))
    # Make skycoord with HPC and distance (transform needs dist if off limb)
    hpc = SkyCoord(Tx=coordM.Tx, Ty=coordM.Ty, distance=dM*u.au, frame= coordM.frame)
    # Convert to Stonyhurst, make a Cartesian array
    ston = hpc.transform_to(frames.HeliographicStonyhurst)
    TSxyz = np.array([ston.cartesian.x.to_value(), ston.cartesian.y.to_value(), ston.cartesian.z.to_value()])
    # Vectors we will need
    LoS = TSxyz - satxyz 
    usatxyz = satxyz / np.linalg.norm(satxyz)
    uTSxyz = TSxyz / np.linalg.norm(TSxyz)
    uLoS = LoS / np.linalg.norm(LoS)
    
    # Start coord arrays with sat loc and FoV cent
    xs = [satxyz[0]*215, TSxyz[0]*215]
    ys = [satxyz[1]*215, TSxyz[1]*215]
    zs = [satxyz[2]*215, TSxyz[2]*215]
    
    # Get the OG coords for each pixels in points
    nPoints = len(points[0])
    #cs = []
    for i in range(nPoints):
        coord0 = myMap.pixel_to_world(points[0][i] * u.pix, points[1][i] * u.pix)
        ell = np.sqrt(coord0.Tx.rad**2 + coord0.Ty.rad**2)
        d0 = np.abs(satR * np.cos(ell))
        hpc0 = SkyCoord(Tx=coord0.Tx, Ty=coord0.Ty, distance=d0*u.au, frame= coord0.frame)
        ston0 = hpc0.transform_to(frames.HeliographicStonyhurst)
        TSxyz0 = np.array([ston0.cartesian.x.to_value(), ston0.cartesian.y.to_value(), ston0.cartesian.z.to_value()])
        LoS0 = TSxyz0 - satxyz    
        uTSxyz0 = TSxyz0 / np.linalg.norm(TSxyz)
        uLoS0 = LoS0 / np.linalg.norm(LoS0)
        dotIt = np.dot(uLoS0, uLoS)
        if dotIt > 1: dotIt = 1.
        elif dotIt < -1: dotIt = -1.
        ang = np.arccos(dotIt)
        newL = dM / np.cos(np.abs(ang))
        TSxyz0 = satxyz + newL*uLoS0
        xs.append(TSxyz0[0]*215)
        ys.append(TSxyz0[1]*215)
        zs.append(TSxyz0[2]*215)
        #cs.append(myMap.data[points[1][i], points[0][i]])
    
    # Convert everyone at the same time
    pts = np.array([xs, ys, zs])      
    
    '''fig = plt.figure(figsize=(8, 5), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    # WF1 scatter
    im = ax.scatter(pts[0][2:], pts[1][2:],pts[2][2:], c=cs, cmap='Greys_r')
    ax.scatter(0, 0, 0, c='y', s=100)
    ax.scatter(pts[0][0], pts[1][0],pts[2][0], c='b')
    
    awf = wf.wireframe('GCS')
    awf.params = [45.2, 147.4, 26.1, 20.5, 59.4, 0.3]
    awf.getPoints()
    pts = np.transpose(awf.points)
    ax.scatter(pts[0], pts[1],pts[2], c='g')
    plt.show()'''
    
    if 'crota' in myMap.meta:
        rollIt = myMap.meta['crota']
    elif 'sc_roll' in myMap.meta:
        rollIt = myMap.meta['sc_roll']
    else:
        print ('Neither crota or sc_roll in map metadata. Assuming zero roll')
        rollIt = 0

    res = StonyCart2CartFoV(pts, satLatD, satLonD, rollIt)

    return res


def wf2CartFoV(myMap, aWF, pixCent=None):
    # points should be [[pixss], [pixys]]
    # pix cent should be [pixx, pixy]
    
    # Get satellite position from the map
    satLonD = myMap.observer_coordinate.lon.degree
    satLatD = myMap.observer_coordinate.lat.degree
    satR = myMap.observer_coordinate.radius.au 
    satLonR = satLonD * np.pi / 180.
    satLatR = satLatD * np.pi / 180.
    
    
    # Get vector from sun to sat
    # (just the cartesian sat loc)
    satxyz = np.array([np.cos(satLatR)*np.cos(satLonR), np.cos(satLatR)*np.sin(satLonR), np.sin(satLatR)]) * satR
    
    # Get the direction the FoV is pointing in Stony frame
    # Use pix cent if provided, otherwise assume middle
    if type(pixCent) == type(None):
        pixCent = [myMap.data.shape[1]/2, myMap.data.shape[0]/2] # pix is xy but shape is yx

    # Get the heliprojective coord of the pixel
    coordM = myMap.pixel_to_world(pixCent[0] * u.pix, pixCent[1] * u.pix)
    # Get elongation angle -> distance to Thompson Sphere
    ell = np.sqrt(coordM.Tx.rad**2 + coordM.Ty.rad**2)
    dM = np.abs(satR * np.abs(np.cos(ell)))
    # Make skycoord with HPC and distance (transform needs dist if off limb)
    hpc = SkyCoord(Tx=coordM.Tx, Ty=coordM.Ty, distance=dM*u.au, frame= coordM.frame)
    # Convert to Stonyhurst, make a Cartesian array
    ston = hpc.transform_to(frames.HeliographicStonyhurst)
    TSxyz = np.array([ston.cartesian.x.to_value(), ston.cartesian.y.to_value(), ston.cartesian.z.to_value()])
    # Vectors we will need
    LoS = TSxyz - satxyz 
    usatxyz = satxyz / np.linalg.norm(satxyz)
    uTSxyz = TSxyz / np.linalg.norm(TSxyz)
    uLoS = LoS / np.linalg.norm(LoS)
    
    # Start coord arrays with sat loc and FoV cent
    xs = [satxyz[0]*215, TSxyz[0]*215]
    ys = [satxyz[1]*215, TSxyz[1]*215]
    zs = [satxyz[2]*215, TSxyz[2]*215]
    
    wfPoints = aWF.points
    xwf = wfPoints[:,0]
    ywf = wfPoints[:,1]
    zwf = wfPoints[:,2]
    
    x = [*xs, *xwf]
    y = [*ys, *ywf]
    z = [*zs, *zwf]
    
    pts = np.array([x, y, z])      
    
    if 'crota' in myMap.meta:
        rollIt = myMap.meta['crota']
    elif 'sc_roll' in myMap.meta:
        rollIt = myMap.meta['sc_roll']
    else:
        print ('Neither crota or sc_roll in map metadata. Assuming zero roll')
        rollIt = 0
        
    res = StonyCart2CartFoV(pts, satLatD, satLonD, rollIt)
    
    return res
    

def mass2dens(myMap, satDict, awf, massMap, doInner=False, densRatio=1, downSelect=8):
    '''
    Fuction to take a mass image map, satellite dictionay, and wireframe object and
    determine the width perp to the plane of sky and convert integrated mass to density.
    A single wireframe or two can be passed (to represent a shock and ejecta). A constant
    density is assumed along the line of sight within each WF. For GCS/torus shapes the
    doInner flag indicates to remove the inner gap between the legs
    
    Inputs:
        myMap:      a astropy/sunpy map where the data has units of g per pix
    
        satDict:    a wombat style satellite dictionary
    
        awf:        a wombat wireframe or two in an array [wf1, wf2] where wf1 is the 
                    primary wf (i.e. ejecta) and wf2 is an outer/surrounding shape (sheath)
    
    Optional Inputs:
        doInner:    remove the gap region at the back for a GCS or torus wf
                    (defaults to false)
    
        densRatio:  the ratio of the density between wf1 and wf2, which is treated as a 
                    constant value over all lines of sight. calculated as wf1/wf2 and 
                    ignored if only a single wf is passed
                    (defaults to 1)
    
        downSelect: an integer to downsample the output resolution
                    (defaults to 8)
    
    Outputs:
        widMap:     an array of the widths (in Rs) perp to the plane of sky 
                    the array contains [wf1, wf1_inner, wf2] where inner represents
                    the inner gap in wf1. wf1_inner and wf2 will be None if not used.
                    each element is an array matching the region shown in subMass and
                    with the field of view given in outFoV
    
        xcMap:      the distance of the center of mass from the plane of sky in the same
                    format as widMap. also in Rs
    
        densMap:    same format as widMap/xcMap but for the calculated density. In g/cm^3
    
        subMass:    an array containing the data from the original mass map but clipped to 
                    the field of view around the projected wf(s)
    
        outFoV:     an array with [x0, xf, y0, yf, downselect] where the first four 
                    elements represent the extent of the sub-field of view (in pix) and
                    downselect is an integer representing the 1D reduction in resolution.
                    The field of values represent the locations in the full resolution image
                    (e.g. a 1024x1024 original image with outFoV [0,512,0,512,2] would be a
                    256x256 image containing one quadrant of the original )
    
        pix2FOV:    an array with three interpolation functions ([FOV2x, FOV2y, FOV2z] ) 
                    that convert from from a pixel in the original image ((pixx, pixy)) to the
                    location in the 3D cartesian system with the image in yz-plane at x=0 with
                    the image center at the origin
                    the interp funcs do need to be called with the pix in parens 
                    (e.g. FoVx((pixx,pixy)) )
    
    ''' 
    
    # Check if have single wireframe or multiple
    if (type(awf) == type(wf.wireframe(None))):
        multiMode = False 
    else:
        if len(awf) > 2:
            sys.exit('Thickness calc only capable of doing two wireframes')
        else:
            multiMode = True
            awf2 = awf[1]
            awf  = awf[0]

    # save downselect input as the final downselect
    # (will use same var name for interp downselects in middle)
    downSelectF = downSelect

    # |------------------------------------------|
    # |--- Map pixels to FoV Cartestian frame ---|
    # |------------------------------------------|
    # FoV Cart = the FoV in the yz plane at x = 0 
    # with the center pixel at the origin
    downSize = 32
    pixx, pixy = [], []
    mvals = []
    for j in range(myMap.data.shape[0])[::downSize]:
        for i in range(myMap.data.shape[1])[::downSize]:
            pixx.append(i)
            pixy.append(j)


    #ptsOut = map2CartFoV(myMap, [[0, myMap.data.shape[1]], [myMap.data.shape[0]/2, myMap.data.shape[0]/2]])
    ptsOut = map2CartFoV(myMap, [pixx, pixy])   # is [x,y,z] where each is same len as pixIn  
        
        
    uniX = np.unique(np.array(pixx))
    uniY = np.unique(np.array(pixy))
    maxX = np.max(uniX)
    maxY = np.max(uniY)
    
    FOVx = ptsOut[0].reshape([len(uniY), len(uniX)])
    FOVy = ptsOut[1].reshape([len(uniY), len(uniX)])
    FOVz = ptsOut[2].reshape([len(uniY), len(uniX)])  
        
    # Set interpolators to take (x, y) as input order but python array is [y,x]
    FOV2x = RegularGridInterpolator((uniX, uniY), np.transpose(FOVx), method='linear')
    FOV2y = RegularGridInterpolator((uniX, uniY), np.transpose(FOVy), method='linear')
    FOV2z = RegularGridInterpolator((uniX, uniY), np.transpose(FOVz), method='linear')

    
    #|-------------------------------------------------|
    #|--- Get the wireframe width perp to FoV plane ---|
    #|-------------------------------------------------|
    # |--- Do the outer WF first ---|
    # Need to set FoV to larger region
    if multiMode:
        awf2.gPoints = [i * 10 for i in awf2.gPoints]
        awf2.getPoints()
        wfPts2 = wf2CartFoV(myMap, awf2)
        wfPts2T = np.transpose(np.array(wfPts2))
        
        wids2, midx2, FoV, nGridY = getWidth(wfPts2T)
        dy, ygs, yms, nGridZ, zgs, zms = createGrid(FoV, nGridY)
        wid_smooth2 = ndimage.gaussian_filter(wids2, sigma=2.0, order=0)
        
        # indexing of func is y,z    
        widFunc2 = RegularGridInterpolator((yms, zms), np.transpose(wid_smooth2), method='linear', bounds_error=False, fill_value=0)
        xcFunc2  = RegularGridInterpolator((yms, zms), np.transpose(midx2), method='linear', bounds_error=False, fill_value=0)
        
    # |--- Do the inner/main WF ---|
    awf.gPoints = [i * 10 for i in awf.gPoints]
    awf.getPoints()
    wfPts = wf2CartFoV(myMap, awf)
    wfPtsT = np.transpose(np.array(wfPts))
    # Check if we have an existing FoV
    if multiMode:
        wids, midx, FoV, nGridY = getWidth(wfPtsT, FoV=FoV, nGridY=nGridY)
    # Otherwise grab the new one
    else:
        wids, midx, FoV, nGridY = getWidth(wfPtsT)
        dy, ygs, yms, nGridZ, zgs, zms = createGrid(FoV, nGridY)
    wid_smooth = ndimage.gaussian_filter(wids, sigma=2.0, order=0)

    # indexing of func is y,z    
    widFunc = RegularGridInterpolator((yms, zms), np.transpose(wid_smooth), method='linear', bounds_error=False, fill_value=0)  
    xcFunc  = RegularGridInterpolator((yms, zms), np.transpose(midx), method='linear', bounds_error=False, fill_value=0)
    
    # |--- Repeat process for the inside (if doing) ---|
    if doInner and (awf.WFtype in ['GCS', 'Torus']):
        awf.getPoints(inside=True)
        wfPtsI = wf2CartFoV(myMap, awf)
        wfPtsI = np.transpose(np.array(wfPtsI))
        widsI, midxI, FoV, nGridY = getWidth(wfPtsI, FoV=FoV, nGridY=nGridY)
        wid_smoothI = ndimage.gaussian_filter(widsI, sigma=2.0, order=0)

        # indexing of func is y,z    
        widFuncI = RegularGridInterpolator((yms, zms), np.transpose(wid_smoothI), method='linear', bounds_error=False, fill_value=0) 
        xcFuncI  = RegularGridInterpolator((yms, zms), np.transpose(midxI), method='linear', bounds_error=False, fill_value=0)
        
    '''
    # Plot of widths    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = plt.imshow(wids, origin='lower', extent=[FoV[0][0], FoV[0][1], FoV[1][0], FoV[1][1]])
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.05, pad=0.02, shrink=0.5) 
    ax.set_xlabel('Proj dist (R$_S$)')
    ax.set_ylabel('Proj dist (R$_S$)')
    cbar.set_label('Width (R$_S$)')
    plt.show()
    print (sd)'''
    
    #|------------------------------------|
    #|--- Get width over full FoV Grid ---|
    #|------------------------------------|
    # Start with grid of pixels
    downSize = 16
    fake_px = np.array(range(myMap.data.shape[1])[::downSize])
    fake_py = np.array(range(myMap.data.shape[0])[::downSize])
    fake_px = np.array(range(myMap.data.shape[1])[::downSize])
    fake_py = np.array(range(myMap.data.shape[0])[::downSize])
    
    fake_px = fake_px[np.where(fake_px <= maxX)[0]]
    fake_py = fake_py[np.where(fake_py <= maxY)[0]]
    
    # Convert to grid of fov cart
    fpxx, fpyy = np.meshgrid(fake_px, fake_py)
    f_fovx = FOV2x((fpxx, fpyy))
    f_fovy = FOV2y((fpxx, fpyy))
    f_fovz = FOV2z((fpxx, fpyy))
    
    # Use width interpolater
    widsInt = widFunc((f_fovy, f_fovz))
    # Same thing for inside, diff func, same grid
    if doInner:
        widsIntI = widFuncI((f_fovy, f_fovz))
    if multiMode:
        widsInt2  = widFunc2((f_fovy, f_fovz))
    
    #|----------------------------------------|
    #|--- Find bounding box where wid != 0 ---|
    #|----------------------------------------|
    if multiMode:
        notZero = np.where((widsInt != 0) | (widsInt2 != 0))
    else:
         notZero = np.where(widsInt != 0) 
         
    # Get nice bounds (in range, multiple of downselect)
    downSize = downSelectF 
    halfwid = int(downSize /2)
    minpx = np.max([np.min(fpxx[notZero]) - downSize,0])
    maxpx = np.min([np.max(fpxx[notZero]+downSize), int(myMap.data.shape[1]-halfwid-1), maxX-halfwid-1])
    npx = int((maxpx-minpx) / downSize)
    minpy = np.max([np.min(fpyy[notZero]) - downSize,0])
    maxpy = np.min([np.max(fpyy[notZero]+downSize), int(myMap.data.shape[0]-halfwid-1), maxY-halfwid-1])
    npy = int((maxpy-minpy) / downSize)
    nzFoV = [minpx, maxpx, minpy, maxpy]
    nzpxs = np.arange(minpx, maxpx+1, downSize)
    nzpys = np.arange(minpy, maxpy+1, downSize)
    nzpxx, nzpyy = np.meshgrid(nzpxs, nzpys)
    

    #|-----------------------------------|
    #|--- Pull the mass data subfield ---|
    #|-----------------------------------|
    # Need to sum up the masses in all the pixels we are 
    # compressing into a single pix in the smaller FoV
    # Compress along pixel x direction
    subMass = np.zeros([massMap.shape[0], nzpyy.shape[1]])
    for i in range(len(nzpxs)):
        mypx = nzpxs[i]
        subMass[:,i] = 0.5*massMap[:,mypx-halfwid] + np.sum(massMap[:,mypx-halfwid+1:mypx+halfwid], axis=1) + 0.5*massMap[:,mypx+halfwid]
        if 0 in massMap[:,mypx-halfwid+1:mypx+halfwid]:
            for k in np.arange(mypx-halfwid+1,mypx+halfwid):
                idx = np.where(massMap[:,k] == 0)[0]
                subMass[idx,i] = 0
    
    # Compress along pixel y direction
    subsubMass = np.zeros(nzpyy.shape)
    for i in range(len(nzpys)):
        mypy = nzpys[i]
        subsubMass[i,:] = 0.5*subMass[mypy-halfwid,:] + np.sum(subMass[mypy-halfwid+1:mypy+halfwid,:], axis=0) + 0.5*subMass[mypy+halfwid,:]
        if 0 in subMass[mypy-halfwid+1:mypy+halfwid,:]:
            for k in np.arange(mypy-halfwid+1,mypy+halfwid):
                idx = np.where(subMass[k,:] == 0)[0]
                subsubMass[i,idx] = 0
    
    # Rename for funsies
    subMass = subsubMass

    
    #|--------------------------------------|
    #|--- Get grid cell area on subfield ---|
    #|--------------------------------------|
    # (in solar radii)
    f_fovx = FOV2x((nzpxx, nzpyy))
    f_fovy = FOV2y((nzpxx, nzpyy))
    f_fovz = FOV2z((nzpxx, nzpyy))
    
    # Could make this smoother, or at least better at edges, TBD
    dys = np.zeros(subMass.shape)
    dys[:,1:] = f_fovy[:,1:] - f_fovy[:,:-1]
    dys[:,0] = dys[:,1] 
    dzs = np.zeros(subMass.shape)
    dzs[1:,:] = f_fovz[1:,:] - f_fovz[:-1,:]
    dzs[0,:] = dzs[1,:]
    cellArea = dys * dzs
    dx = np.mean(dys)
        

    #|----------------------------------------|
    #|--- Get x pos and widths on subfield ---|
    #|----------------------------------------|
    
    #|--- Main wireframe ---|
    widsIntnz = widFunc((f_fovy, f_fovz))
    xcIntnz   = xcFunc((f_fovy, f_fovz))
    # Clean up the edge bc interp makes fuzzy
    widsIntnz[np.where(widsIntnz < dx*0.5)] = 0

    #|--- Inner region of main wireframe ---|
    if doInner:
        widsIntnzI = widFuncI((f_fovy, f_fovz))
        xcIntnzI   = xcFuncI((f_fovy, f_fovz))
        widsIntnzI[np.where(widsIntnzI < dx*0.5)] = 0
    else:
        widsIntnzI, xcIntnzI = None, None

    #|--- Outer wireframe ---|
    if multiMode:
        widsIntnz2 = widFunc2((f_fovy, f_fovz))
        xcIntnz2   = xcFunc2((f_fovy, f_fovz))
        widsIntnz2[np.where(widsIntnz2 < dx*0.5)] = 0
    else:
        widsIntnz2, xcIntnz2 = None, None
    
    #|--- Package for outputs ---|
    widMap  = [widsIntnz, widsIntnzI, widsIntnz2]    
    xcMap   = [xcIntnz, xcIntnzI, xcIntnz2]   
    outFoV  = [minpx, maxpx, minpy, maxpy, downSize]
    
                
    #|--------------------------|
    #|--- Get simple density ---|
    #|--------------------------|
    # Start with assuming full mass goes into WF1
    notZero = np.where(widsIntnz != 0)
    dens = np.zeros(subMass.shape)
    # Account for inner gap in WF1 if needed
    if doInner:
        dens[notZero] = subMass[notZero] / (widsIntnz[notZero] - widsIntnzI[notZero]) / cellArea[notZero] # g/Rs^3
    else:
        dens[notZero] = subMass[notZero] / widsIntnz[notZero] / cellArea[notZero] # g/Rs^3
        
    # Get density for WF2 but ignore effects of WF for now (correct below)
    dens2 = [None]
    if multiMode:
        notZero2 = np.where(widsIntnz2 != 0)
        dens2 = np.zeros(subMass.shape)  
        dens2[notZero2] = subMass[notZero2] / widsIntnz2[notZero2] / cellArea[notZero2] # g/Rs^3  
    
    
    #|-------------------------------------------|
    #|--- Get overlap region and adjacent pts ---|
    #|-------------------------------------------|
    ovMap = None
    if multiMode:
        #|--- Start by finding the overlap ---|
        w1 = widsIntnz
        if doInner:
            w1 = w1 - widsIntnzI
        w2 = widsIntnz2
            
        dx = np.mean(np.sqrt(cellArea))    
        overlap = np.where((w2 >= dx) & (w1 >= dx))
        ovys    = f_fovy[overlap]
        ovzs    = f_fovz[overlap]
        ovl     = np.zeros(f_fovy.shape)
        ovl[overlap] = 1
        
        #|--- Find the edge of the overlap ---|
        minOx, maxOx = np.min(overlap[1]), np.max(overlap[1])
        minOy, maxOy = np.min(overlap[0]), np.max(overlap[0])
        
        # Run through each vertical line
        for i in np.arange(minOx, maxOx+1):
            js = overlap[0][np.where(overlap[1] == i)]
            prej = -1
            for j in js: 
                if (j - prej) > 1:
                    ovl[j,i] = 2
                prej = j
            ovl[j,i] = 2

        # Run through each horiz line
        for j in np.arange(minOy, maxOy+1):
            iis = overlap[1][np.where(overlap[0] == j)]
            prei = -1
            for i in iis: 
                if (i - prei) > 1:
                    ovl[j,i] = 2
                prei = i
            ovl[j,i] = 2
            
        #|--- Find cells one outside/inside overlap ---|  
        # Mark as -2, 2  
        edge = np.where(ovl == 2)
        for k in range(len(edge[0])):
            i,j = edge[1][k], edge[0][k]
            if i-1 > 0: 
                if ovl[j,i-1] == 0: ovl[j,i-1] = -2                    
                if ovl[j,i-1] == 1: ovl[j,i-1] = 2                    
            if i+1 < ovl.shape[1]: 
                if ovl[j,i+1] == 0: ovl[j,i+1] = -2
                if ovl[j,i+1] == 1: ovl[j,i+1] = 2
                    
            if j-1 > 0: 
                if ovl[j-1,i] == 0: ovl[j-1,i] = -2
                if ovl[j-1,i] == 1: ovl[j-1,i] = 2
                    
            if j+1 < ovl.shape[0]: 
                if ovl[j+1,i] == 0: ovl[j+1,i] = -2
                if ovl[j+1,i] == 1: ovl[j+1,i] = 2


    #|-------------------------|
    #|--- Correct densities ---|
    #|-------------------------|
    if multiMode:
        # Assume constant ratio between WF1 and WF2
        # Originally did M = area * wid2 * n2
        # Want to switch to M = area * ((wid2 - wid1) * n2 + ratio * wid1 * n1)
        #scaleIt = np.zeros(ovl.shape)
        scaleIt = w2[overlap] / (w1[overlap] * densRatio + (w2 - w1)[overlap])
        dens[overlap]  = scaleIt * densRatio * dens2[overlap]
        dens2[overlap] = scaleIt * dens2[overlap]
        dens2 = dens2 / (6.957e10 **3 )
    dens = dens / (6.957e10 **3 )
    densMap = [dens , dens2 ] # convert to g/cm^3
    
    # 2d plotting example (for testing)
    if False:
       fig = plt.figure()
       vval = 1e12
       plt.imshow(dens2, vmin=-vval, vmax=vval, origin='lower')
       plt.show()

    #|---------------------|
    #|--- Return things ---|
    #|---------------------|
    return widMap, xcMap, densMap, subMass, outFoV, [FOV2x, FOV2y, FOV2z] 



# |-----------------------------|
# |--- 3D Density Cloud plot ---|
# |-----------------------------|
def dingo3d(widMap, xcMap, densMap, outFoV, pix2FOV, shell=True, plotIt=True):
    #|------------------|
    #|--- Prep stuff ---|
    #|------------------|
    #|--- Unpackage FoV things ---|
    minpx, maxpx, minpy, maxpy, downSize = outFoV
    
    #|--- Check for inner data ---|
    doInner = False
    if type(widMap[1]) != type(None):
        doInner = True

    #|--- Check for second WF ---|
    multiMode = False
    if type(widMap[2]) != type(None):
        multiMode = True
    
    #|--- Make the mini FoV grid in pix ---|
    pxs = np.arange(minpx, maxpx+1, downSize)
    pys = np.arange(minpy, maxpy+1, downSize)
    pxx, pyy = np.meshgrid(pxs, pys)
    
    #|--- Make the mini FoV grid in Rs ---|
    fovx = pix2FOV[0]((pxx, pyy))
    fovy = pix2FOV[1]((pxx, pyy))
    fovz = pix2FOV[2]((pxx, pyy))
    
    #|--- Get resolution ---|
    dys = np.zeros(fovx.shape)
    dys[:,1:] = fovy[:,1:] - fovy[:,:-1]
    dys[:,0] = dys[:,1] 
    dzs = np.zeros(fovx.shape)
    dzs[1:,:] = fovz[1:,:] - fovz[:-1,:]
    dzs[0,:] = dzs[1,:]
    cellArea = dys * dzs
    
    #|--- Get points with non zero wids ---|
    if type(widMap[2]) != type(None):
        notZero  = np.where((widMap[0] != 0) | (widMap[2] != 0))
        notZero2 = np.where(widMap[2] != 0)
    else:
        notZero = np.where(widMap[0] != 0)

    #|---------------------------|
    #|--- Expand in the x dim ---|
    #|---------------------------|
    #|--- Get map of front/back x vals ---|
    maxxMap, minxMap = [], []
    for i in range(3):
        if type(widMap[i]) != type(None):
            maxxMap.append(xcMap[i] + 0.5 * widMap[i])
            minxMap.append(xcMap[i] - 0.5 * widMap[i])
        else:
            maxxMap.append(None)
            minxMap.append(None)
    

    #|--- Get values at non zero wid grid cells---|    
    yn0    = fovy[notZero]
    zn0    = fovz[notZero]
    xn0    = zn0 * 0
    wn0    = widMap[0][notZero]
    xcn0   = xcMap[0][notZero]
    dn0    = densMap[0][notZero]
    if doInner:
        wn0I  = widMap[1][notZero]
        xcn0I = xcMap[1][notZero]
        
    #|--- Set x resolution at mean of y ---|    
    dx = np.mean(dys) # should be sufficient
    
    #|--- Get # of points in x dim ---|
    nptsx = wn0 / dx / 2 
    nptsx = nptsx.astype(int)
    
    
    
    #|-----------------------------|
    #|--- Make the cloud points ---|
    #|-----------------------------|
    #|--- Build an array of points ---|    
    # allpts = [x, y, z, dens]
    allpts = [np.array([]), np.array([]), np.array([]), np.array([])]
    
    #|--- Loop through nonzero cells ---|
    for i in range(len(yn0)):
        myxs = None
        iy, iz = notZero[0][i], notZero[1][i]
        
        # |--- Check if we have at least one pt in width ---|
        if nptsx[i] > 0:
            if shell:
                myxs = np.array([minxMap[0][iy, iz], maxxMap[0][iy, iz]])    
            else:
                myxs = dx *(np.arange(-nptsx[i], nptsx[i]+1)) + xcMap[0][iy, iz]
        elif wn0[i] > 0.9*dx:
            myxs = np.array([xcMap[0][iy, iz]])
        
        # |--- Grab the matching y, z, dens ---|
        if (type(myxs) != type(None)) & (np.abs(xcMap[0][iy, iz]) < 10*wn0[i]):
            if shell:
                myys  = fovy[iy, iz] * np.ones(len(myxs))
                myzs  = zn0[i] * np.ones(len(myxs))
                mydens = dn0[i] * np.ones(len(myxs))
            else:
                myys  = yn0[i] * np.ones(2*nptsx[i] + 1)
                myzs  = zn0[i] * np.ones(2*nptsx[i] + 1)
                mydens = dn0[i] * np.ones(2*nptsx[i] + 1)
            
            # |--- Account for the inner gap ---|
            if doInner:
                if wn0I[i] > dx:
                    # |--- Add inner part of shell ---|
                    if shell:
                        xIa, xIb = minxMap[1][iy, iz], maxxMap[1][iy, iz]
                        myxsI = np.concatenate([xIa * np.ones(5), [xIa-dx, xIa+dx],  xIb * np.ones(5), [xIb-dx, xIb+dx]])
                        myysI = np.array([0, -dx, dx, 0, 0, 0, 0, 0, -dx, dx, 0, 0, 0, 0]) + myys[0]
                        myzsI = np.array([0, 0, 0, -dx, dx, 0, 0, 0, 0, 0, -dx, dx, 0, 0]) + myzs[0]
                        myxs  = np.concatenate((myxs, myxsI))
                        myys  = np.concatenate((myys, myysI))
                        myzs  = np.concatenate((myzs, myzsI))
                        mydens  = np.concatenate((mydens, mydens[0]*np.ones(14)))
                    # |--- Or remove points from full structure ---|
                    else:
                        gapx = xcn0I[i]
                        gapwid = wn0I[i] / 2.
                        dists = np.abs(myxs - gapx)
                        outgap = np.where(dists > gapwid)
                        myxs = myxs[outgap]
                        myys = myys[outgap]
                        myzs = myzs[outgap]
                        mydens = mydens[outgap]
                        
            # |--- Add to output lists ---|   
            # Make sure density isn't exactly 0 -> img gaps:
            if dn0[i] != 0: 
                allpts[0] = np.concatenate((allpts[0], myxs))
                allpts[1] = np.concatenate((allpts[1], myys))
                allpts[2] = np.concatenate((allpts[2], myzs))
                allpts[3] = np.concatenate((allpts[3], mydens))
            
    #|----------------------------|
    #|--- Repeat for second WF ---|
    #|----------------------------|
    allpts2 = None
    if multiMode:
        #|--- Get values at non zero wid grid cells---|
        ynot2    = fovy[notZero2]
        znot2    = fovz[notZero2]
        xnot2    = znot2 * 0
        wnot2    = widMap[2][notZero2]
        xcnot2   = xcMap[2][notZero2]
        dnot2    = densMap[1][notZero2]
        
        #|--- Get # of points in x dim ---|
        nptsx2 = wnot2 / dx / 2 
        nptsx2 = nptsx2.astype(int)
        
        # |--- Get overlap points ---|
        overlap = np.where((widMap[2] >= dx) & (widMap[0] >= dx))
        ovys    = fovy[overlap]
        ovzs    = fovz[overlap]
        
        #|--- Build an array of points ---|    
        # allpts2 = [x, y, z, dens]
        allpts2 = [np.array([]), np.array([]), np.array([]), np.array([])]
        
        #|--- Loop through nonzero cells ---|
        for i in range(len(ynot2)):
            # |--- Check if we have at least one pt in width ---|
            myxs2 = None
            iy, iz = notZero2[0][i], notZero2[1][i]
            if nptsx2[i] > 0:
                if shell:
                    myxs2 = np.array([minxMap[2][iy, iz], maxxMap[2][iy, iz]]) 
                else:
                    myxs2 = dx *(np.arange(-nptsx2[i], nptsx2[i]+1))
            #elif wnot2[i] > 0.9*dx:
            #    myxs2 = np.zeros(1)
                
            # |--- Grab the matching y, z, dens ---|
            if (type(myxs2) != type(None)) & (np.abs(xcMap[2][iy, iz]) < 10*wnot2[i]):                
                if shell:
                    myys2  = ynot2[i] * np.ones(len(myxs2))
                    myzs2  = znot2[i] * np.ones(len(myxs2))
                    mydens2 = dnot2[i] * np.ones(len(myxs2))
                    idxOut = range(len(myxs2))
                else:
                    #myxs2 =  myxs2+xcnot2[i]
                    myys2  = ynot2[i] * np.ones(2*nptsx2[i] + 1)
                    myzs2  = znot2[i] * np.ones(2*nptsx2[i] + 1)
                    mydens2 = dnot2[i] * np.ones(2*nptsx2[i] + 1)
                    inOV = (ynot2[i] in ovys) and (znot2[i] in ovzs)
                    idxOut = range(len(myxs2))
                    
                    #|--- If doing full cloud, take out inner WF1 points ---|
                    if inOV:
                        iy, iz = notZero2[0][i], notZero2[1][i]
                        myxc, myw = xcMap[0][iy,iz], widMap[0][iy,iz]/2
                        myxc2, myw2 = xcMap[2][iy,iz], widMap[2][iy,iz]/2
                    
                        idxa = np.where(myxs2 > (myxc + myw))[0]
                        idxb = np.where(myxs2 < (myxc - myw))[0]
                        idxOut = np.concatenate((idxa, idxb))
                    
                # |--- Add to output lists ---|    
                allpts2[0] = np.concatenate((allpts2[0], myxs2[idxOut]))
                allpts2[1] = np.concatenate((allpts2[1], myys2[idxOut]))
                allpts2[2] = np.concatenate((allpts2[2], myzs2[idxOut]))
                allpts2[3] = np.concatenate((allpts2[3], mydens2[idxOut]))
    
    
    # |--- Logify densities ---|
    negPts = np.where(allpts[3] <= 0)
    if len(negPts[0]) > 0:
        allpts[3][negPts] = np.min(np.abs(allpts[3]))
    logd = np.log10(allpts[3])
    
    if multiMode:
        negPts2 = np.where(allpts2[3] <= 0)
        if len(negPts2[0]) > 0:
            allpts2[3][negPts2] = np.min(np.abs(allpts2[3]))
        logd2 = np.log10(allpts2[3])

        
    #|------------------------|
    #|--- Make the 3D Plot ---|
    #|------------------------|
    if plotIt:
        # Option to downselect the number of pts shown
        # (gets laggy with filled structures)
        if shell:
            showLess1 = 1
            idx1 = range(len(allpts[0]))
            if multiMode:
                showLess2 = 1
                idx2 = np.arange(0,len(allpts2[0])-1)
                np.random.shuffle(idx2)
                idx2 = idx2[::showLess2]
            alpha2 = 0.1
        else:
            showLess1 = 2
            idx1 = np.arange(0,len(allpts[0])-1)
            np.random.shuffle(idx1)
            idx1 = idx1[::showLess1]
            if multiMode:
                showLess2 = 4
                idx2 = np.arange(0,len(allpts2[0])-1)
                np.random.shuffle(idx2)
                idx2 = idx2[::showLess2]
            alpha2   = 0.15
        # Guess at nice density range    
        vval = 1e9 /dx**3
    
        # |--- Initiate figure ---|
        fig = plt.figure(figsize=(8, 5), layout='constrained')
        ax = fig.add_subplot(111, projection='3d')
        # WF1 scatter
        im = ax.scatter(allpts[0][idx1], allpts[1][idx1], allpts[2][idx1], c=logd[idx1], cmap='Reds')
        if multiMode:
            # WF2 scatter
            im2 = ax.scatter(allpts2[0][idx2], allpts2[1][idx2], allpts2[2][idx2], c=logd2[idx2], cmap='Blues', alpha=alpha2)

        # Add contour bar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.05, pad=0.02, shrink=0.5) 
        cbar.set_label('Log$_{10}$ Density\n(g cm$^{-3}$)')
        if multiMode:
            cbar2 = fig.colorbar(im2, ax=ax, orientation='vertical', fraction=.05, pad=0.02, shrink=0.5, location='left') 
            cbar2.set_label('Log$_{10}$ Density\n(g cm$^{-3}$)')
        
        # Prettify
        ax.set_aspect('equal') 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    
    return [allpts, allpts2]
    

# |----------------------------|
# |--- Density Contour plot ---|
# |----------------------------|
def dingo2d(widMaps, densMaps, outFoVs, pix2FOVs, showLog=False, figName=None, times=None):
    # The first four params need to be packaged as lists, even if passing a single time    
    
    #|------------------|
    #|--- Prep stuff ---|
    #|------------------|

    #|--- Get number of time steps
    nTimes = len(widMaps)
    
    #|--- Check for second WF ---|
    multiMode = False
    mmtag = 's'
    if type(widMaps[0][2]) != type(None):
        multiMode = True
        mmtag = 'm'
    
    #|---------------------|    
    #|--- Set up figure ---|
    #|---------------------|
    myStyle = str(nTimes) + mmtag # eg 1s, 3m for single or multi
    # Figure size dictionary
    fSizes  = {'1s':(6.8,5.),'2s':(4,5), '3s':(4,8.), '4s':(4,10.), '5s':(8,8.), '6s':(8,8.), '7s':(6.5,11.), '8s':(6.5,11.), '1m':(7,3.8),'2m':(7,6.6), '3m':(7,8.), '4m':(7,10.), '5m':(10,7.), '6m':(10,7.), '7m':(10,9.), '8m':(10,9.)}
    # Number of plot panels and width_ratios
    rval = 0.88
    lval = 0.1
    if (nTimes <= 4) and not multiMode:
        gnx = 2
        gny = nTimes
        gx  = [0]
        ios = [0] # inner (0) or outer (1)
        widRats = [1,0.1]
        if nTimes != 1:
            rval = 0.80
        else:
            rval = 0.85
            lval = 0.15
    elif (nTimes <= 4) and multiMode:
        gnx = 3
        gny = nTimes
        gx  = [1, 0]
        ios = [0, 1]
        widRats = [1,1,0.1]
    elif (nTimes > 4) and not multiMode:
        gnx = 4
        gny = math.ceil(nTimes/2)
        gx  = [0, 2]
        ios = [0, 0]
        widRats = [1, 0.1, 1, 0.1]    
    elif (nTimes > 4) and multiMode:
        gnx = 6
        gny = math.ceil(nTimes/2)
        gx  = [1, 0, 4, 3]
        ios = [0, 1, 0, 1]
        widRats = [1, 1, 0.1, 1, 1, 0.1]
    # indices of where we actually want contour panels
    gy = range(gny)
    
    fig = plt.figure(figsize=fSizes[myStyle])
    gs = gridspec.GridSpec(gny, gnx, width_ratios=widRats)
    gs.update(wspace=0.05)
    gs.update(hspace=0.15)
    
    axes = [[], []]
    count = [1,1]
    for i in range(len(gx)):
        for j in gy:
            if count[ios[i]] <= nTimes:
                anAx = plt.subplot(gs[j,gx[i]])        
                axes[ios[i]].append(anAx)
                anAx.set_aspect('equal')
                # Turn off tick labels or add main label
                if gx[i] != 0:
                    anAx.set_yticklabels([])
                else:
                    anAx.set_ylabel('Pixels')
                
                if (j != (gny-1)) and (count[ios[i]] != nTimes):
                    anAx.set_xticklabels([])
                else:
                    anAx.set_xlabel('Pixels')
                count[ios[i]] += 1    

    # Make cbar axis
    if gny <=2:
        cax = plt.subplot(gs[:,-1])
    elif gny == 3:
        cax = plt.subplot(gs[1,-1])
    elif gny == 4:
        cax = plt.subplot(gs[1:3,-1])
    

    #|--------------------------|    
    #|--- Unpackage all FoVs ---|
    #|--------------------------| 
    # The FoV is a square surrounding the nonzero region
    # but each time will have a diff square
    # The values are pix wrt the original size given to
    # mass2dens (probably 1024 unless adventurous code usage)
    minpxs, maxpxs, minpys, maxpys, downSizes = [], [], [], [], []
    for i in range(nTimes):   
        minpx, maxpx, minpy, maxpy, downSize = outFoVs[i]
        minpxs.append(minpx)
        maxpxs.append(maxpx)
        minpys.append(minpy)
        maxpys.append(maxpy)
        # downsize should be same for all
    # Get min/max range in each of xy
    limxs = [np.min(minpxs), np.max(maxpxs)]    
    limys = [np.min(minpys), np.max(maxpys)]    
    npx = limxs[1] - limxs[0] + 1
    npy = limys[1] - limys[0] + 1
    nsq = np.max([npx, npy])
    if npx < nsq:
        limxs[1] += nsq - npx 
    elif npy < nsq:
        limys[1] += nsq - npy 
    
    #|-------------------------|    
    #|--- Process each time ---|
    #|-------------------------|
    # Holders for the results
    allDens1 = np.zeros([nTimes, nsq, nsq])
    allpxx   = np.zeros([nTimes, nsq, nsq])
    allpyy   = np.zeros([nTimes, nsq, nsq])
    if multiMode:
        allDens2 = np.zeros([nTimes, nsq, nsq])
    
    # |--- Time loop ---|    
    for i in range(nTimes):   
        #|--- Unpackage FoV things ---|
        minpx, maxpx, minpy, maxpy, downSize = outFoVs[i]
        dx = minpx - limxs[0]
        dy = minpx - limxs[0]
        sx = maxpx - minpx + 1
        sy = maxpy - minpy + 1
    
        dens1 = densMaps[i][0]
        if multiMode:
            dens2 = densMaps[i][1]

        #|--- Make the mini FoV grid in pix ---|
        pxs = np.arange(minpx, maxpx+1, downSize)
        pys = np.arange(minpy, maxpy+1, downSize)
        pxx, pyy = np.meshgrid(pxs, pys)
    
        #|--- Make the mini FoV grid in Rs ---|
        fovx = pix2FOVs[i][0]((pxx, pyy))
        fovy = pix2FOVs[i][1]((pxx, pyy))
        fovz = pix2FOVs[i][2]((pxx, pyy))
        
        #|--- Get resolution ---|
        dys = np.zeros(fovx.shape)
        dys[:,1:] = fovy[:,1:] - fovy[:,:-1]
        dys[:,0] = dys[:,1] 
        dzs = np.zeros(fovx.shape)
        dzs[1:,:] = fovz[1:,:] - fovz[:-1,:]
        dzs[0,:] = dzs[1,:]
        cellArea = dys * dzs
    
        allDens1[i,dy:dy+sy,dx:dx+sx] = dens1
        allpxx[i,dy:dy+sy,dx:dx+sx] = pxx
        allpyy[i,dy:dy+sy,dx:dx+sx] = pyy
        if multiMode:
            allDens2[i,dy:dy+sy,dx:dx+sx] = dens2
            
        # |--- Logify densities ---|
        if showLog:
            negPts = np.where(dens1 <= 0)
            if len(negPts[0]) > 0:
                minval = np.min(np.abs(dens1[np.where(np.abs(dens1) > 0)]))
                dens1[negPts] = minval
            logd1 = np.log10(dens1)
            
            if multiMode:
                negPts2 = np.where(dens2 <= 0)
                if len(negPts2[0]) > 0:
                    minval2 = np.min(np.abs(dens2[np.where(np.abs(dens2) > 0)]))
                    dens2[negPts2] = minval2
                logd2 = np.log10(dens2)
            #|--- Replace non log in the holder ---|    
            allDens1[i,dy:dy+sy,dx:dx+sx] = logd1
            if multiMode:
                allDens2[i,dy:dy+sy,dx:dx+sx] = logd2
                
    #|----------------------|
    #|--- Fill in figure ---|
    #|----------------------|
    # |--- Get density range to set contours ---|
    vval = np.median(np.abs(allDens1[allDens1 !=0])) * 3 
    cmap = plt.get_cmap('RdYlBu_r')
    cmap.set_under('k')
    
    power = int(np.log10(vval)) - 1
    scaleIt = 10 ** power
    vval = vval / scaleIt
    
    for i in range(nTimes):
        thisdens1 = allDens1[i,:,:]
        thisdens1[np.where(thisdens1 == 0)] = -9999
        # Grab the first for the cbar
        if i == 0:
            im = axes[0][i].imshow(thisdens1/scaleIt, origin='lower', vmin=-vval, vmax=vval, cmap=cmap, extent=[limxs[0], limxs[1], limys[0], limys[1]])
        else:
            axes[0][i].imshow(thisdens1/scaleIt, origin='lower', vmin=-vval, vmax=vval, cmap=cmap, extent=[limxs[0], limxs[1], limys[0], limys[1]])
        if type(times) != type(None):
            axes[0][i].set_title(times[i], fontsize=10,loc='right')
        
        if multiMode:
            thisdens2 = allDens2[i,:,:]
            thisdens2[np.where(thisdens2 == 0)] = -9999
            axes[1][i].imshow(thisdens2/scaleIt, origin='lower', vmin=-vval, vmax=vval, cmap=cmap, extent=[limxs[0], limxs[1], limys[0], limys[1]])
            xs = np.arange(limxs[0], limxs[1]+1, 1)
            ys = np.arange(limys[0], limys[1]+1, 1)
            xxxs, yyys = np.meshgrid(xs, ys)
            axes[0][i].contour(xxxs, yyys,thisdens2, levels=[-9999], linestyles='--', colors='w')
            axes[1][i].contour(xxxs, yyys,thisdens1, levels=[-9999], linestyles='--', colors='k')
    
    # Add the color bar
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')     
    cbar.set_label('Density (1e'+str(power)+' g cm$^{-3}$)', rotation=270, labelpad=15)
    fig.subplots_adjust(right=rval, left=lval, top=0.95,bottom=0.1)
    if figName:
        plt.savefig('dingo2d_'+figName+picType)
    else:
        plt.show()
    


# |--------------------|
# |--- In Situ plot ---|
# |--------------------|
def dingo1d(myMaps, widMapIns, xcMapIns, densMapIns, outFoVs, pix2FoVs, obsSats, vCME=400, scaleFactors=[1, 0.2], figName=None, timeMode='hr'):
    # The first four params need to be packaged as lists, even if passing a single time    
    
    # |---------------------|
    # |--- Set up figure ---|
    # |---------------------|
    fig, ax = plt.subplots(1, 2, figsize=(8,5), layout='constrained')
    ax[0].set_xlabel('R (R$_S$)')
    ax[0].set_ylabel('$\\rho$ (g cm$^{-3}$)')
    ax[0].set_title('In Place')
    ax[1].set_ylabel('n (cm$^{-3}$)')
    ax[1].set_title('Expected In Situ, v='+str(int(vCME))+'km/s')
    
    # |--------------------------------|
    # |--- Check if multi time mode ---|
    # |--------------------------------|
    multiMode = False
    nTimes = len(myMaps)
    myCols = ['k'] # to be replace when looping times
    startDiffs = np.zeros(nTimes)
    if nTimes > 1:
        multiMode = True
        
        # Set up colors
        times = []
        myCols = []
        cmap = plt.get_cmap('inferno')
        for aMap in myMaps:
            times.append(aMap.date.datetime)
        times = np.array(times)
        minTime = np.min(times)
        maxTime = np.max(times)
        dTime   = 1.1*(maxTime - minTime).total_seconds() # keep top out of lightest colors by padding
        for i in range(nTimes):
            startDiffs[i]= (times[i]-minTime).total_seconds()
            myCols.append(cmap(startDiffs[i]/dTime))
              
    # |---------------------|
    # |--- Run time loop ---|
    # |---------------------|
    for iii in range(nTimes):
        myMap = myMaps[iii]
        widMapIn = widMapIns[iii]
        xcMapIn = xcMapIns[iii]
        densMapIn = densMapIns[iii]
        outFoV    = outFoVs[iii]
        pix2FoV   = pix2FoVs[iii]
        obsSat    = obsSats[iii] 
        myC       = myCols[iii] 
        
        # |------------------------------|
        # |--- Get Satellite Location ---|
        # |------------------------------|
        # obsSat should be SkyCoord for in situ sat
        # Get satellite position from the map
        satLonD = myMap.observer_coordinate.lon.degree
        satLatD = myMap.observer_coordinate.lat.degree
        satR = myMap.observer_coordinate.radius.au 
        satLonR = satLonD * np.pi / 180.
        satLatR = satLatD * np.pi / 180.
       
        # |-----------------------------|
        # |--- Get important vectors ---|
        # |-----------------------------|
        # Get vector from sun to sat
        # (just the cartesian sat loc)
        satxyz = np.array([np.cos(satLatR)*np.cos(satLonR), np.cos(satLatR)*np.sin(satLonR), np.sin(satLatR)]) * satR
    
        pixCent = [myMap.data.shape[1]/2, myMap.data.shape[0]/2] # pix is xy but shape is yx

        # Get the heliprojective coord of the pixel
        coordM = myMap.pixel_to_world(pixCent[0] * u.pix, pixCent[1] * u.pix)
        # Get elongation angle -> distance to Thompson Sphere
        ell = np.sqrt(coordM.Tx.rad**2 + coordM.Ty.rad**2)
        dM = np.abs(satR * np.abs(np.cos(ell)))
        # Make skycoord with HPC and distance (transform needs dist if off limb)
        hpc = SkyCoord(Tx=coordM.Tx, Ty=coordM.Ty, distance=dM*u.au, frame= coordM.frame)
        # Convert to Stonyhurst, make a Cartesian array
        ston = hpc.transform_to(frames.HeliographicStonyhurst)
        TSxyz = np.array([ston.cartesian.x.to_value(), ston.cartesian.y.to_value(), ston.cartesian.z.to_value()])
        # Vectors we will need
        LoS = TSxyz - satxyz 
        usatxyz = satxyz / np.linalg.norm(satxyz)
        uTSxyz = TSxyz / np.linalg.norm(TSxyz)
        uLoS = LoS / np.linalg.norm(LoS)
    
        # |---------------------------------------|
        # |--- Package and convert to FoV Cart ---|
        # |---------------------------------------|
        # Start coord arrays with sat loc and FoV cent
        xs = [satxyz[0]*215, TSxyz[0]*215]
        ys = [satxyz[1]*215, TSxyz[1]*215]
        zs = [satxyz[2]*215, TSxyz[2]*215]
        
        # Add the in situ sat and Sun
        obsCart = obsSat.cartesian
        xs.append(obsCart.x.to_value()*215) 
        ys.append(obsCart.y.to_value()*215)
        zs.append(obsCart.z.to_value()*215)
        xs.append(0) 
        ys.append(0)
        zs.append(0)
        xs.append(xs[0]) 
        ys.append(ys[0])
        zs.append(zs[0])
    
        # Convert everyone at the same time
        pts = np.array([xs, ys, zs])      
        
        if 'crota' in myMap.meta:
            rollIt = myMap.meta['crota']
        elif 'sc_roll' in myMap.meta:
            rollIt = myMap.meta['sc_roll']
        else:
            print ('Neither crota or sc_roll in map metadata. Assuming zero roll')
            rollIt = 0

        res = StonyCart2CartFoV(pts, satLatD, satLonD, rollIt)
    
        # |--------------------------------------|
        # |--- Get Sun-sat intersect with pts ---|
        # |--------------------------------------|
        # Make a line connecting sun to sat
        npts = 100
    
        sat = [res[0][0], res[1][0], res[2][0]]
        sun = [res[0][1], res[1][1], res[2][1]]
        # Get the r dist of everyone (from sun)
        ssline = np.array([np.linspace(sun[0], sat[0], npts), np.linspace(sun[1], sat[1], npts), np.linspace(sun[2], sat[2], npts)])
        rs = np.sqrt((ssline[0] - sun[0])**2 + (ssline[1] - sun[1])**2 + (ssline[2] - sun[2])**2)
        ryzs = np.sqrt((ssline[1])**2 + (ssline[2])**2) # wrt FoV cent, not sun
        maxryz = np.max(ryzs)
     
        # Find max r to check for in place version
        #|--- Unpackage FoV things ---|
        minpx, maxpx, minpy, maxpy, downSize = outFoV

        #|--- Make the mini FoV grid in pix ---|
        pxs = np.arange(minpx, maxpx+1, downSize)
        pys = np.arange(minpy, maxpy+1, downSize)
        pxx, pyy = np.meshgrid(pxs, pys)
    
        #|--- Make the mini FoV grid in Rs ---|
        fovx = pix2FoV[0]((pxx, pyy))
        fovy = pix2FoV[1]((pxx, pyy))
        fovz = pix2FoV[2]((pxx, pyy))
        fov_ryz = np.sqrt(fovy**2 +fovz**2)
        maxFOV_ryz = np.max(fov_ryz)
    
        # Make a mini line where actually needed
        maxIdx = np.min(np.where(ryzs >= maxFOV_ryz))
        scaleIt = rs[maxIdx] / np.max(rs)
        miniLine = scaleIt * ssline
        minirs = np.sqrt((miniLine[0] - sun[0])**2 + (miniLine[1] - sun[1])**2 + (miniLine[2] - sun[2])**2)
    
        # |---------------------------------------|
        # |--- Check if within wireframe width ---|
        # |---------------------------------------|
        # Check if we have sheath widths or not
        incSheath = False
        if type(np.sum(widMapIn[2])) != type(None):
            incSheath = True
        
        widMap, xcMap, densMap = widMapIn[0], xcMapIn[0], densMapIn[0]
        if incSheath:
            widMaps, xcMaps, densMaps = widMapIn[2], xcMapIn[2], densMapIn[1]
        
        midpty, midptz = int(widMap.shape[1]/2), int(widMap.shape[0]/2)
        gridys = fovy[midptz,:]
        gridzs = fovz[:,midpty]
        rIP, nIP = [], []
        cs = []
        istype = []
        rMain = 9999.
        hitMain = False
        for i in range(npts):
            # |--- Interpolate to get local width ---|
            iy0, iyf, iz0, izf = None, None, None, None
            if (miniLine[1][i] <= np.max(gridys)) & (miniLine[1][i] >= np.min(gridys)):
                iy0 = np.min(np.where(gridys >= miniLine[1][i])[0])
                iyf = np.max(np.where(gridys <= miniLine[1][i])[0])
            if (miniLine[2][i] <= np.max(gridzs)) & (miniLine[2][i] >= np.min(gridzs)):
                iz0 = np.min(np.where(gridzs >= miniLine[2][i])[0])
                izf = np.max(np.where(gridzs <= miniLine[2][i])[0])
            
            if (type(iy0) != type(None)) & (type(iz0) != type(None)):
                w11, xc11, d11 = widMap[iz0, iy0], xcMap[iz0, iy0], densMap[iz0, iy0] 
                w12, xc12, d12 = widMap[iz0, iyf], xcMap[iz0, iyf], densMap[iz0, iyf]
                w21, xc21, d21 = widMap[izf, iy0], xcMap[izf, iy0], densMap[izf, iy0]
                w22, xc22, d22 = widMap[izf, iyf], xcMap[izf, iyf], densMap[izf, iyf]
                # Interp in y
                gdy = gridys[iyf] - gridys[iy0]
                fy = (miniLine[1][i] - gridys[iy0]) / gdy
                w1 = (1-fy) * w11 + fy * w12
                w2 = (1-fy) * w21 + fy * w22
                xc1 = (1-fy) * xc11 + fy * xc12
                xc2 = (1-fy) * xc21 + fy * xc22
                d1 = (1-fy) * d11 + fy * d12
                d2 = (1-fy) * d21 + fy * d22
  
                # Interp in z
                gdz = gridzs[izf] - gridzs[iz0]
                fz = (miniLine[2][i] - gridzs[iz0]) / gdz
                myw = (1-fz) * w1 + fz * w2
                myxc = (1-fz) * xc1 + fz * xc2
                myd = (1-fz) * d1 + fz * d2
            
                # |--- Repeat for the sheath ---|
                if incSheath:
                    w11s, xc11s, d11s = widMaps[iz0, iy0], xcMaps[iz0, iy0], densMaps[iz0, iy0] 
                    w12s, xc12s, d12s = widMaps[iz0, iyf], xcMaps[iz0, iyf], densMaps[iz0, iyf]
                    w21s, xc21s, d21s = widMaps[izf, iy0], xcMaps[izf, iy0], densMaps[izf, iy0]
                    w22s, xc22s, d22s = widMaps[izf, iyf], xcMaps[izf, iyf], densMaps[izf, iyf]
                    # Interp in y
                    gdy = gridys[iyf] - gridys[iy0]
                    fy = (miniLine[1][i] - gridys[iy0]) / gdy
                    w1s = (1-fy) * w11s + fy * w12s
                    w2s = (1-fy) * w21s + fy * w22s
                    xc1s = (1-fy) * xc11s + fy * xc12s
                    xc2s = (1-fy) * xc21s + fy * xc22s
                    d1s = (1-fy) * d11s + fy * d12s
                    d2s = (1-fy) * d21s + fy * d22s
  
                    # Interp in z
                    gdz = gridzs[izf] - gridzs[iz0]
                    fz = (miniLine[2][i] - gridzs[iz0]) / gdz
                    myws = (1-fz) * w1s + fz * w2s
                    myxcs = (1-fz) * xc1s + fz * xc2s
                    myds = (1-fz) * d1s + fz * d2s

                # |--- Check if in main or sheath ---|
                if (np.abs(miniLine[1][i] - myxc)<myw) and (myd != 0):
                    if not hitMain:
                        rMain = minirs[i]
                        hitMain = True
                    rMain = np.max([minirs[i], rMain])
                    rIP.append(minirs[i])
                    nIP.append(myd)
                    cs.append('m')
                    istype.append('m')
                elif incSheath and (minirs[i] > rMain):
                    if np.abs(miniLine[1][i] - myxcs)<myws:
                        rIP.append(minirs[i])
                        nIP.append(myds)
                        cs.append('b') 
                        istype.append('s')
                else:
                    rIP.append(minirs[i])
                    nIP.append(0)
                    cs.append('r')
                    istype.append('a')
            else:
                rIP.append(minirs[i])
                nIP.append(0)
                cs.append('a')
            
    
        # |-----------------------------------|
        # |--- Convert in place to in situ ---|
        # |-----------------------------------|
        # Scale in place density based on size at the in situ satellite        
        # Requires some form of simple expansion model
        rIP = np.array(rIP)
        nIP = np.array(nIP)
    
        # Find portion where it is actually CME/nonzero
        isCME = np.where(nIP != 0)[0]
        rCME  = rIP[isCME]
        nCME  = nIP[isCME]
    
        # Get midpoint, initial radial width, init dist
        Rmid0 = 0.5*(rCME[0] + rCME[-1])
        lilR0 = 0.5*(rCME[-1] - rCME[0])
        R0    = rCME[-1]
    
        # Make parametric array for the points
        # Extends from -1 at back to 1 at front
        fs = (rCME - Rmid0) / lilR0
    
        # Get arrival time for each point at sat
        dSat = satR *215 # satellite distance in Rs 
        nur  = scaleFactors[1]
        tArr = (dSat - R0 - lilR0 * fs) / (1 + nur * fs) / vCME * 7e5 / 3600 # in hr, assuming vCME in km/s

        # Get the radial size at the time of arrival
        lilRs = lilR0 + nur * vCME * tArr * 3600 / 7e5
    
        # Get the front dist for when each parametric point
        # is at the satellite
        dFront = dSat + (1 - fs) * lilRs
    
        # Get relative size in perp direction
        rprp0 =  scaleFactors[0] * dFront / rCME
    
        # Get ratio of volume at arrival to initial
        volRat = rprp0**2 * lilRs / lilR0
    
        # Scale density
        nIS = nIP[isCME] / volRat / 1.974e-24
    
        # Convert times to real time using map data
        # Tend not to use this bc neglects all non-expansion
        # interplanetary effects so garbage for arrival time
        deltaT = 3
        if timeMode == 'rw':
            remObsDT = myMap.date.datetime
            insituDT = []
            for aTime in tArr:
                insituDT.append(remObsDT + datetime.timedelta(hours = aTime))
            deltaT = datetime.timedelta(hours=3)

    
        # |---------------------|
        # |--- Plotting Time ---|
        # |---------------------|
    
        istype = np.array(istype)
        isAmb = np.where(istype == 'a')[0]
        isMain = np.where(istype =='m')[0]
        isSheath = np.where(istype == 's')[0]
    
        # |--- In place panel ---|
        # Don't actually care where is ambient, just throw in zeros
        # around main + sheath
        label = myMap.date.datetime.strftime("%Y-%m-%d %H:%M")
        ax[0].plot(rIP[isMain], nIP[isMain], c=myC, label=label)
        ax[0].plot([rIP[isMain[0]], rIP[isMain[0]]], [0, nIP[isMain[0]]], c=myC)
        ax[0].plot([0.5*rIP[isMain[0]], rIP[isMain[0]]], [0, 0], ':', c=myC)
        if len(isSheath) != 0:
            ax[0].plot(rIP[isSheath], nIP[isSheath], '--', c=myC)
            # Connect to ambient
            ax[0].plot([rIP[isSheath[-1]], rIP[isSheath[-1]]], [nIP[isSheath[-1]], 0], '--', c=myC)
            ax[0].plot([rIP[isSheath[-1]], 1.1* rIP[isSheath[-1]]], [0, 0], ':', c=myC)
            # Connect to main
            ax[0].plot([rIP[isMain[-1]], rIP[isSheath[0]]], [nIP[isMain[-1]], nIP[isSheath[0]]], c=myC)
        else:
            ax[0].plot([rIP[isMain[-1]], rIP[isMain[-1]]], [nIP[isMain[-1]], 0], c=myC)
            ax[0].plot([rIP[isMain[-1]], 1.1* rIP[isMain[-1]]], [0, 0], ':', c=myC)        
    
        ax[0].legend(loc='upper right')
    
        # |--- In situ panel ---|
        # Need new isMain/isSheath bc threw away ambient in IS calc
        isSheath = isSheath - isMain[0]
        isMain   = isMain - isMain[0]
    
        if timeMode == 'rw':
            xvals = insituDT
            #ax[1].plot(insituDT, nIS)
            plt.gcf().autofmt_xdate() 
        elif timeMode == 'hr':
            xvals = tArr + startDiffs[iii] / 3600.
            #ax[1].plot(tArr, nIS)
            ax[1].set_xlabel('t (hr)')
        # same plot script, different vars    
        # also time is backward from r
        ax[1].plot(xvals[isMain], nIS[isMain], c=myC)
        ax[1].plot([xvals[isMain[0]], xvals[isMain[0]]], [0, nIS[isMain[0]]], c=myC)
        ax[1].plot([xvals[isMain[0]] + deltaT , xvals[isMain[0]]], [0, 0], ':', c=myC)
        if len(isSheath) != 0:
            ax[1].plot(xvals[isSheath], nIS[isSheath], '--', c=myC)
            # Connect to ambient
            ax[1].plot([xvals[isSheath[-1]], xvals[isSheath[-1]]], [nIS[isSheath[-1]], 0], '--', c=myC)
            ax[1].plot([xvals[isSheath[-1]], xvals[isSheath[-1]]- deltaT] , [0, 0], ':', c=myC)
            # Connect to main
            ax[1].plot([xvals[isMain[-1]], xvals[isSheath[0]]], [nIS[isMain[-1]], nIS[isSheath[0]]], c=myC)
        else:
            ax[1].plot([xvals[isMain[-1]], xvals[isMain[-1]]], [nIS[isMain[-1]], 0], c=myC)
            ax[1].plot([xvals[isMain[-1]], xvals[isMain[-1]] - deltaT], [0, 0], ':', c=myC)      
        
    
    if figName:
        plt.savefig('dingo1d_'+figName+picType)
    else:
        plt.show()
    

# |------------------------------|
# |--- Calculate total masses ---|
# |------------------------------|
def getMasses(widMap, densMap, outFoV, pix2FOV, printIt=True):
    #|------------------|
    #|--- Prep stuff ---|
    #|------------------|
    #|--- Unpackage FoV things ---|
    minpx, maxpx, minpy, maxpy, downSize = outFoV
    
    dens1 = densMap[0]
    gidx1 = np.where(dens1 != -10)
    
    #|--- Check for second WF ---|
    multiMode = False
    if type(widMap[2]) != type(None):
        multiMode = True
        dens2 = densMap[1]
        gidx2 = np.where(dens2 != -10)

    #|--- Make the mini FoV grid in pix ---|
    pxs = np.arange(minpx, maxpx+1, downSize)
    pys = np.arange(minpy, maxpy+1, downSize)
    pxx, pyy = np.meshgrid(pxs, pys)
    
    #|--- Make the mini FoV grid in Rs ---|
    fovx = pix2FOV[0]((pxx, pyy))
    fovy = pix2FOV[1]((pxx, pyy))
    fovz = pix2FOV[2]((pxx, pyy))
    
    #|--- Get resolution ---|
    dys = np.zeros(fovx.shape)
    dys[:,1:] = fovy[:,1:] - fovy[:,:-1]
    dys[:,0] = dys[:,1] 
    dzs = np.zeros(fovx.shape)
    dzs[1:,:] = fovz[1:,:] - fovz[:-1,:]
    dzs[0,:] = dzs[1,:]
    cellArea = dys * dzs

    #widRs = [wid  for wid in widMap]
    vol1 = widMap[0] * cellArea * (6.96e10 **3)
    mass1 = np.sum(dens1[gidx1]*vol1[gidx1]) / 1e15
    if printIt:
        print ('Inner WF mass (1e15 g)', '{:.2f}'.format(mass1))
    if multiMode:
        vol2 = widMap[2] * cellArea * (6.96e10 **3)
        mass2 = np.sum(dens2[gidx2]*vol2[gidx2]) / 1e15
        if printIt:
            print ('Outer WF mass (1e15 g)', '{:.2f}'.format(mass2))
    if multiMode:
        return [mass1, mass2]
    else:
        return [mass1]
    

# |--------------------|
# |--- Main Wrapper ---|
# |--------------------|
def dingoWrapper(args, doInner=False):
    """ 
    Function that goes from the command line to the appropriate DINGO
    procedure. It can also be used for external calls by passing the 
    arguments to the function as a single array. The args must be passed
    in the order shown
    
    Inputs:
        args: an array/list with the following parameters (in order)
    
            logFile: the name/path for log file from WOMBAT
    
            ids: the integer line number(s) of the fit of interest in the 
                 logFile. Two fits can be passed (for sheath + eject) using
                 a plus sign with no spaces (e.g. 10+11 for lines 10 and 11)
    
            dingoDim: a strong for the dimensions for the density calculation. 
                      the option are 0D, 1D, 2D, 3D which correspond to:
                        0D - total mass
                        1D - line plot (in place and in situ)
                        2D - plane of sky contour plot
                        3D - interactive 3D scatter plot                    
                        
            saveName: the name to use when saving figures. this parameter is 
                      not required and will default to generic names if not
                      provided
    
            target: the name of the in situ satellite of interest. This is only 
                    needed for the 1D case and otherwise ignored Currently the
                    supported options are ACE, BepiColombo, DSCOVR, MAVEN, 
                    PSP, SolO, STEREO-A, STEREO-B, VEX, Wind. Additional
                    satellites could be added as long as they exist in
                    sunpy get_horizons_coord and the correct form of the tag
                    is used. We allow some common short forms of these names 
                    (e.g Bepi, STA, ...) and nothing is case sensitive. This
                    is 
            
    
    """
    
    #|--- Standard input error message ---|
    errorStrings = [' ', '  python3 dingo.py logFile id(s) dim otherParams', '          where:', '          - logFile is a wombat log file', '          - ids is an integer or int+int', '          - dim set the mode/dimensions from [0D, 1D, 2D, 3D]', '          - otherParams includes target (only for 1D), pic type,', '             save name, expf1_*, expf2_*, vcme_*, and densratio_*']
    
    
    #|-------------------------------------|
    #|-------------------------------------|
    #|--- Check the critical parameters ---|     
    #|-------------------------------------|
    #|-------------------------------------|
    
    #|----------------------------------|
    #|--- Check the number of inputs ---|     
    #|----------------------------------|
    nArgs = len(args)
    if (nArgs < 3) or (len(args) > 10):
        print ('Incorrect number of parameters provided. Syntax is')
        for astr in errorStrings:
            print (astr)
        sys.exit()
    
        
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
    saveName = None
    
    # Will work with no + (= single id)
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
            
    # Check that things match as needed
    if nplus > 0:
        txtIds = np.array(ids) - 1 # indexing from 0 in python
        miniLog = logFile[txtIds,:]
        
        # |--- Check for pickle/instrument match ---|
        if (len(np.unique(logFile[:, 13])) != 1) or (len(np.unique(miniLog[:, 1])) != 1):
            print ('Currently selecting ', np.unique(miniLog[:, 13]))
            print ('                and ', np.unique(miniLog[:, 1]))
            sys.exit('Can only combine results using the same wombat pickle and the same instrument.')
        
        # |--- Check for two compatible shapes or single ---|
        uniqTs = np.unique(miniLog[:, 2])
        uniqShapes = np.unique(miniLog[:, 3])
        if len(uniqShapes) > 2:
            print ('Currently selecting ', np.unique(miniLog[:, 3]))
            sys.exit('Can only combine results using two types of wireframes')
        elif len(uniqShapes) == 1:
            nTimes = len(uniqTs)
            # need to sort miniLog bc unique will sort
            # and were not doing any additional process for 
            # multi wf here
            sortIdx = np.argsort(miniLog[:,2])
            miniLog = miniLog[sortIdx,:]
        else:
            singleWF = False
            inWFs = ['GCS', 'Torus', 'GCS*', 'Tube']
            outWFs = ['Sphere', 'HalfSphere', 'Ellipse', 'HalfEllipse']
            if (uniqShapes[0] in inWFs) and (uniqShapes[1] in outWFs):
                inWF, outWF = uniqShapes[0], uniqShapes[1]
            elif (uniqShapes[1] in inWFs) and (uniqShapes[0] in outWFs):
                inWF, outWF = uniqShapes[1], uniqShapes[0]
            else:
                print ('Cannot combine selected WF types. Currently have ', uniqShapes)
                print ('Need to select one from each of ')
                print ('   ', inWFs, '(inner)')
                print ('   ', outWFs, '(outer)')
            
            # |--- Make sure each time has both inner/outer ---|
            pairTimes = []
            pairIds   = []
            for aTime in uniqTs:
                flagIt = False
                # Find the inner shape at this time
                inIndex  = np.where((miniLog[:,2] == aTime) & (miniLog[:,3] == inWF))[0]
                if len(inIndex) == 0:
                    flagIt = True
                    print ('Missing inner WF fit ('+inWF+') for', aTime, 'skipping it but running others')
                elif len(inIndex) > 1:
                    sys.exit('Multiple fits for '+inWF+ ' at time '+ aTime + ' cannot proceed')
                # Find the inner shape at this time    
                outIndex = np.where((miniLog[:,2] == aTime) & (miniLog[:,3] == outWF))[0]
                if len(outIndex) == 0:
                    flagIt = True
                    print ('Missing outer WF fit ('+outWF+') for', aTime, 'skipping it but running others')
                elif len(outIndex) > 1:
                    sys.exit('Multiple fits for '+ outWF+ ' at time '+ aTime + ' cannot proceed')
                if not flagIt:
                    pairTimes.append(aTime)
                    pairIds.append([inIndex[0], outIndex[0]])
            nTimes = len(pairTimes)     
    else:
        line = logFile[ids[0]-1,:]
        miniLog = np.array(line).reshape([1,-1])
        uniqTs = np.unique(miniLog[:, 2])
        uniqShapes = np.unique(miniLog[:, 3])
        nTimes = 1
            

    #|-------------------------------|
    #|--- Check the dimension tag ---|     
    #|-------------------------------|
    dim = args[2].lower()
    if dim in ['0', '0d']:
        mode = 0
    elif dim in ['1', '1d']:
        mode = 1
    elif dim in ['2', '2d']:
        mode = 2
        if nTimes > 8:
            sys.exit('Quitting... 2D mode only supports 8 time steps or fewer.')
    elif dim in ['3', '3d']:
        mode = 3
        if nTimes > 1:
            sys.exit('Quitting... 3D mode only supports single time and given more than one.')
    else:
        print ('Error in reading dimension tag. Full command line syntax is')
        for astr in errorStrings:
            print (astr)
        sys.exit()
    
    
    #|----------------------------------|
    #|----------------------------------|
    #|--- Check the bonus parameters ---|     
    #|----------------------------------|
    #|----------------------------------|    
    if len(args) >= 4:
        allBonus = args[3:]
        
        #|------------------------|
        #|--- Check for target ---|     
        #|------------------------|
        # Only in 1D mode
        satNames = ['ace', 'bepi', 'bepicolombo', 'dscovr', 'maven', 'parker', 'parkersolarprobe', 'parker_solar_probe', 'psp', 'solarorbiter', 'solo', 'so', 'stereoa', 'stereo-a', 'sta', 'stereob', 'stereo-b', 'stb', 'venusexpress', 'venus_express', 'vex', 've', 'wind']
        if mode == 1:
            target = None
            temp = []
            for aTag in allBonus:
                if aTag.lower() in satNames:
                    if type(target) == type(None):
                        targetIn = aTag.lower()
                        # Convert target to a code we know works with get_horizons_coord
                        if targetIn in ['ace', 'dscovr', 'maven', 'wind']:
                            target = targetIn
                        elif targetIn in ['ace']:
                            target = -92
                        elif targetIn in ['bepi', 'bepicolombo']:
                            target = 'bepi'
                        elif targetIn in ['parker', 'parkersolarprobe', 'parker_solar_probe', 'psp']:
                            target = 'psp'
                        elif targetIn in ['solarorbiter', 'solo', 'so']:
                            target = 'solo'
                        elif targetIn in ['stereoa', 'stereo-a', 'sta']:
                            target = 'stereo-a'
                        elif targetIn in ['stereob', 'stereo-b', 'stb']:
                            target = 'stereo-b'
                        elif targetIn in ['venusexpress', 'venus_express', 'vex']:
                            target = 'vex'
                    else:
                        print('Multiple target tags provided, cannot proceed')
                        print('   args include', target, aTag.lower())
                        print('')
                        sys.exit('(Check that save name is not an exact match to satellite tag)')
                else:
                    temp.append(aTag)
                allBonus = np.array(temp)
                
                
        #|--------------------------|
        #|--- Check for pic type ---|     
        #|--------------------------|
        temp = []
        for aTag in allBonus:
            if aTag.lower() in ['png', '.png', 'pdf', '.pdf']:
                picType = aTag.lower()
                if picType[0] != '.':
                    picType = '.' + picType
            else:
                temp.append(aTag)
        allBonus = np.array(temp)      
                
        #|----------------------------------------|
        #|--- Check for string specific params ---|     
        #|----------------------------------------|
        # Set the defaults
        expf1 = 1
        expf2 = 0.1
        densratio = 1.8
        vcme = 400
        
        temp = []
        for aTag in allBonus:
            if 'expf1_' in aTag.lower():
                try: 
                    expf1 = float(aTag.lower().replace('expf1_',''))
                except:
                    sys.exit('Error in converting '+aTag+' to expf1 float')
            elif 'expf2_' in aTag.lower():
                try: 
                    expf2 = float(aTag.lower().replace('expf2_',''))
                except:
                    sys.exit('Error in converting '+aTag+' to expf2 float')
            elif 'densratio_' in aTag.lower():
                try: 
                    densratio = float(aTag.lower().replace('densratio_',''))
                except:
                    sys.exit('Error in converting '+aTag+' to densratio float')
            elif 'vcme_' in aTag.lower():
                try: 
                    vcme = float(aTag.lower().replace('vcme_',''))
                except:
                    sys.exit('Error in converting '+aTag+' to vcme float')
            else:
                temp.append(aTag)
        allBonus = np.array(temp)
        
        #|-------------------------------------|
        #|--- Check for remaining save name ---|     
        #|-------------------------------------|
        saveName = None
        if len(allBonus) == 1:
            saveName = allBonus[0]
            # Pull out main name, no .txt .png .pdf
            if '.png' in saveName:
                saveName = saveName.replace('.png', '')
                picType  = '.png'
            elif '.pdf' in saveName:
                saveName = saveName.replace('.png', '')
                picType  = '.pdf'
            elif '.txt' in saveName:
                saveName = saveName.replace('.txt', '')
                
        elif len(allBonus) > 1:
            print ('Too many unprocessed inputs to assign the last one to save name')
            print ('Have', allBonus, 'remaining')
            
            print ('Full command line syntax is')
            for astr in errorStrings:
                print (astr)
            sys.exit()
            
               
    #|--------------------------|
    #|--- Set up run details ---|     
    #|--------------------------|        
    #|--- Open the pickle ---|
    if nplus == 0:
        line = logFile[ids[0]-1,:]
        miniLog = np.array(line).reshape([1,-1])
    else:
        line = miniLog[0,:]
    
    with open(line[13], 'rb') as file:
        bkgData = pickle.load(file)
    
    #|--- Get the instrument ---|
    # Same for all lines
    obs  = line[1]
    
    satDicts = []
    imMaps   = []
    showMaps = []
    massMaps = []
    for i in range(nTimes):
        if singleWF:
            tidx = int(miniLog[i,14])
        else:
            # Inner and outer at same tidx
            tidx = int(miniLog[pairIds[i][0],14])
        
        satDicts.append(bkgData['satStuff'][obs][0][tidx])
        imMaps.append(bkgData['proImMaps'][obs][0][tidx])
        showMaps.append(bkgData['scaledIms'][obs][0][tidx][0])
        massMaps.append(bkgData['massIms'][obs][tidx])

    #|--- Make the wireframe(s) ---|
    wfsI, wfsO = [], []
    aboutMe = []
    for i in range(nTimes):
        # Pull the lines
        if singleWF:
            lineI = miniLog[i,:]
            aboutMe.append(lineI[2] + ' ' + lineI[1]+ ' ' + lineI[3])
        else:
            lineI = miniLog[pairIds[i][0],:]
            lineO = miniLog[pairIds[i][1],:]
            aboutMe.append(lineI[2]+ ' ' + lineI[1] + ' ' + lineI[3] + ' ' + lineO[3])
        # Make the wfs
        aWFi = wf.wireframe(lineI[3])
        ps = []
        for i in range(9):
            if lineI[i+4] != 'None':
                ps.append(float(lineI[i+4]))
        aWFi.params = ps
        aWFi.getPoints()
        wfsI.append(aWFi)
        if not singleWF:
            aWFo = wf.wireframe(lineO[3])
            ps = []
            for i in range(9):
                if lineO[i+4] != 'None':
                    ps.append(float(lineO[i+4]))
            aWFo.params = ps
            aWFo.getPoints()
            wfsO.append(aWFo)


    #|--------------|
    #|--- Run it ---|     
    #|--------------|
    # Replace these with inputs?
    dI = False
    ds = 1
    if mode == 3:
        ds = 8
        
    #|--- Process mass maps into density ---|
    widMaps, xcMaps, densMaps, subMasss, outFoVs, pix2FOVs = [], [], [], [], [], []
    for i in range(nTimes):
        print ('Processing', uniqTs[i])
        if singleWF:
            widMap, xcMap, densMap, subMass, outFoV, pix2FOV  = mass2dens(imMaps[i], satDicts[i], wfsI[i], massMaps[i], doInner=dI,  downSelect=ds)
        else:
            widMap, xcMap, densMap, subMass, outFoV, pix2FOV  = mass2dens(imMaps[i], satDicts[i], [wfsI[i], wfsO[i]], massMaps[i], doInner=dI,  densRatio=densratio, downSelect=ds)
        # Package it
        widMaps.append(widMap)
        xcMaps.append(xcMap)
        densMaps.append(densMap)
        subMasss.append(subMass)
        outFoVs.append(outFoV) 
        pix2FOVs.append(pix2FOV)    
    
    #|-------------------------------|
    #|-------------------------------|
    #|--- Send to plotting script ---|
    #|-------------------------------|
    #|-------------------------------|
    
    #|-----------------------|
    #|--- 0d - total mass ---|
    #|-----------------------|
    if mode == 0:
        if singleWF:
            print ('Time Inst WF Mass(1e15 g)')
        else:
            print ('Time Inst WF1 WF2 Mass1(1e15 g) Mass2(1e15 g)')
        for i in range(nTimes):
            masses = getMasses(widMaps[i], densMaps[i], outFoVs[i], pix2FOVs[i], printIt=False)
            moreOut = ''
            for mass in masses:
                moreOut += '{:.2f}'.format(mass) + ' '

    #|----------------------|
    #|--- 1d - line plot ---|
    #|----------------------|
    elif mode ==1:
        obsSats = []
        for i in range(nTimes):
            obsSats.append(get_horizons_coord(target, time=satDicts[i]['DATEOBS']))
        dingo1d(imMaps, widMaps, xcMaps, densMaps, outFoVs, pix2FOVs, obsSats, vCME=vcme, scaleFactors=[expf1, expf2], figName=saveName)
        
    #|--------------------------|
    #|--- 2d - contour plots ---|
    #|--------------------------|
    elif mode == 2:
        dingo2d(widMaps, densMaps, outFoVs, pix2FOVs, figName=saveName, times=uniqTs)
            
    
    #|--------------------------------|
    #|--- 3d - Interactive scatter ---|
    #|--------------------------------|
    elif mode == 3:
        # Already forced to be single time so can use single versions of these vars
        allPts = dingo3d(widMap, xcMap, densMap,outFoV, pix2FOV, shell=True, plotIt=True)



# |-----------------------|
# |--- Text line input ---|
# |-----------------------|
if __name__ == '__main__':
    dingoWrapper(sys.argv[1:])



'''# python3 wombatProcessObs.py 2012-07-12T16:00 2012-07-13T05:00 HI1A rdiffHI
# python3 wombatProcessObs.py 2012-07-13T20:00 2012-07-14T12:00 HI2A rdiffHI
#theFile = 'wbPickles/WBGUI_temp.pkl'
#theFile = 'wbPickles/test_solopsp.pkl'

theFile = 'wbPickles/201207_COR2.pkl'

# Open the pickle
with open(theFile, 'rb') as file:
    bkgData = pickle.load(file)

# testing values 
# COR2a tidx 17
# SOLOHI 7

#tidx = 6
obs  ='COR2A'
#tidx = 7

if obs == 'COR2B':
    tidx = 6
elif obs == 'COR2A':
    tidx = 9

satDict = bkgData['satStuff'][obs][0][tidx]
imMap = bkgData['proImMaps'][obs][0][tidx]
showMap = bkgData['scaledIms'][obs][0][tidx][0]
massMap = bkgData['massIms'][obs][tidx]

#print (satDict['DATEOBS'])
#print (imMap)
dingoWrapper(sys.argv[1:])

#awf = wf.wireframe('GCS')
#awf.params = [253.5, 25.2, -13.5, 77.4, 54.9, 0.3]
#awf.params = [10., 25.2, -13.5, 77.4, 54.9, 0.3]
#awf.params = [50, 45, 0, 0., 50.0, 0.25]
#awf.params = [45.2, 77.4, 26.1, 40.5, 59.4, 0.3] # solohi 2059

# 20120712 COR2 at 17:54
if True:
    awf = wf.wireframe('GCS')
    awf.params = [11.54, 5.4, -9.0, 73.8, 45.45, 0.4]
awf.getPoints()

# 20120712 COR2 at 17:54
if True:
    awf2 = wf.wireframe('Sphere') 
    awf2.params = [11.93, 5.4, -6.3, 60.30]
    awf2.getPoints()
else:
    awf2 = None
    
#awf2.params = [11.2, 25.2, -2.7, 66] # COR2A 17:54
#awf2.params = [36.2, 43.2, -5.4, 70]
#awf2.params = [266.74, 25.2, -13.5, 55]
#awf2.getPoints()

ds = 1
dI = False
if type(awf2) == type(None):
    widMap, xcMap, densMap, subMass, outFoV, pix2FOV  = mass2dens(imMap, satDict, awf, massMap, doInner=dI,  downSelect=ds)  
else:
    widMap, xcMap, densMap, subMass, outFoV, pix2FOV  = mass2dens(imMap, satDict, [awf, awf2], massMap, doInner=dI,  densRatio=1.8, downSelect=ds)  

#allPts = dingo3d(widMap, xcMap, densMap,outFoV, pix2FOV, shell=True, plotIt=True)
#dingo2d(densMap, outFoV, pix2FOV, figName = 'DINGO_contour'+obs+'.png')
#dingo2d(densMap, outFoV, pix2FOV)
#getMasses(widMap, densMap, outFoV, pix2FOV)

obsSat = get_horizons_coord('Wind', time=satDict['DATEOBS'])
dingo1d(imMap, widMap, xcMap, densMap, outFoV, pix2FOV, obsSat, vCME=550, scaleFactors=[1., 0.1])'''
