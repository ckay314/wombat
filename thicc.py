import pickle
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.coordinates
import numpy as np
import matplotlib.pyplot as plt
from sunpy.coordinates import frames
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator
import matplotlib.cm as cm

import sys
sys.path.append('prepCode/') 
sys.path.append('wombatCode/') 
import wombatWF as wf
from wombatLoadCTs import *


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
                    if len(sortx) > 0:               
                        fullwid = sortx[-1] - sortx[0]
                        wids[j,i] = fullwid
                        midx[j,i] = 0.5*(sortx[-1] + sortx[0])
                        
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
        pixCent = [imMap.data.shape[1]/2, imMap.data.shape[0]/2] # pix is xy but shape is yx

    # Get the heliprojective coord of the pixel
    coordM = imMap.pixel_to_world(pixCent[0] * u.pix, pixCent[1] * u.pix)
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
        
    # Convert everyone at the same time
    pts = np.array([xs, ys, zs])      

    res = StonyCart2CartFoV(pts, satLatD, satLonD, myMap.meta['crota'])

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
        pixCent = [imMap.data.shape[1]/2, imMap.data.shape[0]/2] # pix is xy but shape is yx

    # Get the heliprojective coord of the pixel
    coordM = imMap.pixel_to_world(pixCent[0] * u.pix, pixCent[1] * u.pix)
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

    res = StonyCart2CartFoV(pts, satLatD, satLonD, myMap.meta['crota'])
    
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
    for i in range(myMap.data.shape[1])[::downSize]:
        for j in range(myMap.data.shape[0])[::downSize]:
            pixx.append(i)
            pixy.append(j)

    #ptsOut = map2CartFoV(myMap, [[0, myMap.data.shape[1]], [myMap.data.shape[0]/2, myMap.data.shape[0]/2]])
    ptsOut = map2CartFoV(myMap, [pixx, pixy])   # is [x,y,z] where each is same len as pixIn  
        
    uniX = np.unique(np.array(pixx))
    uniY = np.unique(np.array(pixy))
    maxX = np.max(uniX)
    maxY = np.max(uniY)
    
    FOVx = np.transpose(ptsOut[0].reshape([len(uniY), len(uniX)]))
    FOVy = np.transpose(ptsOut[1].reshape([len(uniY), len(uniX)]))
    FOVz = np.transpose(ptsOut[2].reshape([len(uniY), len(uniX)]))  
    
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
    
    # Compress along pixel y direction
    subsubMass = np.zeros(nzpyy.shape)
    for i in range(len(nzpys)):
        mypy = nzpys[i]
        subsubMass[i,:] = 0.5*subMass[mypy-halfwid,:] + np.sum(subMass[mypy-halfwid+1:mypy+halfwid,:], axis=0) + 0.5*subMass[mypy+halfwid,:]
    
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
def cloudPlot(widMap, xcMap, densMap, outFoV, pix2FOV, shell=True, plotIt=True):
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
        maxxMap.append(xcMap[i] + 0.5 * widMap[i])
        minxMap.append(xcMap[i] - 0.5 * widMap[i])
    

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
        if type(myxs) != type(None):
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
            if nptsx2[i] > 0:
                if shell:
                    myxs2 = np.array([-wnot2[i]/2, wnot2[i]/2])
                else:
                    myxs2 = dx *(np.arange(-nptsx2[i], nptsx2[i]+1))
            elif wnot2[i] > 0.9*dx:
                myxs2 = np.zeros(1)
                
            # |--- Grab the matching y, z, dens ---|
            if type(myxs2) != type(None):                
                if shell:
                    myys2  = ynot2[i] * np.ones(len(myxs2))
                    myzs2  = znot2[i] * np.ones(len(myxs2))
                    mydens2 = dnot2[i] * np.ones(len(myxs2))
                    idxOut = range(len(myxs2))
                else:
                    myxs2 =  myxs2+xcnot2[i]
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
    
    # Stop printing mass bc shell flag will make only shell mass
    #totalMass = np.sum(allpts[3]*(dx*6.957e10)**3)/1e15
    
    if multiMode:
        negPts2 = np.where(allpts2[3] <= 0)
        if len(negPts2[0]) > 0:
            allpts2[3][negPts2] = np.min(np.abs(allpts2[3]))
        logd2 = np.log10(allpts2[3])
    
        #totalMass2 = np.sum(allpts2[3]*(dx*6.957e10)**3)/1e15
        #print ('WF2 total mass (1e15 g): ', totalMass2)



        
    #|------------------------|
    #|--- Make the 3D Plot ---|
    #|------------------------|
    if plotIt:
        # Option to downselect the number of pts shown
        # (gets laggy with filled structures)
        if shell:
            showLess1 = 1
            showLess2 = 2
            alpha2 = 0.3
        else:
            showLess1 = 2
            showLess2 = 4
            alpha2   = 0.15
        # Guess at nice density range    
        vval = 1e9 /dx**3
    
        # |--- Initiate figure ---|
        ax = plt.figure().add_subplot(projection='3d')
        # WF1 scatter
        ax.scatter(allpts[0][::showLess1], allpts[1][::showLess1], allpts[2][::showLess1], c=logd[::showLess1], cmap='Reds')
        # WF2 scatter
        ax.scatter(allpts2[0][::showLess2], allpts2[1][::showLess2], allpts2[2][::showLess2], c=logd2[::showLess2], cmap='Blues', alpha=alpha2)
    
        # Add contour legend?
    
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
def contourDens(densMap, outFoV, pix2FOV, showLog=False, figName=None):
    #|------------------|
    #|--- Prep stuff ---|
    #|------------------|
    #|--- Unpackage FoV things ---|
    minpx, maxpx, minpy, maxpy, downSize = outFoV
    
    dens1 = densMap[0]
    

    #|--- Check for second WF ---|
    multiMode = False
    if type(widMap[2]) != type(None):
        multiMode = True
        dens2 = densMap[1]

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
    
    #|--------------------|
    #|--- Setup Figure ---|
    #|--------------------|
    scl =  (maxpx-minpx) / (maxpy-minpy) 
    if multiMode:
        fig, ax = plt.subplots(1, 2, figsize = (5*scl+4.5,6), sharex=True, layout='constrained')
    else:
        fig, ax = plt.subplots(1, 1)
        ax = [ax]
    
    vval = np.median(np.abs(dens1[dens1 !=0])) * 3 
    
    cmap = plt.get_cmap('RdYlBu_r')
    cmap.set_under('k')
    dens1[np.where(dens1 == 0)] = -10
    im = ax[-1].imshow(dens1, vmin=-vval, vmax=vval, cmap=cmap, extent=[minpx, maxpx, minpy, maxpy])
    ax[-1].set_xlabel('Pixels')
    ax[-1].set_ylabel('Pixels')
    
    if multiMode:
        dens2[np.where(dens2 == 0)] = -10
        ax[0].imshow(dens2, vmin=-vval, vmax=vval, cmap=cmap, extent=[minpx, maxpx, minpy, maxpy])
        ax[0].set_xlabel('Pixels')
        ax[0].set_ylabel('Pixels')
        ax[-1].set_title('Inner WF')
        ax[0].set_title('Outer WF')
        
        ax[-1].contour(pxx,pyy,dens2, levels=[-10], linestyles='--', colors='w')
        ax[0].contour(pxx,pyy,dens1, levels=[-10], linestyles='--', colors='k')
        
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=.03, pad=0.02) #
    cbar.set_label('Density (g cm$^{-3}$)')
    
    ax[0].set_aspect('equal')
    #fig.subplots_adjust(wspace=0.1)
    #plt.tight_layout()
    if figName:
        plt.savefig(figName)
    else:
        plt.show()

# |------------------------------|
# |--- Calculate total masses ---|
# |------------------------------|
def getMasses(widMap, densMap, outFoV, pix2FOV, showLog=False, figName=None):
    #|------------------|
    #|--- Prep stuff ---|
    #|------------------|
    #|--- Unpackage FoV things ---|
    minpx, maxpx, minpy, maxpy, downSize = outFoV
    
    dens1 = densMap[0]
    
    #|--- Check for second WF ---|
    multiMode = False
    if type(widMap[2]) != type(None):
        multiMode = True
        dens2 = densMap[1]

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
    print ('Inner WF mass (1e15 g)', '{:.2f}'.format(np.sum(dens1*vol1) / 1e15))
    if multiMode:
        vol2 = widMap[2] * cellArea * (6.96e10 **3)
        print ('Outer WF mass (1e15 g)', '{:.2f}'.format(np.sum(dens2*vol2) / 1e15))
    
# python3 wombatProcessObs.py 2012-07-12T16:00 2012-07-13T05:00 HI1A rdiffHI
# python3 wombatProcessObs.py 2012-07-13T20:00 2012-07-14T12:00 HI2A rdiffHI
theFile = 'wbPickles/WBGUI_temp.pkl'

# Open the pickle
with open(theFile, 'rb') as file:
    bkgData = pickle.load(file)

tidx = 5
obs  ='HI1A'
satDict = bkgData['satStuff'][obs][0][tidx]
imMap = bkgData['proImMaps'][obs][0][tidx]
showMap = bkgData['scaledIms'][obs][0][tidx][0]
massMap = bkgData['massIms'][obs][tidx]


awf = wf.wireframe('GCS')
awf.params = [23.5, 25.2, -13.5, 77.4, 54.9, 0.3]
#awf.params = [50, 45, 0, 0., 50.0, 0.25]
awf.getPoints()

awf2 = wf.wireframe('Half Sphere')
#awf2.params = [36.2, 43.2, -5.4, 70]
awf2.params = [26.74, 25.2, -13.5, 77]
awf2.getPoints()

widMap, xcMap, densMap, subMass, outFoV, pix2FOV  = mass2dens(imMap, satDict, [awf, awf2], massMap, doInner=True,  densRatio=0.8)    

#allPts = cloudPlot(widMap, xcMap, densMap,outFoV, pix2FOV, shell=True, plotIt=False)
contourDens(densMap, outFoV, pix2FOV)
getMasses(widMap, densMap, outFoV, pix2FOV)




if False:
        ax = plt.figure().add_subplot(projection='3d')
        #ax.scatter(f_fovx, f_fovy, f_fovz, c=widsInt)
        ax.scatter(f_fovx, f_fovy, f_fovz, c=xcIntnz)
        plt.show()

if False:
        fig = plt.figure()
        plt.imshow(widsIntnz2, extent=[fake_px[0], fake_px[-1], fake_py[0], fake_py[-1]], origin='lower')
        plt.show()
