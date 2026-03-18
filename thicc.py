import pickle
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.coordinates
import numpy as np
import matplotlib.pyplot as plt
from sunpy.coordinates import frames
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator

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

def getWidth(points, FoV=None, nGridY=100):
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
                zidx0 = np.max(np.where(zms <= myminz))
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
        ang = np.arccos(np.dot(uLoS0, uLoS))
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
    

def mass2dens(myMap, satDict, aWF, massMap, doInner=False):
    
    # Check if have single wireframe or multiple
    if (type(aWF) == type(wf.wireframe(None))):
        multiMode = False 
    
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
    awf.gPoints = [i * 10 for i in awf.gPoints]
    awf.getPoints()
    #awf.getPoints(inside=True)
    wfPts = wf2CartFoV(myMap, aWF)
    wfPtsT = np.transpose(np.array(wfPts))
    wids, midx, FoV, nGridY = getWidth(wfPtsT)
    dy, ygs, yms, nGridZ, zgs, zms = createGrid(FoV, nGridY)
    wid_smooth = ndimage.gaussian_filter(wids, sigma=2.0, order=0)
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

    # indexing of func is y,z    
    widFunc = RegularGridInterpolator((yms, zms), np.transpose(wid_smooth), method='linear', bounds_error=False, fill_value=0)  
    xcFunc  = RegularGridInterpolator((yms, zms), np.transpose(midx), method='linear', bounds_error=False, fill_value=0)
    
    # |--- Repeat process for the inside (if doing) ---|
    if doInner:
        awf.getPoints(inside=True)
        wfPtsI = wf2CartFoV(myMap, aWF)
        wfPtsI = np.transpose(np.array(wfPtsI))
        widsI, midxI, FoV, nGridY = getWidth(wfPtsI, FoV=FoV, nGridY=nGridY)
        wid_smoothI = ndimage.gaussian_filter(widsI, sigma=2.0, order=0)
        # fill in xc
        for i in range(midxI.shape[1]):
            notOutI = np.where(midxI[:,i] !=-9999)[0]
            if len(notOutI) >= 2:
                midxI[:notOutI[0], i]  = midxI[notOutI[0],i]
                midxI[notOutI[-1]:, i] = midxI[notOutI[-1],i]
        for i in range(midxI.shape[0]):
            notOutI = np.where(midxI[i,:] !=-9999)[0]
            if len(notOutI) >= 2:
                midxI[i,:notOutI[0]]  = midxI[i, notOutI[0]]
                midxI[i,notOutI[-1]:] = midxI[i, notOutI[-1]]

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
    

    #|----------------------------------------|
    #|--- Find bounding box where wid != 0 ---|
    #|----------------------------------------|
    notZero = np.where(widsInt != 0)
    
    # Get nice bounds (in range, multiple of downselect)
    downSize = 8
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
    #|--- Redo cart and wids on subfield ---|
    #|--------------------------------------|
    f_fovx = FOV2x((nzpxx, nzpyy))
    f_fovy = FOV2y((nzpxx, nzpyy))
    f_fovz = FOV2z((nzpxx, nzpyy))
    widsIntnz = widFunc((f_fovy, f_fovz))
    xcIntnz   = xcFunc((f_fovy, f_fovz))
    if doInner:
        widsIntnzI = widFuncI((f_fovy, f_fovz))
        xcIntnzI   = xcFuncI((f_fovy, f_fovz))
        
    if False:
        ax = plt.figure().add_subplot(projection='3d')
        #ax.scatter(f_fovx, f_fovy, f_fovz, c=widsInt)
        ax.scatter(f_fovx, f_fovy, f_fovz, c=xcIntnz)
        plt.show()

    if False:
        fig = plt.figure()
        plt.imshow(widsIntI, extent=[fake_px[0], fake_px[-1], fake_py[0], fake_py[-1]], vmin=0, vmax=30, origin='lower')
        plt.show()


    #|--------------------------------------|
    #|--- Get grid cell area on subfield ---|
    #|--------------------------------------|
    # (in solar radii)
    # Could make this smoother, or at least better at edges, TBD
    dys = np.zeros(subMass.shape)
    dys[:,1:] = f_fovy[:,1:] - f_fovy[:,:-1]
    dys[:,0] = dys[:,1] 
    dzs = np.zeros(subMass.shape)
    dzs[1:,:] = f_fovz[1:,:] - f_fovz[:-1,:]
    dzs[0,:] = dzs[1,:]
    cellArea = dys * dzs
    
    #|-------------------------------------|
    #|--- Get density where wid nonzero ---|
    #|-------------------------------------|
    notZero = np.where(widsIntnz != 0)
    dens = np.zeros(subMass.shape)
    if doInner:
        dens[notZero] = subMass[notZero] / (widsIntnz[notZero] - widsIntnzI[notZero]) / cellArea[notZero] # g/Rs^3
    else:
        dens[notZero] = subMass[notZero] / widsIntnz[notZero] / cellArea[notZero] # g/Rs^3
    
    # 2d plotting example (for testing)
    if False:
       fig = plt.figure()
       vval = 1e12
       plt.imshow(dens, vmin=-vval, vmax=vval, origin='lower')
       plt.show()

    #|---------------------------|
    #|--- Expand in the x dim ---|
    #|---------------------------|
    ynot0    = f_fovy[notZero]
    znot0    = f_fovz[notZero]
    xnot0    = znot0 * 0
    wnot0    = widsIntnz[notZero]
    xcnot0   = xcIntnz[notZero]
    dnot0    = dens[notZero]
    if doInner:
        wnot0I  = widsIntnzI[notZero]
        xcnot0I = xcIntnzI[notZero]
    
    dx = np.mean(dys) # should be sufficient
    nptsx = wnot0 / dx / 2 
    nptsx = nptsx.astype(int)
    
    allpts = [np.array([]), np.array([]), np.array([]), np.array([])]
    for i in range(len(ynot0)):
        myxs = None
        if nptsx[i] > 0:
            myxs = dx *(np.arange(-nptsx[i], nptsx[i]+1))
        elif wnot0[i] > 0.9*dx:
            myxs = np.zeros(1)
        if type(myxs) != type(None):
            myxs =  myxs+xcnot0[i]
            myys  = ynot0[i] * np.ones(2*nptsx[i] + 1)
            myzs  = znot0[i] * np.ones(2*nptsx[i] + 1)
            mydens = dnot0[i] * np.ones(2*nptsx[i] + 1)
            
            if doInner:
                if wnot0I[i] > dx:
                    gapx = xcnot0I[i]
                    gapwid = wnot0I[i] / 2.
                    dists = np.abs(myxs - gapx)
                    outgap = np.where(dists > gapwid)
                    myxs = myxs[outgap]
                    myys = myys[outgap]
                    myzs = myzs[outgap]
                    mydens = mydens[outgap]
                
            allpts[0] = np.concatenate((allpts[0], myxs))
            allpts[1] = np.concatenate((allpts[1], myys))
            allpts[2] = np.concatenate((allpts[2], myzs))
            allpts[3] = np.concatenate((allpts[3], mydens))
            
    # polish density    
    allpts[3] = allpts[3] / (6.957e10 **3 ) # convert to g/cm^3   
    negPts = np.where(allpts[3] <= 0)
    if len(negPts[0]) > 0:
        allpts[3][negPts] = np.min(np.abs(allpts[3]))
    logd = np.log10(allpts[3])
    
    totalMass = np.sum(allpts[3]*(dx*6.957e10)**3)/1e15
    
    # 3d plotting example (for testing)
    if True:
        showLess = 10
        vval = 1e9 /dx**3
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(allpts[0][::showLess], allpts[1][::showLess], allpts[2][::showLess], c=logd[::showLess])
        ax.set_aspect('equal') 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    

# python3 wombatProcessObs.py 2012-07-12T16:00 2012-07-13T05:00 HI1A rdiffHI
# python3 wombatProcessObs.py 2012-07-13T20:00 2012-07-14T12:00 HI2A rdiffHI
theFile = 'wbPickles/WBGUI_temp.pkl'

# Open the pickle
with open(theFile, 'rb') as file:
    bkgData = pickle.load(file)

tidx = 6    
obs  ='HI1A'
satDict = bkgData['satStuff'][obs][0][tidx]
imMap = bkgData['proImMaps'][obs][0][tidx]
showMap = bkgData['scaledIms'][obs][0][tidx][0]
massMap = bkgData['massIms'][obs][tidx]

'''fig = plt.figure()
#plt.imshow(showMap, vmin=63, vmax=128, cmap='gist_heat')
plt.imshow(massMap,  cmap='gist_heat', vmin=-1e9, vmax=1e9)
plt.show()

print (sd)'''

awf = wf.wireframe('GCS')
awf.params = [27, 25, -13.4, 78.0, 55.0, 0.48]
#awf.params = [50, 45, 0, 0., 50.0, 0.25]
awf.getPoints()
    
mass2dens(imMap, satDict, awf, massMap, doInner=True)    
