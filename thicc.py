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



def anOldDistaster():
    theFile = 'wbPickles/WBGUI_temp.pkl'

    # Open the pickle
    with open(theFile, 'rb') as file:
        bkgData = pickle.load(file)

    
    imMap = bkgData['proImMaps']['HI2A'][0][0]
    satDict = bkgData['satStuff']['HI2A'][0][0]
    coordM = imMap.pixel_to_world(imMap.data.shape[1]/2 * u.pix, imMap.data.shape[0]/2 * u.pix)
    alpha = coordM.Tx.rad
    coord0 = imMap.pixel_to_world(0 * u.pix, 0 * u.pix)
    coord1 = imMap.pixel_to_world(0 * u.pix, imMap.data.shape[0] * u.pix)
    coord2 = imMap.pixel_to_world(imMap.data.shape[1] * u.pix, 0 * u.pix)
    coord3 = imMap.pixel_to_world(imMap.data.shape[1] * u.pix, imMap.data.shape[0] * u.pix)


    angs = [coord0.Tx.rad, coord1.Tx.rad, coord2.Tx.rad, coord3.Tx.rad]
    wid = np.abs(0.5*(np.max(angs) - np.min(angs)) )  
    ra = 0.5*(np.max(angs) + np.min(angs))
    angs2 = [coord0.Ty.rad, coord1.Ty.rad, coord2.Ty.rad, coord3.Ty.rad]
    hei = np.abs(0.5*(np.max(angs2) - np.min(angs2)) )  
    dec = 0.5*(np.max(angs2) + np.min(angs2)) 
    lon = (satDict['POS'][1] % 360) * np.pi / 180
    rSat = satDict['POS'][2] / 1.5e11
    FoVlen = 1.5 * rSat

    # Playing with skycoords
    myCoord = coordM
    ell = np.sqrt(myCoord.Tx.rad**2 + myCoord.Ty.rad**2)
    d = np.abs(rSat * np.cos(ell))
    hpc = SkyCoord(Tx=myCoord.Tx, Ty=myCoord.Ty, distance=d*u.au, frame= myCoord.frame)
    ston = hpc.transform_to(frames.HeliographicStonyhurst)


    # Make edges of FoV in spacecraft frame
    pt1 = np.array([-FoVlen * np.cos(wid) * np.sin(hei), -FoVlen * np.sin(wid) * np.sin(hei), FoVlen * np.sin(hei)])
    pt2 = np.array([-FoVlen * np.cos(wid) * np.sin(hei), FoVlen * np.sin(wid) * np.sin(hei), FoVlen * np.sin(hei)])
    pt3 = np.array([-FoVlen * np.cos(wid) * np.sin(hei), -FoVlen * np.sin(wid) * np.sin(hei),-FoVlen * np.sin(hei)])
    pt4 = np.array([-FoVlen * np.cos(wid) * np.sin(hei), FoVlen * np.sin(wid) * np.sin(hei), -FoVlen * np.sin(hei)])
    ptM = np.array([-FoVlen, 0, 0])

    sc  = [0, 0, 0]
    pts = [pt1, pt2, pt3, pt4, ptM]


    # Rotate FoV by dec
    for i in range(len(pts)):
        pt = pts[i]
        newx =  np.cos(dec)*pt[0] + np.sin(dec)*pt[2]
        newz = -np.sin(dec)*pt[0] + np.cos(dec)*pt[2]
        pts[i][0] = newx
        pts[i][2] = newz

    # Rotate FoV by -ra
    for i in range(len(pts)):
        pt = pts[i]
        newx =  np.cos(-ra)*pt[0] - np.sin(-ra)*pt[1]
        newy =  np.sin(-ra)*pt[0] + np.cos(-ra)*pt[1]
        pts[i][0] = newx
        pts[i][1] = newy

    # Move sat to dist
    for i in range(len(pts)):
        pts[i][0] = pts[i][0]+ rSat
    sc[0] =  sc[0] + rSat


    # Adjust FoV lens to Thomson sphere
    # Get elon angs
    sc = np.array(sc)
    pts2 = []
    for i in range(len(pts)):
        v1 = -sc
        v1 = v1 / np.sqrt(np.sum(v1**2))
        v2 = pts[i] - sc
        v2 = v2 / np.sqrt(np.sum(v2**2))
        ell = np.abs(np.arccos(np.dot(v1, v2)))
        l = np.abs(rSat * np.cos(ell))
        pts2.append(sc + l * v2)
    
    # Rotate everyone to satLon
    for i in range(len(pts)):
        pt = pts[i]
        newx =  np.cos(lon)*pt[0] - np.sin(lon)*pt[1]
        newy =  np.sin(lon)*pt[0] + np.cos(lon)*pt[1]
        pts[i][0] = newx
        pts[i][1] = newy
        pt2 = pts2[i]
        newx2 =  np.cos(lon)*pt2[0] - np.sin(lon)*pt2[1]
        newy2 =  np.sin(lon)*pt2[0] + np.cos(lon)*pt2[1]
        pts2[i][0] = newx2
        pts2[i][1] = newy2
    scR = np.zeros(3)
    scR[0] = np.cos(lon)*sc[0] - np.sin(lon)*sc[1]
    scR[1] = np.sin(lon)*sc[0] + np.cos(lon)*sc[1]
    scR[2] = sc[2]
    sc = scR 

    TSxyz = np.array([ston.cartesian.x.to_value(), ston.cartesian.y.to_value(), ston.cartesian.z.to_value()])
    print (np.dot(sc-TSxyz, -TSxyz))

    ax = plt.figure().add_subplot(projection='3d')
    for i in range(len(pts)):
        ax.plot([sc[0], pts[i][0]], [sc[1], pts[i][1]], [sc[2],pts[i][2]], 'k--')
        ax.plot(pts2[i][0],pts2[i][1],pts2[i][2], 'ko')
    ax.scatter(0,0,0, c='y')
    ax.scatter(sc[0], sc[1], sc[2], c='b')
    ax.scatter(ston.cartesian.x.to_value(), ston.cartesian.y.to_value(), ston.cartesian.z.to_value(), 'g')
    ax.set_xlabel('x')
    plt.show()


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
    

def mass2dens(myMap, satDict, aWF, toShow=None):
    
    # |-----------------------------------------|
    # Get functions for converting map pixel to 3D cartesian coords
    # where the FoV is in the yz plane at x=0
    # |-----------------------------------------|
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
    FOVfunc = [FOV2x, FOV2y, FOV2z] # pixels to FOVcart
    
    # For plotting
    '''fake_px = np.unique(np.array(pixx))
    fake_py = np.unique(np.array(pixy))
    fpxx, fpyy = np.meshgrid(fake_px, fake_py)
    f_fovx = FOV2x((fpxx, fpyy))
    f_fovy = FOV2y((fpxx, fpyy))
    f_fovz = FOV2z((fpxx, fpyy))'''
    
    if False:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(f_fovx, f_fovy, f_fovz, c=fpyy)
        plt.show()
        
    if False:
        fig = plt.figure()
        plt.imshow(f_fovz, extent=[fake_px[0], fake_px[-1], fake_py[0], fake_py[-1]], vmin=-30, vmax=30, origin='lower')
        plt.show()
    
    #|-------------------------------------------------|
    #|--- Get the wireframe width perp to FoV plane ---|
    #|-------------------------------------------------|
    awf.gPoints = [i * 10 for i in awf.gPoints]
    awf.getPoints()
    wfPts = wf2CartFoV(myMap, aWF)
    wfPtsT = np.transpose(np.array(wfPts))
    wids, midx, FoV, nGridY = getWidth(wfPtsT)
    if False:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(awf.points[::10,0], awf.points[::10,1], awf.points[::10,2])
        ax.scatter(wfPts[0][::10], wfPts[1][::10], wfPts[2][::10], c='m')
        plt.show()
        
    if False:
        fig = plt.figure()
        plt.imshow(wids, extent=[FoV[0][0], FoV[0][1], FoV[1][0], FoV[1][1]], vmin=0, vmax=30, origin='lower')
        plt.show()

    awf.getPoints(inside=True)
    wfPtsI = wf2CartFoV(myMap, aWF)
    wfPtsI = np.transpose(np.array(wfPtsI))
    widsI, midxI, FoV, nGridY = getWidth(wfPtsI, FoV=FoV, nGridY=nGridY)
    #awf.getPoints()

    dy, ygs, yms, nGridZ, zgs, zms = createGrid(FoV, nGridY)
    wid = wids#-widsI
    mask = wid > 0
    wid_smooth = ndimage.gaussian_filter(wid, sigma=2.0, order=0) 

    # indexing of func is y,z    
    widFunc = RegularGridInterpolator((yms, zms), np.transpose(wid_smooth), method='linear', bounds_error=False, fill_value=0)

    
    #|-----------------------------|
    #|--- Get width on FoV Grid ---|
    #|-----------------------------|
    # Start with grid of pixels
    downSize = 8
    fake_px = np.array(range(myMap.data.shape[1])[::downSize])
    fake_py = np.array(range(myMap.data.shape[0])[::downSize])
    
    fake_px = fake_px[np.where(fake_px <= maxX)[0]]
    fake_py = fake_py[np.where(fake_py <= maxY)[0]]
    
    fpxx, fpyy = np.meshgrid(fake_px, fake_py)
    f_fovx = FOV2x((fpxx, fpyy))
    f_fovy = FOV2y((fpxx, fpyy))
    f_fovz = FOV2z((fpxx, fpyy))
    widsInt = widFunc((f_fovy, f_fovz))
    if False:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(f_fovx, f_fovy, f_fovz, c=widsInt)
        plt.show()
    if False:
        fig = plt.figure()
        plt.imshow(widsInt, extent=[fake_px[0], fake_px[-1], fake_py[0], fake_py[-1]], vmin=0, vmax=30, origin='lower')
        plt.show()
        
    
    #|--------------------------|
    #|--- Pull the mass data ---|
    #|--------------------------|
    # Might want to take avg of nearby pix but TVD
    subMass = toShow[fpyy, fpxx]
    
    # Get the width of each grid cell (in solar radii)
    # Could make this smoother, or at least better at edges, TBD
    dys = np.zeros(subMass.shape)
    dys[:,1:] = f_fovy[:,1:] - f_fovy[:,:-1]
    dys[:,0] = dys[:,1] 
    dzs = np.zeros(subMass.shape)
    dzs[1:,:] = f_fovz[1:,:] - f_fovz[:-1,:]
    dzs[0,:] = dzs[1,:]
    cellArea = dys * dzs
    
    notZero = np.where(widsInt != 0)
    dens = np.zeros(subMass.shape)
    dens[notZero] = subMass[notZero] / widsInt[notZero] / cellArea[notZero]
    
    if False:
       fig = plt.figure()
       vval = 1e9
       plt.imshow(subMass, extent=[fake_px[0], fake_px[-1], fake_py[0], fake_py[-1]], vmin=-vval, vmax=vval, origin='lower')
       plt.show()
    if False:
        vval = 1e9
        ax = plt.figure().add_subplot(projection='3d')
        #ax.scatter(f_fovx[notZero], f_fovy[notZero], f_fovz[notZero], c=subMass[notZero],vmin=-vval, vmax=vval )
        ax.scatter(f_fovx, f_fovy, f_fovz, c=subMass,vmin=-vval, vmax=vval )
        ax.scatter(wfPts[0][::10], wfPts[1][::10], wfPts[2][::10], c='m')
        ax.set_xlim([-60,60])
        ax.set_ylim([-60,60])
        ax.set_zlim([-60,60])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show() 
    
    #|---------------------------|
    #|--- Expand in the x dim ---|
    #|---------------------------|

    
    ynot0    = f_fovy[notZero]
    znot0    = f_fovz[notZero]
    xnot0    = znot0 * 0
    wnot0    = widsInt[notZero]
    dnot0    = dens[notZero]
    
    dx = np.mean(dys) # should be sufficient
    nptsx = wnot0 / dx / 2 
    nptsx = nptsx.astype(int)
    
    allpts = [np.array([]), np.array([]), np.array([]), np.array([])]
    for i in range(len(ynot0)):
    #for i in range(4):
        myxs = None
        if nptsx[i] > 0:
            myxs = dx *(np.arange(-nptsx[i], nptsx[i]+1))
        elif wnot0[i] > 0.75*dx:
            myxs = np.zeros(1)
        if type(myxs) != type(None):
            myys  = ynot0[i] * np.ones(2*nptsx[i] + 1)
            myzs  = znot0[i] * np.ones(2*nptsx[i] + 1)
            mydens = dnot0[i] * np.ones(2*nptsx[i] + 1)
            allpts[0] = np.concatenate((allpts[0], myxs))
            allpts[1] = np.concatenate((allpts[1], myys))
            allpts[2] = np.concatenate((allpts[2], myzs))
            allpts[3] = np.concatenate((allpts[3], mydens))
            
    # polish density    
    allpts[3] = allpts[3] / (6.957e10 **3 ) # convert to g/cm^3   
    #meand, maxd = np.mean(allpts[3]), np.max(allpts[3]) 
    allpts[3][np.where(allpts[3] <= 0)] = np.min(np.abs(allpts[3]))
    logd = np.log10(allpts[3])
    #logv1, logv2 = np.log10(meand + deld), np.log10(meand - deld)
    
    if True:
        vval = 1e9 /dx**3
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(allpts[0], allpts[1], allpts[2], c=logd)
        ax.set_aspect('equal') 
        plt.show()
    
    print (sd)
    cs ='m'
    # In toShow need to index by pix xy but then compare with wid (yz) at that pix
    if type(toShow) != type(None):
        cs = toShow[pixxx, pixyy]
        '''cs = []
        for i in range(len(pixx)):
            cs.append(toShow[pixy[i], pixx[i]])'''
    cs = wid
    cs[np.transpose(isZero)] = 0
    cs[np.transpose(~isZero)] = widGrid[~isZero]#cs[np.transpose(~isZero)] / widGrid[~isZero]
    
    
    
    #fig = plt.figure()
    #plt.contourf(pixxx,pixyy, widGrid)
    #plt.imshow(pixyy,pixxx,cs, vmin=0, vmax=256)
    #plt.imshow(toShow, vmin=63, vmax=128)
    #plt.imshow(toShow, vmin=-1e9, vmax=1e9, origin='lower')
    #plt.show()

    ax = plt.figure().add_subplot(projection='3d')
    #ax.scatter(ptsOut[0],ptsOut[1],ptsOut[2], c=cs, vmin=0, vmax=1e8)
    ax.scatter(cartxx, cartyy, cartzz, c=widGrid)
    
    #ax.scatter(cartxx, cartyy, cartzz, c=widGrid)
    
    ax.scatter(wfPtsT[::10,0],wfPtsT[::10,1],wfPtsT[::10,2], c='gray')

    ax.set_xlim([-60,60])
    ax.set_ylim([-60,60])
    ax.set_zlim([-60,60])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    #print (satDict)

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
#awf.params = [27, 25, -13.4, 78.0, 55.0, 0.48]
awf.params = [30, 45, -13.4, 90., 55.0, 0.48]
awf.getPoints()
    
mass2dens(imMap, satDict, awf, toShow=massMap)    

if False:
    awf = wf.wireframe('GCS')
    # Increase resolution of the wf
    awf.gPoints = [i * 10 for i in awf.gPoints]
    # Set some arb parameters
    awf.params[0] = 10
    awf.params[1] = 80
    awf.params[2] = 0
    awf.params[3] = 10
    awf.getPoints()
    wids, midx, FoV, nGridY = getWidth(awf.points)
    awf.getPoints(inside=True)
    widsI, midxI, FoV, nGridY = getWidth(awf.points, FoV=FoV, nGridY=nGridY)

    dy, ygs, yms, nGridZ, zgs, zms = createGrid(FoV, nGridY)
    wid = wids-widsI
    mask = wid > 0
    wid_smooth = ndimage.gaussian_filter(wid, sigma=2.0, order=0) 

    # indexing of func is y,z    
    interp_func = RegularGridInterpolator((yms, zms), np.transpose(wid_smooth), method='linear')
    
    if False:
            points = awf.points
            x = points[:,0]
            y = points[:,1]
            z = points[:,2]
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(x,y,z, c='y')
            #ax.scatter(np.zeros(len(gapPts[:,0])),gapPts[:,0],gapPts[:,1], c='r')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.show()
        
    if True:
        fig = plt.figure()
        YY, ZZ = np.meshgrid(yms, zms)
        plt.contourf(YY, ZZ, wid_smooth * mask)
        maskIt = midx != -9999
        plt.contourf(YY, ZZ, midx*maskIt)
        plt.show()
    
