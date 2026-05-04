"""
Density Inferred from Nice Grid Object (DINGO) Module

Set of functions to take a wombat pickle and recon log file and line(s)
of interest, convert the mass/pixel map into densities, and compute 
results in zero to three dimensions.
    
    0D - total mass
    1D - density profile both in place at the time of the remote obs
         and propagated to a specific target satellite 
    2d - contour map of density
    3d - interactive 3d scatter plot of densities

The script is run either by passing a series of arguments via the command
line or by passing a single dingo_config text file. For the command line 
the arguments are

    MAIN PARAMS: (required, in order)
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

    BONUS PARAMS: (not required, order doesn't matter as long as after main 3)
                    
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
                (e.g Bepi, STA, ...) and nothing is case sensitive. 

        *** Flags - must be included as shown below, all default to false if they
            are not included ***

        doinner: a flag to try and remove the space corresponding to an internal 
                 gap between the legs of a GCS/torus wireframe. This is only meant
                 to be used if the gap is obscured from the satellites PoV, the 
                 gap will automatically be included when it is observed. This is not
                 fully tested and discourage use for now.

        projoff: a flag to not include projection effects when converting masses to
                 densities. The code calculates the Billings factor for each pixel 
                 using the corresponding wireframe center at that point and corrects
                 the observed mass. This correction factor is capped at a max of 10
                 and warns about large plane-of-sky separations
                 (defaults to including projection)
        
        logplot: a flag to plot 2d contours in a log scale instead of linear



        *** Prefix tags - the following options all require using the listed prefix 
            with the # replaced by the desired value (e.g. expf1_0.4 would set  
            expf1 at 0.4) ***
           
        expf1_#: the first expansion factor which sets the amount of the expansion
                 in the nonradial direction. The value is the ratio of the physical
                 size at the time of impact (width in perp direction) compared to what
                 it would be with self-similar expansion
                 (Only used in 1D, defaults to 1)

        expf2_#: the second expansion factor which sets the rate of the radial
                 expansion speed relative to the radial propagation speed. The 
                 # should be a decimal value between 0 and 1 (and likely closer to 0)
                 (Only used in 1D, defaults to 0.1)

        densratio_#: the ratio between the inner and outer wireframe densities (n1/n2).
                     the densities vary from pixel to pixel but the ratio between the 
                     two remains the same in any overlapping regions
                     (defaults to 1.)

        vcme_#: the average interplanetary velocity of the CME. this allows conversion
                of the initial distances into in situ times which only should be taken
                as representative as DINGO does not include any IP evolution beyond expansion
                (Only used in 1D, defaults to 400)

        ds_#: an integer indicating how much to downselect the resolution from the input 
              mass maps. Running full resolution is fine for modes 0-2 but it will break
              3d scatter plots.
              (defaults to 8 for 3D mode, 1 for everything else)


Alternatively, one can just pass a dingo_config text file. This is a simple text file 
with two columns containing the nametag+':' (nametag from the above options) and the 
corresponding values. For example, a file could contain:
    logFile:	wbOutputs/WomBlog.txt
    ids:		5+6
    dim:		2d
    logplot:	True
                
The code will either save a plot (if saveName is given) or pop up a window with the
desired figure. In the case of a 1D figure, the a save name will also be used to save
the in place and in situ profile as text files (dingo_IP/IS_savename.dat)

"""


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
from wombatMass import elTheory 
from wcs_funs import fitshead2wcs, wcs_get_pixel, wcs_get_coord

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
np.seterr(divide='ignore', invalid='ignore')

global picType
picType = '.png' # png or pdf, gets overwritten if set in input 

# |--------------------------------------------------|
# |--------------------------------------------------|
# |--- Functions to convert mass image to density ---|
# |--------------------------------------------------|
# |--------------------------------------------------|

# |---------------------------|
# |--- Get background grid ---|
# |---------------------------|
def createGrid(FoV, nGridY):
    '''
    Basic helper to convert FOV and y resolution to a grid. The
    coordinates are in yz (horizontal/vertical) as we treat x 
    as the direction perpendicular to the FoV. The same resolution
    is used in the y and z directions
    
    Inputs:
        FoV: an array of the FoV distances as [miny, maxy, minz, maxz]
             this code doesn't care about the units and will return 
             values in the same form as given but in practice is used
             in solar radii
    
        nGridY: the number of grid points to use in the y direction
    
    Outputs:
        dy: the resolution in pixels per grid cell. this is the same 
            in the y and z directions
        
        ygs: the distances of the edge of the grid cells (length nGridY+1)
    
        yms: the distances of the middle of the grid cells (length nGridY)
    
        nGridZ: the number of grid cells in the z direction, which is calc from 
                the range in z and dy
    
        zgs: the distances of the edge of the grid cells (length nGridZ+1)
    
        zms: the distances of the middle of the grid cells (length nGridZ)
           
    '''
    miny, maxy = FoV[0][0], FoV[0][1]
    minz, maxz = FoV[1][0], FoV[1][1]
    dy = (maxy - miny) / nGridY
    # Points on the grid
    ygs = np.array([miny + i*dy for i in range(nGridY+1)])
    # Midpoints
    yms = np.array([miny + (0.5+i)*dy for i in range(nGridY)])
    # Same for z, but need to get nGridZ
    nGridZ = int((maxz - minz) / dy)
    zgs = np.array([minz + i * dy  for i in range(nGridZ+1)])
    zms = np.array([minz + (0.5+i)*dy for i in range(nGridZ)])
    return dy, ygs, yms, nGridZ, zgs, zms

# |------------------------------------|
# |--- Convert points to widths map ---|
# |------------------------------------|
def getWidthNew(points, FoVfs, maxPix, satFOVxyz, flatLim=5, nGridY=100):
    '''
    Helper function to take a set of points representing some 3d
    shape and determine the width perpendicular to the PoS. This will
    account for curved PoS. The returning array is done for the
    uniformly spaced pixels, which will not be uniform in physical
    space for the wide imagers. This code is written with y being
    the horizontal pixel direction and z being the vertical to be
    analogous to when they're converted to proper cartesian
    
    Inputs:
        points: a [nPoints, 3] array with the xyz values for each
                wireframe point. the coords are such that
                    - x is the line of sight direction (at FOV center)
                    - y is the horizontal/~longitude direction
                    - z is the vertical/~latitude direction
                this is meant to work with the transposed results of
                wf2CartFoV applied to wf.points
    
        FoVfs:  the functions that convert from pixels to FoV Cart in the
                form [fovfx, fovfy, fovfz] where it expects f((pixy, pixz))
    
        maxPix: the maximum pixel value as [maxy, maxz]
    
        satFOVxyz: the location of the satellite in FoV cartesian. this 
                   should be [dThom, ~0, ~0] where dThom is the distance to
                   the Thomson surface at the center of the FoV (given as an
                   output by map2CartFoV)
    
    Optional Inputs:
        flatLim: the angle in degrees used as the limit on allowing a flat 
                 PoS calculation (above treated as curved)
                 defaults to 5 deg which lets COR2 be flat but HIs curved
        
        nGridY: the number of grid points in the y direction. this is the
                resolution for the output array which dingo intends to pass
                to an interpolator so it doesn't need to be as high of res
                as the mass maps.
                (defaults to 100)
    
    Outputs:
        wids: a 2d array with the width in the x direction. Grid cells that
              don't have any corresponding points are set to zero
        
        midx: a 2d array with the center x value for all points in that grid
              cell (e.g a segment symmetric about the yz plane would be 0).
              Grid cells without corresponding points are set to the median
              value from the other points so that when dingo passes this to 
              an interpolator the edges don't have weird artifacts
        
        mask: a 2d binary array with 1 for grid cells that have an non zero width  
              and 0 for grid cells with no corresponding points
        
        FoV:  the packaged FOV in pixels based on maxPix. The format is 
              [miny, maxy, minz, maxz]
        
        nGridY: the number of grid points, either the same as the provided input
                or the default value (enables reproducing createGrid)

    '''
    # Make a grid in pixel coords
    # This can be lower res (e.g. 100 pix across) bc we 
    # pass these results to an interpolator
    dy, ygs, yms, nGridZ, zgs, zms = createGrid([[0, maxPix[0]], [0, maxPix[1]]], nGridX)
    yys, zzs = np.meshgrid(yms, zms)
    FoV = [[np.min(yys), np.max(yys)], [np.min(zzs), np.max(zzs)]]
    
    # Convert everyone to FoV cart coords with sat at 0!
    # FoV plane
    FOVx = FoVfs[0]((yys, zzs)) - satFOVxyz[0]
    FOVy = FoVfs[1]((yys, zzs))
    FOVz = FoVfs[2]((yys, zzs))
    eqR  = np.sqrt(FOVx**2 + FOVy**2) 
    FoVlon = np.arctan(FOVy /FOVx) * 180/3.14
    FoVlat = np.arctan2(FOVz, eqR)* 180/3.14
    # WF points
    WFpts = np.transpose(points)
    WFpts[0] -= satFOVxyz[0]
    # Sat itself
    satPos = np.copy(satFOVxyz)
    satPos[0] -= satFOVxyz[0]
    
    # |--- Check if small enough angle to calc width using flat PoS ---|
    canFlat = False
    if isinstance(flatLim, (int, float)): 
        if (np.max([np.max(np.abs(FoVlon)), np.max(np.abs(FoVlat))])) < flatLim:
            canFlat = True
    
    pad = 1.4 # distance from midpoint to check (in dy)   
    # Set up empty arrays
    wids = np.zeros(FOVx.shape)
    xcs = np.zeros(FOVx.shape) -9999
    mask = np.zeros(FOVx.shape)
    
    # |--- Loop and match pixels to WF points ---|
    ncount = FOVx.shape[1]
    for i in range(FOVx.shape[1]):
        # A little slow so give progress
        print ('Calc widths', i+1, '/', ncount)
        for j in range(FOVx.shape[0]):
            # For each pixel we want to rotate it so the LoS is
            # parallel to x-axis so the WF width is just the range
            # in x for a given yz. For flat assume things are good
            # enough (LoS only small angle off). For curved, rotate
            # each pix to x axis using the lon/lat in the s/c centered
            # coordinate. Resulting wids and center vals are relative
            # to the PoS for both calcs.
            if canFlat:
                # If can do flat reassign things to names at
                # the end of curved calc
                temp2 = [FOVx, FOVy, FOVz]
                temp22 = WFpts
                temp2s = satPos
            else:
                # Rot about z by lon 
                temp1 = wf.rotz([FOVx, FOVy, FOVz], -FoVlon[j,i])
                temp11 = wf.rotz(WFpts, -FoVlon[j,i])
                temp1s = wf.rotz(satPos, -FoVlon[j,i])
    
                # Rot about y by lat 
                temp2 = wf.roty(temp1, -FoVlat[j,i])
                temp22 = wf.roty(temp11, -FoVlat[j,i])
                temp2s = wf.roty(temp1s, -FoVlat[j,i])
            
            # Find who is close to y=z=0
            if i < 2:
                yspace = np.abs(2 * (temp2[1][j,1] - temp2[1][j,0]))
            elif i > (ncount -3):
                yspace = np.abs(2 * (temp2[1][j,-1] - temp2[1][j,-2]))
            else:
                yspace = np.abs(temp2[1][j,i-1] - temp2[1][j,i+1])
            
            # the temp2 values should be near 0 but some rounding errors
            myidx = np.where((np.abs(temp22[1]-temp2[1][j,i])< pad*yspace) & (np.abs(temp22[2]-temp2[2][j,i])< pad*yspace))
            if len(myidx[0]) > 0:
                minx, maxx = np.min(temp22[0][myidx]), np.max(temp22[0][myidx])
                mywid = maxx - minx
                myxc  = 0.5 * (maxx + minx)
                mask[j,i] = 1
                wids[j,i] = mywid
                xcs[j,i]  = myxc - temp2[0][j,i]
            
    #FoVlon[j,i] = -999       
            
    #fig = plt.figure()
    #plt.imshow(wids, origin='lower')
    #plt.show()    
    '''fig = plt.figure(figsize=(8, 5), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(temp2[0], temp2[1], temp2[2], c=FoVlon)
    ax.scatter(temp2s[0], temp2s[1], temp2s[2], c='k')
    im = ax.scatter(temp22[0][::60], temp22[1][::60], temp22[2][::60])
    ax.set_aspect('equal') 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    print (sd)'''
    
    # |------------------------------------------|
    # |--- Clean up grid cells with no points ---|
    # |------------------------------------------|
    # need to fill in the outer -9999 region of xc so interp is happy
    for i in range(xcs.shape[1]):
        notOut = np.where(xcs[:,i] !=-9999)[0]
        if len(notOut) >= 2:
            xcs[:notOut[0], i]  = xcs[notOut[0],i]
            xcs[notOut[-1]:, i] = xcs[notOut[-1],i]
    for i in range(xcs.shape[0]):
        notOut = np.where(xcs[i,:] !=-9999)[0]
        if len(notOut) >= 2:
            xcs[i,:notOut[0]]  = xcs[i, notOut[0]]
            xcs[i,notOut[-1]:] = xcs[i, notOut[-1]]
    
    # fill in the remaining -9999 spots (inner hole) with the med midx
    medxcs = np.median(xcs[np.where(mask == 1)])
    xcs[np.where(xcs == -9999)] = medxcs

    return wids, xcs, mask, FoV, nGridY

def getWidth(points, FoV=None, nGridY=100):
    '''
    Helper function to take a set of points representing some 3d
    shape and determine the width in the x direction. This code
    is not sensitive to the units of distances but points and 
    FoV should be consistent and in practice solar radii are used
    
    Inputs:
        points: a [nPoints, 3] array with the xyz values for each
                point. the coords are such that
                    - x is the line of sight direction (at FOV center)
                    - y is the horizontal/~longitude direction
                    - z is the vertical/~latitude direction
    
    Optional Inputs:
        FoV: the field of view to use for the output 2d arrays. If
             not provided it will be calculated from the points. This
             allows for a consistent FoV for multiple calls to getWidth
             for different sets of points
             The format is [miny, maxy, minz, maxz]
        
        nGridY: the number of grid points in the y direction. this is the
                resolution for the output array which dingo intends to pass
                to an interpolator so it doesn't need to be as high of res
                as the mass maps.
                (defaults to 100)
    
    Outputs:
        wids: a 2d array with the width in the x direction. Grid cells that
              don't have any corresponding points are set to zero
        
        midx: a 2d array with the center x value for all points in that grid
              cell (e.g a segment symmetric about the yz plane would be 0).
              Grid cells without corresponding points are set to the median
              value from the other points so that when dingo passes this to 
              an interpolator the edges don't have weird artifacts
        
        mask: a 2d binary array with 1 for grid cells that have an non zero width  
              and 0 for grid cells with no corresponding points
        
        FoV:  either the calculated FoV or simply returning the array given as 
              input. The format is [miny, maxy, minz, maxz]
        
        nGridY: the number of grid points, either the same as the provided input
                or the default value

    '''

    # |--- Unpackage points ---|
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    
    # |---------------------|
    # |--- Determine FoV ---|
    # |---------------------|
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
    
    # |-------------------|
    # |--- Set up grid ---|
    # |-------------------|
    dy, ygs, yms, nGridZ, zgs, zms = createGrid(FoV, nGridY)
        
    wids = np.zeros([nGridZ, nGridY])
    midx = np.zeros([nGridZ, nGridY]) - 9999
    mask = np.zeros([nGridZ, nGridY]) 

    # |----------------------------------|
    # |--- Match points to grid cells ---|
    # |----------------------------------|
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
                        mask[j,i] = 1
                  
    # |------------------------------------------|
    # |--- Clean up grid cells with no points ---|
    # |------------------------------------------|
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
    
    # fill in the remaining -9999 spots (inner hole) with the med midx
    medmidx = np.median(midx[np.where(mask == 1)])
    midx[np.where(midx == -9999)] = medmidx

    return wids, midx, mask, FoV, nGridY

# |---------------------------------------------|
# |--- Stonyhurst Cartesian to FoV Cartesian ---|
# |---------------------------------------------|
def StonyCart2CartFoV(pts, satLat, satLon, roll):
    ''' 
    Helper function to convert points from Stonyhurst Cartesian
    frame into the FoV Cartesian plane where the FoV is in the
    yz plane and the spacecraft is on the x-axis. The FoV is treated
    as flat and perpendicular to the line connecting the center of
    the FoV to the satellite at a distance corresponding to the 
    Thomson sphere for that direction.
    
    Inputs:
        pts: a [npts, 3] array of points in stony cart xyz. the 
             first two points should be the spacecraft location and 
             the FoV center. the remaining points are the points of
             interest to convert into FoV cartesian coordinates
    
        satLat: the latitude of the satellite in degrees
    
        satLon: the stonyhurst longitude of the satellite in degrees
    
        roll:   the roll angle of the satellite in degrees
    
    Outputs:
        pts: a [3, npts-2] array of points in FoV Cartesian coordinates.
             The first two points from the input pts array (sat, FoV center)
             are removed before returning the results
    '''    
    # Rotate so sc at in xz plane
    pts = wf.rotz(pts,-satLon)
    
    
    # Rotate so sc at z = 0 -> on x-axis
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
    '''if True:
        fig = plt.figure(figsize=(8, 5), layout='constrained')
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(pts[0], pts[1], pts[2])
        ax.set_aspect('equal') 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        print (sd)'''
    
    # Take off the first two points we added
    xOut = pts[0][2:]
    yOut = pts[1][2:]
    zOut = pts[2][2:]
    
    satLoc = [pts[0][0], pts[1][0], pts[2][0],]
    
    return [xOut, yOut, zOut], satLoc
    
    
# |-----------------------------------------|
# |--- Convert full map to FoV Cartesian ---|
# |-----------------------------------------|
def map2CartFoV(myMap, points, pixCent=None):
    ''' 
    Helper function using the header info from a map to convert
    a set of points (in pixels) into FoV Cartesian coordinates 
    where the FoV is in the yz plane and the spacecraft is on the 
    x-axis. The FoV is treated as flat and perpendicular to the line 
    connecting the center of the FoV to the satellite at a distance 
    corresponding to the Thomson sphere for that direction.
    
    Inputs:
        myMap: an astropy map used to define the field of view of the 
               observation. Only the header info and size are used, not
               the data itself
        
        points: a list of pixels for which to calculate the FoV Cartesian
                points. The form should be [[pixel_xs], [pixel_ys]]
    
    Optional Inputs:
        pixCent: the center of the FoV which is used to calculate the 
                 plane of the sky. The format should be [pixx, pixy].
                 If not provided the center of the map will be determined
                 from the size.
    
    Outputs:
        res: the points converted into FoV Cartesian (in Rs) as a 
             3 x nPoints array ([xs, ys, zs])
     
    '''
    
    #|------------------------|
    #|--- Get sat position ---|
    #|------------------------|
    # Get satellite position from the map
    satLonD = myMap.observer_coordinate.lon.degree
    satLatD = myMap.observer_coordinate.lat.degree
    satR = myMap.observer_coordinate.radius.au 
    satLonR = satLonD * np.pi / 180.
    satLatR = satLatD * np.pi / 180.
    
    
    # Get vector from sun to sat
    # (just the cartesian sat loc)
    satxyz = np.array([np.cos(satLatR)*np.cos(satLonR), np.cos(satLatR)*np.sin(satLonR), np.sin(satLatR)]) * satR
    
    #|------------------------|
    #|--- Get sat pointing ---|
    #|------------------------|
    # Get the direction the FoV is pointing in Stony frame
    # Use pix cent if provided, otherwise assume middle
    if type(pixCent) == type(None):
        pixCent = [myMap.data.shape[1]/2, myMap.data.shape[0]/2] # pix is xy but shape is yx

    # Get the heliprojective coord of the pixel
    coordM = myMap.pixel_to_world(pixCent[0] * u.pix, pixCent[1] * u.pix)
    # Get elongation angle -> distance to Thomson Sphere
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
    
    #|-------------------------|
    #|--- Make pts packages ---|
    #|-------------------------|
    # Start coord arrays with sat loc and FoV cent and Sun
    xs = [satxyz[0]*215, TSxyz[0]*215]
    ys = [satxyz[1]*215, TSxyz[1]*215]
    zs = [satxyz[2]*215, TSxyz[2]*215]
    
    #|---------------------------------|
    #|--- Convert pix to Stony cart ---|
    #|---------------------------------|
    # Get the OG coords for each pixels in points
    nPoints = len(points[0])
    cs = []
    for i in range(nPoints):
        # Convert pixel to helioproj
        coord0 = myMap.pixel_to_world(points[0][i] * u.pix, points[1][i] * u.pix)
        # Get Thomson distance based on elong angle
        eps = np.sqrt(coord0.Tx.rad**2 + coord0.Ty.rad**2)
        dThom = np.abs(satR * np.cos(eps))
        cs.append(dThom)
        
        hpc0 = SkyCoord(Tx=coord0.Tx, Ty=coord0.Ty, distance=dThom*u.au, frame= coord0.frame)
        ston0 = hpc0.transform_to(frames.HeliographicStonyhurst)
        TSxyz0 = np.array([ston0.cartesian.x.to_value(), ston0.cartesian.y.to_value(), ston0.cartesian.z.to_value()])
        # Old code from forcing flat PoS
        '''LoS0 = TSxyz0 - satxyz    
        uTSxyz0 = TSxyz0 / np.linalg.norm(TSxyz)
        uLoS0 = LoS0 / np.linalg.norm(LoS0)
        dotIt = np.dot(uLoS0, uLoS)
        if dotIt > 1: dotIt = 1.
        elif dotIt < -1: dotIt = -1.
        ang = np.arccos(dotIt)
        newL = dM / np.cos(np.abs(ang))
        TSxyz0 = satxyz + newL*uLoS0'''
        
        xs.append(TSxyz0[0]*215)
        ys.append(TSxyz0[1]*215)
        zs.append(TSxyz0[2]*215)
    pts = np.array([xs, ys, zs])      
    
    #|---------------------------|
    #|--- Secret testing plot ---|
    #|---------------------------|
    '''fig = plt.figure(figsize=(8, 5), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(pts[0][2:], pts[1][2:], pts[2][2:], c=cs)
    ax.set_aspect('equal') 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()'''
    
    
    #|----------------|
    #|--- Get roll ---|
    #|----------------|
    if 'crota' in myMap.meta:
        rollIt = myMap.meta['crota']
    elif 'sc_roll' in myMap.meta:
        rollIt = myMap.meta['sc_roll']
    else:
        print ('Neither crota or sc_roll in map metadata. Assuming zero roll')
        rollIt = 0

    #|------------------------------------|
    #|--- Pass to cart 2 cart function ---|
    #|------------------------------------|
    res, satLoc = StonyCart2CartFoV(pts, satLatD, satLonD, rollIt)
    
    # Pass the normal stony pts
    xOut = pts[0][2:]
    yOut = pts[1][2:]
    zOut = pts[2][2:]
    res2 = [xOut, yOut, zOut]
    
    '''fig = plt.figure(figsize=(8, 5), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(res[0], res[1], res[2], c=cs)
    ax.scatter(satLoc[0], satLoc[1], satLoc[2])
    ax.set_aspect('equal') 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    print(sd)'''
    return res, res2, satLoc


# |-----------------------------------------|
# |--- Wireframe points to FoV Cartesian ---|
# |-----------------------------------------|
def wf2CartFoV(myMap, inPts, pixCent=None):
    ''' 
    Helper function using the header info from a map to convert
    points from a wireframe object into FoV Cartesian coordinates 
    where the FoV is in the yz plane and the spacecraft is on the 
    x-axis. The FoV is treated as flat and perpendicular to the line 
    connecting the center of the FoV to the satellite at a distance 
    corresponding to the Thomson sphere for that direction.
    
    Inputs:
        myMap: an astropy map used to define the field of view of the 
               observation. Only the header info and size are used, not
               the data itself
        
        inPts: a list of points in the form n x 3. this can be the results
               of aWF.points
    
    Optional Inputs:
        pixCent: the center of the FoV which is used to calculate the 
                 plane of the sky. The format should be [pixx, pixy].
                 If not provided the center of the map will be determined
                 from the size.
    
    Outputs:
        res: the wf points converted into FoV Cartesian (in Rs) as a 
             3 x nPoints array ([xs, ys, zs])
     
    '''
    
    #|------------------------|
    #|--- Get sat position ---|
    #|------------------------|
    satLonD = myMap.observer_coordinate.lon.degree
    satLatD = myMap.observer_coordinate.lat.degree
    satR = myMap.observer_coordinate.radius.au 
    satLonR = satLonD * np.pi / 180.
    satLatR = satLatD * np.pi / 180.
    
    
    #|------------------------|
    #|--- Get sat pointing ---|
    #|------------------------|
    # Get vector from sun to sat
    # (just the cartesian sat loc)
    satxyz = np.array([np.cos(satLatR)*np.cos(satLonR), np.cos(satLatR)*np.sin(satLonR), np.sin(satLatR)]) * satR
    
    # Get the direction the FoV is pointing in Stony frame
    # Use pix cent if provided, otherwise assume middle
    if type(pixCent) == type(None):
        pixCent = [myMap.data.shape[1]/2, myMap.data.shape[0]/2] # pix is xy but shape is yx

    # Get the heliprojective coord of the pixel
    coordM = myMap.pixel_to_world(pixCent[0] * u.pix, pixCent[1] * u.pix)
    # Get elongation angle -> distance to Thomson Sphere
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
    
    #|-------------------------|
    #|--- Make pts packages ---|
    #|-------------------------|
    # Start coord arrays with sat loc and FoV cent
    xs = [satxyz[0]*215, TSxyz[0]*215]
    ys = [satxyz[1]*215, TSxyz[1]*215]
    zs = [satxyz[2]*215, TSxyz[2]*215]
    
    #wfPoints = aWF.points
    xwf = inPts[:,0]
    ywf = inPts[:,1]
    zwf = inPts[:,2]
    
    x = [*xs, *xwf]
    y = [*ys, *ywf]
    z = [*zs, *zwf]
    
    pts = np.array([x, y, z])      
    
    #|----------------|
    #|--- Get roll ---|
    #|----------------|
    if 'crota' in myMap.meta:
        rollIt = myMap.meta['crota']
    elif 'sc_roll' in myMap.meta:
        rollIt = myMap.meta['sc_roll']
    else:
        print ('Neither crota or sc_roll in map metadata. Assuming zero roll')
        rollIt = 0
        
    #|------------------------------------|
    #|--- Pass to cart 2 cart function ---|
    #|------------------------------------|
    res, satLoc = StonyCart2CartFoV(pts, satLatD, satLonD, rollIt)
    
    return res
    

# |------------------------------------|
# |--- Main mass to density routine ---|
# |------------------------------------|
def mass2dens(myMap, satDict, awf, massMap, doInner=False, densRatio=1, downSelect=8, deproj=True):
    '''
    Fuction to take a mass image map, satellite dictionay, and wireframe object and
    determine the width perp to the plane of sky and convert integrated mass to density.
    A single wireframe or two can be passed (to represent a shock and ejecta). A constant
    density is assumed along the line of sight within each WF. The routine returns a set
    of useful arrays and other variables that the plotting scripts use to generate figures.
    
    Inputs:
        myMap:      a astropy/sunpy map defining the pointing. the data is not used
    
        satDict:    a wombat style satellite dictionary
    
        awf:        a wombat wireframe or two in an array [wf1, wf2] where wf1 is the 
                    primary wf (i.e. ejecta) and wf2 is an outer/surrounding shape (sheath)
    
        massMap:    an 2d array with the mass data in g/pixel. This should be the same FoV as
                    defined by myMap. the wombat save file just keeps the mass as an array
                    an not a proper map so we pass things separately
    
    Optional Inputs:
        doInner:    remove the gap region at the back for a GCS or torus wf. this is only for
                    an obscured gap in a fairly edge on case. Any visible gaps are automatically
                    removed by the normal routine
                    (defaults to false, not thoroughly tested and recommend not using yet)
    
        densRatio:  the ratio of the density between wf1 and wf2, which is treated as a 
                    constant value over all lines of sight. calculated as wf1/wf2 and 
                    ignored if only a single wf is passed
                    (defaults to 1)
    
        downSelect: an integer to downsample the output resolution. This defaults to 8, which is
                    an appropriate value if making a 3d scatter plot with the results but setting
                    it to 1 is fine for other modes
    
        deproj:     flag to deproject the masses accounting for the wf width perp to the
                    PoS via Billings instead of treating all pts as at Thomson sphere
    
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
    
    # |--------------------------------------|
    # |--- Decide single or multi WF mode ---|
    # |--------------------------------------|
    if (type(awf) == type(wf.wireframe(None))):
        multiMode = False 
    else:
        if len(awf) > 2:
            sys.exit('Thickness calc only capable of doing two wireframes')
        else:
            multiMode = True
            awf2 = awf[1]
            awf  = awf[0]            

            
    # Save downselect input to use as the final downselect
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
    ptsOut, ptsOutStony, satFOVxyz = map2CartFoV(myMap, [pixx, pixy])  # is [x,y,z] where each is same len as pixIn  
    
    # Get range            
    uniX = np.unique(np.array(pixx))
    uniY = np.unique(np.array(pixy))
    maxX = np.max(uniX)
    maxY = np.max(uniY)
    
    # Reshape FoV arrays
    FOVx = ptsOut[0].reshape([len(uniY), len(uniX)])
    FOVy = ptsOut[1].reshape([len(uniY), len(uniX)])
    FOVz = ptsOut[2].reshape([len(uniY), len(uniX)])  
        
    # Set interpolators to take (x, y) as input order but python array is [y,x]
    FOV2x = RegularGridInterpolator((uniX, uniY), np.transpose(FOVx), method='linear')
    FOV2y = RegularGridInterpolator((uniX, uniY), np.transpose(FOVy), method='linear')
    FOV2z = RegularGridInterpolator((uniX, uniY), np.transpose(FOVz), method='linear')
    
    #|--- Repeat process for normal Stony Cart ---|
    FOVxS = ptsOutStony[0].reshape([len(uniY), len(uniX)])
    FOVyS = ptsOutStony[1].reshape([len(uniY), len(uniX)])
    FOVzS = ptsOutStony[2].reshape([len(uniY), len(uniX)])
    FOV2xS = RegularGridInterpolator((uniX, uniY), np.transpose(FOVxS), method='linear')
    FOV2yS = RegularGridInterpolator((uniX, uniY), np.transpose(FOVyS), method='linear')
    FOV2zS = RegularGridInterpolator((uniX, uniY), np.transpose(FOVzS), method='linear')
    
    #|-------------------------------------------------|
    #|--- Get the wireframe width perp to FoV plane ---|
    #|-------------------------------------------------|
    # This sets up generators for the full WF shape using the pointing
    # of the satellite but not the specific bounds of the FoV
    
    # |--- Do the outer WF first ---|
    # Need to set FoV to larger region and outer should be bigger
    if multiMode:
        awf2.gPoints = [i * 10 for i in awf2.gPoints]
        awf2.getPoints()
        wfPts2 = wf2CartFoV(myMap, awf2.points)
        wfPts2T = np.transpose(np.array(wfPts2))
        
        wids2, midx2, mask2, FoV, nGridY = getWidth(wfPts2T)
        dy, ygs, yms, nGridZ, zgs, zms = createGrid(FoV, nGridY)
        wid_smooth2 = ndimage.gaussian_filter(wids2, sigma=2.0, order=0)
        
        # indexing of func is y,z    
        widFunc2  = RegularGridInterpolator((yms, zms), np.transpose(wid_smooth2), method='linear', bounds_error=False, fill_value=0)
        xcFunc2   = RegularGridInterpolator((yms, zms), np.transpose(midx2), method='linear', bounds_error=False, fill_value=0)
        maskFunc2 = RegularGridInterpolator((yms, zms), np.transpose(mask2), method='linear', bounds_error=False, fill_value=0)
        
    # |--- Do the inner/main WF ---|
    awf.gPoints = [i * 10 for i in awf.gPoints]
    awf.getPoints()
    wfPts = wf2CartFoV(myMap, awf.points)
    wfPtsT = np.transpose(np.array(wfPts))
    
    '''fig = plt.figure(figsize=(8, 5), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    rs = np.sqrt(np.array(pixx)**2 + np.array(pixy)**2)
    im = ax.scatter(ptsOut[0], ptsOut[1], ptsOut[2], c=rs, cmap='inferno')
    ax.scatter(wfPts[0][::10], wfPts[1][::10], wfPts[2][::10])
    ax.set_aspect('equal') 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    print (sd)'''
    # Check if we have an existing FoV
    if multiMode:
        wids, midx, mask, FoV, nGridY = getWidth(wfPtsT, FoV=FoV, nGridY=nGridY)
    # Otherwise grab the new one
    else:
        wids, midx, mask =getWidthNew(wfPtsT,[FOV2x, FOV2y, FOV2z],[maxX, maxY], satFOVxyz)
        
        fig = plt.figure()
        plt.imshow(mask, origin='lower')
        plt.show()
        print (sd)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        wids, midx, mask, FoV, nGridY = getWidth(wfPtsT)
        dy, ygs, yms, nGridZ, zgs, zms = createGrid(FoV, nGridY)
    wid_smooth = ndimage.gaussian_filter(wids, sigma=2.0, order=0)

    # indexing of func is y,z    
    widFunc  = RegularGridInterpolator((yms, zms), np.transpose(wid_smooth), method='linear', bounds_error=False, fill_value=0) 
    xcFunc   = RegularGridInterpolator((yms, zms), np.transpose(midx), method='linear', bounds_error=False, fill_value=0)
    maskFunc = RegularGridInterpolator((yms, zms), np.transpose(mask), method='linear', bounds_error=False, fill_value=0)
    
    # |--- Repeat process for the inside (if doing) ---|
    if doInner and (awf.WFtype in ['GCS', 'Torus']):
        awf.getPoints(inside=True)
        wfPtsI = wf2CartFoV(myMap, awf.points)
        wfPtsI = np.transpose(np.array(wfPtsI))
        widsI, midxI, maskI, FoV, nGridY = getWidth(wfPtsI, FoV=FoV, nGridY=nGridY)
        wid_smoothI = ndimage.gaussian_filter(widsI, sigma=2.0, order=0)

        # indexing of func is y,z    
        widFuncI = RegularGridInterpolator((yms, zms), np.transpose(wid_smoothI), method='linear', bounds_error=False, fill_value=0) 
        xcFuncI  = RegularGridInterpolator((yms, zms), np.transpose(midxI), method='linear', bounds_error=False, fill_value=0)
        maskFuncI = RegularGridInterpolator((yms, zms), np.transpose(maskI), method='linear', bounds_error=False, fill_value=0)
    
    # Plot of widths    
    '''fig = plt.figure()
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
    # Takes the generators and applys to the actual FoV
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
    maskIntnz = maskFunc((f_fovy, f_fovz))
    maskIntnz[np.where(maskIntnz < 0.99)] = 0
    widsIntnz = widsIntnz * maskIntnz
    xcIntnz = xcIntnz * maskIntnz
    

    #|--- Inner region of main wireframe ---|
    if doInner:
        widsIntnzI = widFuncI((f_fovy, f_fovz))
        xcIntnzI   = xcFuncI((f_fovy, f_fovz))
        widsIntnzI[np.where(widsIntnzI < dx*0.5)] = 0
        maskIntnzI = maskFuncI((f_fovy, f_fovz))
        maskIntnzI[np.where(maskIntnzI < 1)] = 0
        widsIntnzI = widsIntnzI * maskIntnzI
        xcIntnzI = xcIntnzI * maskIntnzI
    else:
        widsIntnzI, xcIntnzI, maskIntnzI = None, None, None

    #|--- Outer wireframe ---|
    if multiMode:
        widsIntnz2 = widFunc2((f_fovy, f_fovz))
        xcIntnz2   = xcFunc2((f_fovy, f_fovz))
        widsIntnz2[np.where(widsIntnz2 < dx*0.5)] = 0
        maskIntnz2 = maskFunc2((f_fovy, f_fovz))
        maskIntnz2[np.where(maskIntnz2 < 0.99)] = 0
        widsIntnz2 = widsIntnz2 * maskIntnz2
        xcIntnz2 = xcIntnz2 * maskIntnz2
    else:
        widsIntnz2, xcIntnz2, maskIntnz2 = None, None, None
    
    #|--- Package for outputs ---|
    widMap  = [widsIntnz, widsIntnzI, widsIntnz2]    
    xcMap   = [xcIntnz, xcIntnzI, xcIntnz2]   
    maskMap = [maskIntnz, maskIntnzI, maskIntnz2]    
    outFoV  = [minpx, maxpx, minpy, maxpy, downSize]

    #|--------------------------|
    #|--- Get deproj weights ---|
    #|--------------------------|
    deprojScale = [np.ones(subMass.shape), np.ones(subMass.shape)]
    if deproj:
        # |------------------------------|
        # |--- Get satellite location ---|
        # |------------------------------|
        satLonD = myMap.observer_coordinate.lon.degree
        satLatD = myMap.observer_coordinate.lat.degree
        satR = myMap.observer_coordinate.radius.au 
        satLonR = satLonD * np.pi / 180.
        satLatR = satLatD * np.pi / 180.
        
        if 'crota' in myMap.meta:
            rollIt = myMap.meta['crota']
        elif 'sc_roll' in myMap.meta:
            rollIt = myMap.meta['sc_roll']
        else:
            print ('Neither crota or sc_roll in map metadata. Assuming zero roll')
            rollIt = 0

        satxyz = np.array([np.cos(satLatR)*np.cos(satLonR), np.cos(satLatR)*np.sin(satLonR), np.sin(satLatR)]) * satR
    
        # |-----------------------------------------|
        # |--- Convert into Cart FoV coordinates ---|
        # |-----------------------------------------|
        pixCent = [myMap.data.shape[1]/2, myMap.data.shape[0]/2] # pix is xy but shape is yx

        # Get the heliprojective coord of the pixel
        coordM = myMap.pixel_to_world(pixCent[0] * u.pix, pixCent[1] * u.pix)
        # Get elongation angle -> distance to Thomson Sphere
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
        xs = [satxyz[0]*215, TSxyz[0]*215, satxyz[0]*215, 0]
        ys = [satxyz[1]*215, TSxyz[1]*215, satxyz[1]*215, 0]
        zs = [satxyz[2]*215, TSxyz[2]*215, satxyz[2]*215, 0]
        pts = np.array([xs, ys, zs])   
        
        res = StonyCart2CartFoV(pts, satLatD, satLonD, rollIt)
        res = np.array(res)
        
        satxyz = res[:,0] # not needed after all but keeping along for the ride now
        sunxyz = res[:,1]
        
        
        # |------------------------------------------|
        # |--- Get distance/elong for Billings eq ---|
        # |------------------------------------------|
        # Start by correcting using a single x pos for each Los (the xc value)
        # Possible upgrade to full LoS integation later
        sun_dx1 = sunxyz[0] - xcMap[0]
        sun_dy = sunxyz[1] - f_fovy
        sun_dz = sunxyz[2] - f_fovz
        sun_vec = np.array([sun_dx1, sun_dy, sun_dz] )
        sunDists = np.sqrt(sun_dx1**2 + sun_dy**2 + sun_dz**2)* maskMap[0]

        projR = np.sqrt(sun_dy **2 + sun_dz**2)
        PoSang = np.arctan2(-sun_dx1, projR) * 180 / np.pi
        
        rdp, Bfact = elTheory(projR, PoSang)
        rdp0, Bfact0 = elTheory(projR, 0)
        deprojScale[0] = Bfact/Bfact0 * maskMap[0]  
        bigProj = np.where((deprojScale[0] < 0.1) & (deprojScale[0] != 0.))  
        allProj = np.where(deprojScale[0] != 0.)
        if len(bigProj[0]) > 0:
            deprojScale[0][bigProj] = 0.1
            print ('!!!------ Warning ------!!!')
            print ('Wireframe shape includes points far from plane of sky')
            print (str(len(bigProj[0])) + ' pixels ('+'{:3.1f}'.format(100*len(bigProj[0])/len(allProj[0]))+'%) have deprojection factor of 10x or greater' )
            print( 'Capping these points at 10x')
        
        if multiMode:
            sun_dx1 = sunxyz[0] - xcMap[2]
            sun_dy = sunxyz[1] - f_fovy
            sun_dz = sunxyz[2] - f_fovz
            sun_vec = np.array([sun_dx1, sun_dy, sun_dz] )
            sunDists = np.sqrt(sun_dx1**2 + sun_dy**2 + sun_dz**2)* maskMap[2]
        
            sun_uvec = sun_vec / sunDists

            projR = np.sqrt(sun_dy **2 + sun_dz**2)
            PoSang = np.arctan2(-sun_dx1, projR) * 180 / np.pi
        
            rdp, Bfact = elTheory(projR, PoSang)
            rdp0, Bfact0 = elTheory(projR, 0)
            deprojScale[1] = Bfact/Bfact0 * maskMap[2]
            bigProj = np.where((deprojScale[1] < 0.1) & (deprojScale[1] != 0.))  
            allProj = np.where(deprojScale[1] != 0.)
            if len(bigProj[0]) > 0:
                deprojScale[1][bigProj] = 0.1
                print ('!!!------ Warning ------!!!')
                print ('Wireframe 2 shape includes points far from plane of sky')
                print (str(len(bigProj[0])) + ' pixels ('+'{:3.1f}'.format(100*len(bigProj[0])/len(allProj[0]))+'%) have deprojection factor of 10x or greater' )
                print( 'Capping these points at 10x')
            
        
        #fig = plt.figure()
        #plt.imshow(deprojScale[1], origin='lower')
        #plt.show()
        #print (sd)
                
    #|--------------------------|
    #|--- Get simple density ---|
    #|--------------------------|
    # Start with assuming full mass goes into WF1
    # All in  g/Rs^3 for now
    dens = subMass * maskMap[0]
    notZero = np.where(maskMap[0] != 0)
    if doInner:
        dens[notZero] = dens[notZero] / (widsIntnz[notZero] - widsIntnzI[notZero]) / cellArea[notZero] / deprojScale[0][notZero]
    else:
        dens[notZero] = dens[notZero] / widsIntnz[notZero] / cellArea[notZero] / deprojScale[0][notZero]
    
    dens2 = [None]
    if multiMode:
        dens2 = subMass * maskMap[2]
        notZero = np.where(maskMap[2] != 0)
        dens2[notZero] = dens2[notZero] / widsIntnz2[notZero] / cellArea[notZero] / deprojScale[1][notZero]
     
   
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
        scaleIt = deprojScale[1][overlap] * w2[overlap] / (w1[overlap] * densRatio*deprojScale[0][overlap] + (w2 - w1)[overlap]*deprojScale[1][overlap])
        dens[overlap]  = scaleIt * densRatio * dens2[overlap]
        dens2[overlap] = scaleIt * dens2[overlap]
        dens2 = dens2 / (6.957e10 **3 )
    dens = dens / (6.957e10 **3 )
    densMap = [dens , dens2 ] # convert to g/cm^3
    
    
    # 2d plotting example (for testing)
    if False:
       fig = plt.figure()
       vval = 1e12
       plt.imshow(dens, vmin=-vval, vmax=vval, origin='lower')
       plt.show()
    
    #|---------------------|
    #|--- Return things ---|
    #|---------------------|
    return widMap, xcMap, maskMap, densMap, subMass, outFoV, [FOV2x, FOV2y, FOV2z, sunxyz], [FOV2xS, FOV2yS, FOV2zS]  



# |--------------------------|
# |--------------------------|
# |--- Plotting functions ---|
# |--------------------------|
# |--------------------------|

# |-----------------------------|
# |--- 3D Density Cloud plot ---|
# |-----------------------------|
def dingo3d(widMap, xcMap, densMap, outFoV, pix2FOV, shell=True, plotIt=True):
    ''' 
    3D scatter plot of the wireframe points colored by density. This launches 
    an interactive plot window that one can rotate to see the cloud from 
    different perspectives. Python is not always happy about an interactive plot
    with a large number of points so results from mass2dens with downselect greater
    than 1 should be used. This does not automatically save a figure because we 
    do not want to force a specific viewing anglue in 3d
    
    Inputs:
        widMap:     an array of the widths (in Rs) perp to the plane of sky 
                    the array contains [wf1, wf1_inner, wf2] 
                    (direct output from mass2dens)
    
        xcMap:      the distance of the center of mass from the plane of sky in the same
                    format as widMap. also in Rs
                    (direct output from mass2dens)
    
        densMap:    same format as widMap/xcMap but for the calculated density. In g/cm^3
                    (direct output from mass2dens)
    
        outFoV:     an array with [x0, xf, y0, yf, downselect] where the first four 
                    elements represent the extent of the sub-field of view (in pix) and
                    downselect is an integer representing the 1D reduction in resolution.
                    (direct output from mass2dens)
    
        pix2FOV:    an array with three interpolation functions ([FOV2x, FOV2y, FOV2z] ) 
                    that convert from from a pixel in the original image ((pixx, pixy)) 
                    (direct output from mass2dens)
    
    Optional Inputs:
        shell:      flag to plot just the shell of a wireframe versus a solid wf object
                    with internal points. the density is uniform along a LoS so this really
                    doesn't do much beyond overloading the plot window
    
    '''
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
def dingo2d(widMaps, densMaps, maskMaps, outFoVs, pix2FOVs, showLog=False, figName=None, times=None, showRs=False):
    ''' 
    2D contour plot(s) of the width of the wireframe(s) perpendicular to the plane
    of the sky. If there are two wireframes the outer will be shown on the left and
    the innner on the right. If a time series is passed then there will be one row
    for each time. 
    
    Inputs:
        widMaps:    an array of the widths (in Rs) perp to the plane of sky 
                    the array contains [wf1, wf1_inner, wf2] 
                    (direct output from mass2dens)
    
    
        densMaps:   same format as widMaps but for the calculated density. In g/cm^3
                    (direct output from mass2dens)

        masksMaps:  same format as widMaps but for a binary mask showing the extent of the
                    wireframe shape(s) in the FoV (1 inside wf, 0 outside)
                    (direct output from mass2dens)
    
        outFoVs:    an array with [x0, xf, y0, yf, downselect] where the first four 
                    elements represent the extent of the sub-field of view (in pix) and
                    downselect is an integer representing the 1D reduction in resolution.
                    (direct output from mass2dens)
    
        pix2FOVs:   an array ([FOV2x, FOV2y, FOV2z, sun_pos] ) with three interpolation functions
                    that convert from from a pixel in the original image ((pixx, pixy)) 
                    to FoV cartesian and the corresponding location of the sun
                    (direct output from mass2dens)

    Optional Inputs:
        showLog:    flag to show the contours on a log scale instead of linear 
                    (defaults to false)
    
        showRs:     flag to show the axes in Rs instead of pixels
                    (defaults to false)
    
        figName:    a string name to use when saving the figure. saved as dingo2d_figname.png
                    (defaults to None and just displays the fig instead of saving)
    
        times:      a list of observation times (strings) to use as titles for the panels
    
    '''
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
    fSizes  = {'1s':(6.8,5.),'2s':(4,5), '3s':(4,8.), '4s':(4,10.), '5s':(8,8.), '6s':(8,8.), '7s':(6.5,11.), '8s':(6.5,11.), '1m':(7,3.8),'2m':(7,6.6), '3m':(7,8.), '4m':(7,10.), '5m':(10,7.), '6m':(10,7.), '7m':(10,9.), '8m':(10,9.)} #tagIt:2dsize
    # Number of plot panels and width_ratios
    rval = 0.85
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
    
    # Ignoring posibility of weirdly wide but short
    aspR = npx / npy
    if aspR > 0.9:
        nsq = np.max([npx, npy])
        if npx < nsq:
            limxs[1] += nsq - npx 
        elif npy < nsq:
            limys[1] += nsq - npy 
        picx, picy = nsq, nsq
        figmod = 1
    else:
        picx, picy = npx, npy
        figmod = 0.3 + 0.7*aspR
    

    #|----------------------------|    
    #|--- Actually make figure ---|
    #|----------------------------|
    fsize = fSizes[myStyle]
    fig = plt.figure(figsize=(fsize[0]*figmod, fsize[1]))
    gs = gridspec.GridSpec(gny, gnx, width_ratios=widRats)
    gs.update(wspace=0.05)
    gs.update(hspace=0.15)
    
    axes = [[], []]
    count = [1,1]
    myLab = 'Pixels'
    if showRs: myLab = 'R$_S$'
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
                    anAx.set_ylabel(myLab)
                
                if (j != (gny-1)) and (count[ios[i]] != nTimes):
                    anAx.set_xticklabels([])
                else:
                    anAx.set_xlabel(myLab)
                count[ios[i]] += 1    

    # Make cbar axis
    if gny <=2:
        cax = plt.subplot(gs[:,-1])
    elif gny == 3:
        cax = plt.subplot(gs[1,-1])
    elif gny == 4:
        cax = plt.subplot(gs[1:3,-1])
    
    
    #|-------------------------|    
    #|--- Process each time ---|
    #|-------------------------|
    # Holders for the results
    allDens1 = np.zeros([nTimes, picy, picx])
    allMask1 = np.zeros([nTimes, picy, picx])
    allpxx   = np.zeros([nTimes, picy, picx])
    allpyy   = np.zeros([nTimes, picy, picx])
    if multiMode:
        allDens2 = np.zeros([nTimes, picy, picx])
        allMask2 = np.zeros([nTimes, picy, picx])
    
    # |--- Time loop ---|    
    for i in range(nTimes):   
        #|--- Unpackage FoV things ---|
        minpx, maxpx, minpy, maxpy, downSize = outFoVs[i]
        dx = minpx - limxs[0]
        dy = minpx - limxs[0]
        sx = maxpx - minpx + 1
        sy = maxpy - minpy + 1
    
        dens1 = densMaps[i][0]
        mask1 = maskMaps[i][0]
        if multiMode:
            dens2 = densMaps[i][1]
            mask2 = maskMaps[i][2]

        #|--- Make the mini FoV grid in pix ---|
        pxs = np.arange(minpx, maxpx+1, downSize)
        pys = np.arange(minpy, maxpy+1, downSize)
        pxx, pyy = np.meshgrid(pxs, pys)
    
        #|--- Make the mini FoV grid in Rs ---|
        fovx = pix2FOVs[i][0]((pxx, pyy))
        fovy = pix2FOVs[i][1]((pxx, pyy))
        fovz = pix2FOVs[i][2]((pxx, pyy))
        
        if showRs:
            limxs = [np.min(fovy)-pix2FOVs[i][3][1], np.max(fovy)-pix2FOVs[i][3][1]]
            limys = [np.min(fovz)-pix2FOVs[i][3][2], np.max(fovz)-pix2FOVs[i][3][2]]
 
        
        #|--- Get resolution ---|
        dys = np.zeros(fovx.shape)
        dys[:,1:] = fovy[:,1:] - fovy[:,:-1]
        dys[:,0] = dys[:,1] 
        dzs = np.zeros(fovx.shape)
        dzs[1:,:] = fovz[1:,:] - fovz[:-1,:]
        dzs[0,:] = dzs[1,:]
        cellArea = dys * dzs
    
        allDens1[i,dy:dy+sy,dx:dx+sx] = dens1 * maskMaps[i][0]
        allMask1[i,dy:dy+sy,dx:dx+sx] = maskMaps[i][0]
        allpxx[i,dy:dy+sy,dx:dx+sx] = pxx
        allpyy[i,dy:dy+sy,dx:dx+sx] = pyy
        if multiMode:
            allDens2[i,dy:dy+sy,dx:dx+sx] = dens2 * maskMaps[i][2]
            allMask2[i,dy:dy+sy,dx:dx+sx] = maskMaps[i][2]
            
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
            allDens1[i,dy:dy+sy,dx:dx+sx] = logd1 * maskMaps[i][0]
            if multiMode:
                allDens2[i,dy:dy+sy,dx:dx+sx] = logd2 * maskMaps[i][2]
                
    #|----------------------|
    #|--- Fill in figure ---|
    #|----------------------|
    # |--- Get density range to set contours ---|
    vval = np.median(np.abs(allDens1[allDens1 !=0])) * 3 
    cmap = plt.get_cmap('RdYlBu_r')
    cmap.set_under('k')
    
    power = int(np.log10(vval)) - 1
    scaleIt = 10 ** power
    vvals = [-vval / scaleIt, vval / scaleIt]

    if showLog:
        nz = allDens1[allDens1 !=0]
        scaleIt = 1
        vmax = int(np.percentile(nz, 90))
        vvals = [vmax-3, vmax]
        
    
    for i in range(nTimes):
        thisdens1 = allDens1[i,:,:]
        mask1 = allMask1[i]
        thisdens1[np.where(thisdens1 == 0)] = -9999
        # Grab the first for the cbar
        if i == 0:
            im = axes[0][i].imshow(thisdens1/scaleIt, origin='lower', vmin=vvals[0], vmax=vvals[1], cmap=cmap, extent=[limxs[0], limxs[1], limys[0], limys[1]])
        else:
            axes[0][i].imshow(thisdens1/scaleIt, origin='lower', vmin=vvals[0], vmax=vvals[1], cmap=cmap, extent=[limxs[0], limxs[1], limys[0], limys[1]])
        if type(times) != type(None):
            axes[0][i].set_title(times[i], fontsize=10,loc='right')
        
        if multiMode:
            thisdens2 = allDens2[i,:,:]
            mask2 = allMask2[i]
            thisdens2[np.where(thisdens2 == 0)] = -9999
            axes[1][i].imshow(thisdens2/scaleIt, origin='lower', vmin=vvals[0], vmax=vvals[1], cmap=cmap, extent=[limxs[0], limxs[1], limys[0], limys[1]])
            xs = np.arange(limxs[0], limxs[1]+1, 1)
            ys = np.arange(limys[0], limys[1]+1, 1)
            xxxs, yyys = np.meshgrid(xs, ys)
            axes[0][i].contour(xxxs, yyys, mask2, levels=[0], linestyles='--', colors='w')
            axes[1][i].contour(xxxs, yyys, mask1, levels=[0], linestyles='--', colors='k')
            
    # Add the color bar
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')   
    if showLog:
        cbar.set_label('Log$_{10}$ Density (g cm$^{-3}$)', rotation=270, labelpad=15)
    else:  
        cbar.set_label('Density (1e'+str(power)+' g cm$^{-3}$)', rotation=270, labelpad=15)
    fig.subplots_adjust(right=rval, left=lval, top=0.95,bottom=0.1)
    if figName:
        if not os.path.exists('dingoOutputs/'):
            # Make if if it doesn't exist
            os.mkdir('dingoOutputs/')
        plt.savefig('dingoOutputs/dingo2d_'+figName+picType)
    else:
        plt.show()
    


# |--------------------|
# |--- In Situ plot ---|
# |--------------------|
def dingo1d(myMaps, widMapIns, xcMapIns, densMapIns, outFoVs, pix2FoVs, obsSats, vCME=400, scaleFactors=[1, 0.2], figName=None, timeMode='hr', writeIt=True):
    ''' 
    1D line plots of the density both in place at the time of observation (left panel)
    and shifted to an observing satellite using a very simple propagation model (right).
    The in situ results assume a constant propagation speed and expansion but no drag or
    any other evolutionary effects. The expansion model assumes a constant ratio between
    the expansion speed in the radial direction and the radial propagation speed and a 
    second factor sets the size at impact relative to what it would be assuming self-similar
    expansion. Please do not treat this as a proper arrival time model, the times are meant
    to be illustrative and it will not include any sheath material accumulated beyond the 
    last remote observation.
    
    
    Inputs:
        myMaps:     the astropy/sunpy maps defining the pointing. the data is not used
        
        widMapIns:  an array of the widths (in Rs) perp to the plane of sky 
                    the array contains [wf1, wf1_inner, wf2] 
                    (direct output from mass2dens)
    
    
        xcMapIns:   the distance of the center of mass from the plane of sky in the same
                    format as widMap. also in Rs
                    (direct output from mass2dens)
        
        densMapIns: same format as widMaps but for the calculated density. In g/cm^3
                    (direct output from mass2dens)

    
        outFoVs:    an array with [x0, xf, y0, yf, downselect] where the first four 
                    elements represent the extent of the sub-field of view (in pix) and
                    downselect is an integer representing the 1D reduction in resolution.
                    (direct output from mass2dens)
    
        pix2FOVs:   an array with three interpolation functions ([FOV2x, FOV2y, FOV2z] ) 
                    that convert from from a pixel in the original image ((pixx, pixy)) 
                    (direct output from mass2dens)
    
        obsSats:    an array of results from get_horizons_coord corresponding to the in situ
                    satellite at the time each observation. technically this would be more 
                    accurate using the time of impact but we haven't calculated that so we're
                    gonna assume it doesn't move too much
    
    Optional Inputs:
        vCME:       the average propagation velocity of the CME, used to determine the transit
                    time and convert physical size to time. in km/s 
                    (defaults to 400 )
    
        scalefactors: an array with [expf1, expf2] where
                    expf1: the first expansion factor which sets the amount of the expansion
                         in the nonradial direction. The value is the ratio of the physical
                         size at the time of impact (width in perp direction) compared to what
                         it would be with self-similar expansion
                         (defaults to 1)

                    expf2: the second expansion factor which sets the rate of the radial
                         expansion speed relative to the radial propagation speed. The 
                         # should be a decimal value between 0 and 1 (and likely closer to 0)
                         (defaults to 0.1)
    
        figName:    a string name to use when saving the figure. saved as dingo2d_figname.png
                    (defaults to None and just displays the fig instead of saving)
    
        timeMode:   either 'hr' or 'rw' to show the in situ results in hours from the earliest
                    remote observation time or to convert to real world dates (eg yyyy-mm-dd hh:mm)
                    (defaults to hr bc not an arrival time model and avoiding implying as much)
    
        writeIt:    a flag to save the results in text files. This generates two text files
                    dingo1d_IP/IS_savename.dat in the dingOutputs folder. The IP file has the
                    time of the remote observation, the radial distance (Rs), the density (g/cm3)
                    and whether that point corresponds to the inner or outer wf. The IS file has 
                    the time of the remote observation, the time since the earliest remote obs (hr),
                    the number density, and the inner/outer flag. All profiles from different remote
                    times will be dumped in the same file.
                    (defaults to True)
    
    '''
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
            
    # |-------------------------|
    # |--- Set up save files ---|    
    # |-------------------------|
    if (type(figName) == type(None)):
        writeIt = False
    if writeIt:
        if not os.path.exists('dingoOutputs/'):
            # Make if if it doesn't exist
            os.mkdir('dingoOutputs/')
        
        # File for in place data    
        outFileIP = open('dingoOutputs/dingo1d_IP_'+figName+'.dat', 'w')
        
        # File for in situ data    
        outFileIS = open('dingoOutputs/dingo1d_IS_'+figName+'.dat', 'w')
            
              
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
        # Get elongation angle -> distance to Thomson Sphere
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
        npts = 300
    
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
    
        # |---------------------|
        # |--- Write IP data ---|    
        # |---------------------|
        if writeIt:
            myTime = times[iii].strftime("%Y-%m-%d %H:%M:%S")
            for i in isMain:
                if nIP[i] != 0:
                    aLine = str(rIP[i]) + ' ' + str(nIP[i]) + ' Inner'
                    outFileIP.write(myTime + ' ' + aLine + '\n')
            for i in isSheath:
                if nIP[i] != 0:
                    aLine = str(rIP[i]) + ' ' + str(nIP[i]) + ' Outer'
                    outFileIP.write(myTime + ' ' + aLine + '\n')

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
                            
        # |---------------------|
        # |--- Write IS data ---|    
        # |---------------------|
        if writeIt:
            myTime = times[iii].strftime("%Y-%m-%d %H:%M:%S")
            for i in isSheath[::-1]:
                aLine = str(xvals[i]) + ' ' + str(nIS[i]) + ' Outer'
                outFileIS.write(myTime + ' ' + aLine + '\n')
            for i in isMain[::-1]:
                aLine = str(xvals[i]) + ' ' + str(nIS[i]) + ' Inner'
                outFileIS.write(myTime + ' ' + aLine + '\n')
                
            
    # |------------------------|
    # |--- Close save files ---|    
    # |------------------------|
    if writeIt:
        outFileIP.close() 
        outFileIS.close() 
    
    
    # |-------------------|
    # |--- Save figure ---|    
    # |-------------------|
    if figName:
        plt.savefig('dingoOutputs/dingo1d_'+figName+picType)
    else:
        plt.show()
    

# |------------------------------|
# |--- Calculate total masses ---|
# |------------------------------|
def getMasses(widMap, densMap, outFoV, pix2FOV, printIt=True):
    ''' 
    0D dingo aka just the total mass in the wireframe region. For a single WF
    case this will differ slightly from the wombat mass calculation because we
    account for projection better (unless it's turned off). For two WF this 
    includes the effects of separating the WFs in the overlap regions
    
    Inputs:
        widMap:     an array of the widths (in Rs) perp to the plane of sky 
                    the array contains [wf1, wf1_inner, wf2] 
                    (direct output from mass2dens)
        
        densMap:    same format as widMap/xcMap but for the calculated density. In g/cm^3
                    (direct output from mass2dens)
    
        outFoV:     an array with [x0, xf, y0, yf, downselect] where the first four 
                    elements represent the extent of the sub-field of view (in pix) and
                    downselect is an integer representing the 1D reduction in resolution.
                    (direct output from mass2dens)
    
        pix2FOV:    an array with three interpolation functions ([FOV2x, FOV2y, FOV2z] ) 
                    that convert from from a pixel in the original image ((pixx, pixy)) 
                    (direct output from mass2dens)
    
    Optional Inputs:
        printIt:    flag to print the results to screen
                    (defaults to true)
    
    Outputs:
        res:        either [mass] or [mass1, mass2] depending on whether one or two wf
                    have been provided. In units of 1e15 g
    
    '''
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
    


# |-----------------------|
# |-----------------------|
# |--- Wrapper Helpers ---|
# |-----------------------|
# |-----------------------|

# |--------------------------------------|
# |--- Covert text file to args array ---|
# |--------------------------------------|
def input2args(inputData):
    ''' 
    Helper function to take a text files and convert it to
    an argument array, analogous to what would be used if 
    calling fully from command line. Command line version
    was written first so this avoids rewriting code.
    
    Inputs:
        inputData:  a string pointing to a dingo_config text file
    
    Outputs:
        args: an array with the contents of the input file sorted
              as would be expected from the sys.argv command line
              version of running dingo

    '''
    tags = inputData[:,0]
    inputDict = {}
    args = []
    for i in range(len(tags)):
        aTag= tags[i].replace(':','').lower()
        inputDict[aTag] = inputData[i,1]

    # |--- Log file ---|    
    if 'logfile' in inputDict.keys():
        args.append(inputDict['logfile'])
    else:
        sys.exit('Dingo config file missing logfile entry')
    # |--- IDs ---|    
    if 'ids' in inputDict.keys():
        args.append(inputDict['ids'])
    else:
        sys.exit('Dingo config file missing ids entry')
    # |--- dim ---|    
    if 'dim' in inputDict.keys():
        args.append(inputDict['dim'])
    else:
        sys.exit('Dingo config file missing dim entry')
    
    # |--- Optional params ---|
    if 'pictype' in inputDict.keys():
        if inputDict['pictype'].lower() in ['.png', '.pdf', 'png', 'pdf']:
            args.append(inputDict['pictype'])
        else:
            sys.exit('Dingo config error - pictype must be .png or .pdf')
    for atag in ['expf1', 'expf2', 'densratio', 'vcme']:
        if atag in inputDict.keys():
            try:
                temp = float(inputDict[atag])
            except:
                sys.exit('Dingo config error - '+atag+' must be float')
            args.append(atag+'_'+inputDict[atag])
    if 'ds' in inputDict.keys():
        try:
            temp = int(inputDict['ds'])
        except:
            sys.exit('Dingo config error - ds must be integer')
        args.append('ds_'+temp)
        
    for atag in ['doinner', 'projoff', 'logplot']:    
        if atag in inputDict.keys():
            if inputDict[atag] in [True, 'True', 'true', '1']:
                args.append(atag)   
    if 'target' in  inputDict.keys():
        args.append(inputDict['target'])
    if 'savename' in inputDict.keys():
        args.append(inputDict['savename'])
    
    return args

# |-----------------------------|
# |--- Process Required Args ---|
# |-----------------------------|
def processArgs(args):
    '''
    Helper script to check that all the required inputs
    have been included and that they have reasonable values.
    The required inputs and checks are:
        log file - an existing wombat log file
        
        log ids  - the id or ids of lines to process via dingo. it can
                   be a single wf shape or two wfs but the same instrument
                   and wombat save pickle must be used for all cases. This
                   should be a string of the form id+id+... with up to two
                   different wf shapes and up to eight times.
    
        dimension - 0d, 1d, 2d, or 3d. sets if we want total mass, line plots
                    contour plot, or 3d scatter plot
    
    Inputs:
        args: the results from sys.argv or from input2args
    
    Outputs:
        logFile - the full contents of the log file input file
        
        miniLog - logfile, but only the lines selected by the ids
    
        uniqTs - an array of the times selected by ids (without duplicates in 
                 the case of two wf mode)
    
        uniqShapes - an array of wf shapes selected by ids (without duplicates
                     in the case of multi time mode)
        
        nTimes - the number of times selected by ids 
    
        singleWF - a flag if single wf or multi wf mode
        
        mode - dimension converted to an integer
    
        pairTimes - a list of all times with a pair of inner/out wf. We allow for
                    multiple wfs to have times with only one of the two wfs and these
                    appear in uniqTs but not pairTimes
    
        pairIds   - a list of all the paired ids [[in1, out1], [in2, out2], ...]
        
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
    pairTimes = None
    pairIds   = None
    if nplus > 0:
        txtIds = np.array(ids) - 1 # indexing from 0 in python
        miniLog = logFile[txtIds,:]
        
        # |--- Check for pickle/instrument match ---|
        if (len(np.unique(miniLog[:, 13])) != 1) or (len(np.unique(miniLog[:, 1])) != 1):
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
            singleWF  = False
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
        if len(args) < 4:
            sys.exit('Missing arguments (probably target) for 1d mode')
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
         
    return logFile, miniLog, uniqTs, uniqShapes, nTimes, singleWF, mode, pairTimes, pairIds

# |--------------------------|
# |--- Process Bonus Args ---|
# |--------------------------|
def processBonusArgs(allBonus, mode):
    '''
    Helper script to check for any bonus inputs after all the required
    arguments have been pulled from args. If 1D mode the target is 
    necessary but everything else is optional. All outputs are returned
    regardless of finding a matching value in allBonus (defaults used in
    that case) 
    
    Inputs:
        allBonus: anything remaining in args after the mandatory values have
                  been removed
    
        mode:     the dingo mode (0-3)
    
    Outputs:
        target: the name of the in situ satellite of interest. This is only 
                needed for the 1D case and otherwise ignored Currently the
                supported options are ACE, BepiColombo, DSCOVR, MAVEN, 
                PSP, SolO, STEREO-A, STEREO-B, VEX, Wind. Additional
                satellites could be added as long as they exist in
                sunpy get_horizons_coord and the correct form of the tag
                is used. We allow some common short forms of these names 
                (e.g Bepi, STA, ...) and nothing is case sensitive. 
        
        saveName: a string name tag to add to the figures/output files. if not
                  provided then figures will be displayed and not saved
    
        expf1: the first expansion factor which sets the amount of the expansion
               in the nonradial direction. The value is the ratio of the physical
               size at the time of impact (width in perp direction) compared to what
               it would be with self-similar expansion
               (defaults to 1)
    
        expf2: the second expansion factor which sets the rate of the radial
               expansion speed relative to the radial propagation speed. It 
               should be a decimal value between 0 and 1 (and likely closer to 0)
               defaults to 0.1)
    
        densratio: the ratio of the density between wf1 and wf2, which is treated as a 
                   constant value over all lines of sight. calculated as wf1/wf2 and 
                   ignored if only a single wf is passed
                   (defaults to 1)
    
        vcme: the average propagation velocity of the CME, used to determine the transit
              time and convert physical size to time. in km/s 
              (defaults to 400 )
        
        ds: (downSelect) an integer to downsample the output resolution. This defaults to 8, which is
            an appropriate value if making a 3d scatter plot with the results but setting
            it to 1 is fine for other modes
        
        dI: (doInner) a flag to try and remove the space corresponding to an internal 
            gap between the legs of a GCS/torus wireframe. This is only meant
            to be used if the gap is obscured from the satellites PoV, the 
            gap will automatically be included when it is observed. This is not
            fully tested and discourage use for now.
    
        logPlot: flag to show the contours on a log scale instead of linear 
                 (defaults to false)
        
        projoff: flag to deproject the masses accounting for the wf width perp to the
                PoS via Billings instead of treating all pts as at Thomson sphere
    
    '''
    # Set the defaults
    expf1 = 1
    expf2 = 0.1
    densratio = 1.
    vcme = 400
    ds = 1
    if mode == 3:
        ds = 8
    # Binary options, set at defaults
    dI = False # doInner - take out mid gap of WFs
    logPlot = False
    deproj  = True
    target = None
    saveName = None
    
    #|------------------------|
    #|--- Check for target ---|     
    #|------------------------|
    # Only in 1D mode
    satNames = ['ace', 'bepi', 'bepicolombo', 'dscovr', 'maven', 'parker', 'parkersolarprobe', 'parker_solar_probe', 'psp', 'solarorbiter', 'solo', 'so', 'stereoa', 'stereo-a', 'sta', 'stereob', 'stereo-b', 'stb', 'venusexpress', 'venus_express', 'vex', 've', 'wind']
    
    if mode == 1:
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
        if type(target) == type(None):
            sys.exit('No target given for 1d mode, cannot proceed')
            
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
        elif 'ds_' in aTag.lower():
            try: 
                ds = int(aTag.lower().replace('ds_',''))
            except:
                sys.exit('Error in converting '+aTag+' to ds int')
        elif aTag.lower() == 'doinner':
            dI = True
        elif aTag.lower() == 'projoff':
            deproj = False
        elif aTag.lower() == 'logplot':
            logPlot = True    
        
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
            
            
    return target, saveName, expf1, expf2, densratio, vcme, ds, dI, logPlot, deproj
    


# |--------------------|
# |--- Main Wrapper ---|
# |--------------------|
def dingoWrapper(args):
    """ 
    Function that goes from the command line to the appropriate DINGO
    procedure. It can also be used for external calls by passing the 
    arguments to the function as a single array. The args must be passed
    in the order shown.
    
    MAIN PARAMS: (required, in order)
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

    BONUS PARAMS: (not required, order doesn't matter as long as after main 3)
                    
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
                (e.g Bepi, STA, ...) and nothing is case sensitive. 

        *** Flags - must be included as shown below, all default to false if they
            are not included ***

        doinner: a flag to try and remove the space corresponding to an internal 
                 gap between the legs of a GCS/torus wireframe. This is only meant
                 to be used if the gap is obscured from the satellites PoV, the 
                 gap will automatically be included when it is observed. This is not
                 fully tested and discourage use for now.

        projoff: a flag to not include projection effects when converting masses to
                 densities. The code calculates the Billings factor for each pixel 
                 using the corresponding wireframe center at that point and corrects
                 the observed mass. This correction factor is capped at a max of 10
                 and warns about large plane-of-sky separations
                 (defaults to including projection)
        
        logplot: a flag to plot 2d contours in a log scale instead of linear



        *** Prefix tags - the following options all require using the listed prefix 
            with the # replaced by the desired value (e.g. expf1_0.4 would set  
            expf1 at 0.4) ***
           
        expf1_#: the first expansion factor which sets the amount of the expansion
                 in the nonradial direction. The value is the ratio of the physical
                 size at the time of impact (width in perp direction) compared to what
                 it would be with self-similar expansion
                 (Only used in 1D, defaults to 1)

        expf2_#: the second expansion factor which sets the rate of the radial
                 expansion speed relative to the radial propagation speed. The 
                 # should be a decimal value between 0 and 1 (and likely closer to 0)
                 (Only used in 1D, defaults to 0.1)

        densratio_#: the ratio between the inner and outer wireframe densities (n1/n2).
                     the densities vary from pixel to pixel but the ratio between the 
                     two remains the same in any overlapping regions
                     (defaults to 1.)

        vcme_#: the average interplanetary velocity of the CME. this allows conversion
                of the initial distances into in situ times which only should be taken
                as representative as DINGO does not include any IP evolution beyond expansion
                (Only used in 1D, defaults to 400)

        ds_#: an integer indicating how much to downselect the resolution from the input 
              mass maps. Running full resolution is fine for modes 0-2 but it will break
              3d scatter plots.
              (defaults to 8 for 3D mode, 1 for everything else)
                
    
    """
    
    #|--------------|
    #|--------------|
    #|--- Set up ---|
    #|--------------|
    #|--------------|    
    
    #|--- Standard input error message ---|
    global errorStrings
    errorStrings = [' ', '  python3 dingo.py logFile id(s) dim otherParams', '          where:', '          - logFile is a wombat log file', '          - ids is an integer or int+int', '          - dim set the mode/dimensions from [0D, 1D, 2D, 3D]', '          - otherParams includes target (only for 1D), pic type,', '             save name, expf1_*, expf2_*, vcme_*, and densratio_*']
    

    #|----------------------------------|
    #|--- Check the number of inputs ---|     
    #|----------------------------------|
    nArgs = len(args)
    textInput = False
    if (nArgs == 2) or (len(args) > 10):
        print ('Incorrect number of parameters provided. Syntax is')
        for astr in errorStrings:
            print (astr)
        sys.exit()
    # |--- Allow for single input if passing txt file ---|
    elif nArgs == 1:
        try:
            inputData = np.genfromtxt(args[0], dtype=str)
            textInput = True
        except:
            sys.exit('One have one input and cannot process as dingo config file')
    
    #|--------------------------|
    #|--- Process input file ---|     
    #|--------------------------|
    if textInput:
        args = input2args(inputData) 
        
    #|-------------------------------------|
    #|--- Check the critical parameters ---|     
    #|-------------------------------------|
    logFile, miniLog, uniqTs, uniqShapes, nTimes, singleWF, mode, pairTimes, pairIds = processArgs(args)          
                
    #|----------------------------------|
    #|--- Check the bonus parameters ---|     
    #|----------------------------------|
    # Set the defaults
    expf1 = 1
    expf2 = 0.1
    densratio = 1.
    vcme = 400
    ds = 1
    if mode == 3:
        ds = 8
    # Binary options, set at defaults
    dI = False # doInner - take out mid gap of WFs
    logPlot = False
    deproj  = True
    saveName = None
      
    if len(args) >= 4:
        allBonus = args[3:]
        target, saveName, expf1, expf2, densratio, vcme, ds, dI, logPlot, deproj = processBonusArgs(allBonus, mode)
                        
               
    #|--------------------------|
    #|--- Set up run details ---|     
    #|--------------------------|        
    #|--- Open the pickle ---|
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
        aWFi = wf.wireframe(lineI[3].replace('Half', 'Half '), doBack=True)
        ps = []
        for i in range(9):
            if lineI[i+4] != 'None':
                ps.append(float(lineI[i+4]))
        aWFi.params = ps
        aWFi.getPoints()
        wfsI.append(aWFi)
        if not singleWF:
            aWFo = wf.wireframe(lineO[3].replace('Half', 'Half '), doBack=True)
            ps = []
            for i in range(9):
                if lineO[i+4] != 'None':
                    ps.append(float(lineO[i+4]))
            aWFo.params = ps
            aWFo.getPoints()
            wfsO.append(aWFo)


    #|--------------|
    #|--------------|
    #|--- Run it ---|     
    #|--------------|
    #|--------------|
        
    #|--- Process mass maps into density ---|
    widMaps, xcMaps, maskMaps, densMaps, subMasss, outFoVs, pix2FOVs, pix2Sts = [], [], [], [], [], [], [], []
    for i in range(nTimes):
        print ('Processing', uniqTs[i])
        if singleWF:
            widMap, xcMap, maskMap, densMap, subMass, outFoV, pix2FOV, pix2St  = mass2dens(imMaps[i], satDicts[i], wfsI[i], massMaps[i], doInner=dI,  downSelect=ds, deproj=deproj)
        else:
            widMap, xcMap, maskMap, densMap, subMass, outFoV, pix2FOV, pix2St  = mass2dens(imMaps[i], satDicts[i], [wfsI[i], wfsO[i]], massMaps[i], doInner=dI,  densRatio=densratio, downSelect=ds, deproj=deproj)
        # Package it
        widMaps.append(widMap)
        xcMaps.append(xcMap)
        maskMaps.append(maskMap)
        densMaps.append(densMap)
        subMasss.append(subMass)
        outFoVs.append(outFoV) 
        pix2FOVs.append(pix2FOV)    
        pix2Sts.append(pix2St)    
    

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
            out = aboutMe[i] + ' '
            for mass in masses:
                moreOut += '{:.2f}'.format(mass) + ' '
            print (out+moreOut)

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
        dingo2d(widMaps, densMaps, maskMaps, outFoVs, pix2FOVs, figName=saveName, times=uniqTs, showLog=logPlot, showRs=True)
            
    
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

