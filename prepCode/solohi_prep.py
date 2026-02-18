"""
Module for converting a set of solohi images into a mosaic.
Only tested/used on L2 data so little to no processing, just
dumping multiple sets of data into same frame. Port of IDL
function of same name.

"""
import numpy as np
import sunpy.map
import sys
from scc_funs import rebinIDL
from sunspyce import get_sunspyce_hpc_point, get_sunspyce_roll
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.time import parse_time

# L2 data avail at https://solohi.nrl.navy.mil/so_data/L2/

#|----------------------------|
#|---Get Camera Parameters ---|
#|----------------------------|
def get_def_cam_params(instr):
    """
    Function to pull the default camera parameters based on an
    instrument tag
    
    Input:
        instr: the instrument tag
               (from SOLOHI1, SOLOHI2, SOLOHI3, SOLOHI4)

    Output:
        ncol: the number of columns
    
        nrow:
    
        xycoords: a 1x4 array with [col0, col_end, row0, row_end]
    
        roxy: a 1x2 array with two values from [x0, x1, y0, y1]
              (read order x y?)
    
        rodur: a duration in seconds (eq to nrow*ncol / reaout rate)
        
        ypr: an 1x3 array (degrees)
    
        pscal: a float plate scale (?) in deg/pixel

    """
    readout_rate=1.86e6  # 2 Mpixel/sec
    
    if instr == 'SOLOHI1': # top left
        ncol = 2048
        nrow = 1920
        xycoords = [1920+44,1920+44+2047,128,128+1919]	# LL coordinate of detector in mosaic (start=[0,0])
        roxy = ['x1','y0']
        rodur = ncol * nrow / readout_rate    # sec 
        ypr = [53.5,-37.5/2,0.]	# Degrees
        pscal = 0.00986 	# deg/pixel
    elif instr == 'SOLOHI2': # top right
        ncol = 2048
        nrow = 1920
        xycoords = [2048+44,2048+44+1919,2048+44,2048+44+2047]
        roxy = ['y1','x1']
        rodur = ncol * nrow / readout_rate    # sec 
        ypr = [108.,-54.4/2,0.]	# Degrees
        pscal = 0.00986 	# deg/pixel
    elif instr == 'SOLOHI3': # bot right
        ncol = 2048
        nrow = 1920
        xycoords = [0,2047,2048+44,2048+44+1919]
        rotate_order = [2,3,0,1] # why only solohi3 has?
        roxy = ['x0','y1']
        rodur = ncol * nrow / readout_rate    # sec 
        ypr = [53.5,-37.5/2,0.]	# Degrees
        pscal = 0.00986 	# deg/pixel
    elif instr == 'SOLOHI4': # bot left
        ncol = 2048
        nrow = 1920
        xycoords = [0,1919,0,2047]
        roxy = ['y0','x0']
        rodur = ncol * nrow / readout_rate    # sec 
        ypr = [108.,-54.4/2,0.]	# Degrees
        pscal = 0.00986 	# deg/pixel
    else:
        sys.exit('Unrecognized instrument in get_def_cam_params')
    
    
    return ncol, nrow, xycoords, roxy, rodur, ypr, pscal


#|-----------------------------------------|
#|--- ID where each camera goes in grid ---|
#|-----------------------------------------|
def solohi_getgrid(s=0,get_det=True, reduce=4):
    """
    Function to get the locations of each of the solohi
    instruments within the standardized grids. Only set
    up to run with get_det flagged as True. The IDL version
    does more that is not ported here. IDL is solohi_get_grid
    
    Input:
        None
    
    Optional Input:
        s: uncertain, exists in IDL but not used here (defaults to 0)
    
        get_det: flag to get the detector locations for showing instruments
                 on a grid (defaults to True)
    
        reduce: factor to scale down images (not implemented here, defaults to 4)
    
    Output:
        det: an mosaic array with the regions for each SOLOHI instr labeled
    
    Note:
        A lot of steps skipped because only doing the get_det portion
        of this, not any of the plotting 
    """
    
    #|--------------------------------------|
    #|--- Make generic empty grid and XY ---|
    #|--------------------------------------|
    degperpix = 0.00986
    
    fullview = [44.73, 5.2011, -20.73, 20.62]
    degc     = [25.0, 0.0] # value for cter pixels defined as coords [2007,2071]
    pixc     = [2007, 2071]
    xpixels  = 2048 + 1920 + 44 + 4
    ypixels  = 2048 + 2048 + 44 + 4
    
    xgrid  = np.tile(np.arange(xpixels, dtype=float), (ypixels, 1))
    ygrid  = np.rot90(np.tile(np.arange(ypixels, dtype=float), (xpixels, 1)), k=-1)

    xdegval1 = np.arange(xpixels) * degperpix
    xdegval  = xdegval1 + (degc[0] - xdegval1[pixc[0]])
    ydegval1 = np.arange(ypixels) * degperpix
    ydegval  = ydegval1 + (degc[1] - ydegval1[pixc[1]])
    
    det = np.zeros([ypixels,xpixels]) # keeping indexing oppo IDL
    roarrow = np.zeros([4,4])
    
    #|-------------------------------|
    #|--- Fill grid by instrument ---|
    #|-------------------------------|
    for i in range(4):
        instr = 'SOLOHI'+str(i+1)
        ncol, nrow, xycoords, roxy, rodur, ypr, pscal = get_def_cam_params(instr)
        det[np.where((xgrid >= xycoords[0]) & (xgrid <= xycoords[1]) & (ygrid >= xycoords[2]) & (ygrid <= xycoords[3]))] = int(i+1)
        # base wombat doesn't even use the rest of this but keep for now
        tx = 0
        if 'y' in roxy[0]: tx=2
        if '1' in roxy[0]:
            roarrow[tx:tx+2,i] = [xycoords[1+tx]-400, xycoords[1+tx]] 
        else: 
            roarrow[tx:tx+2,i] = [xycoords[tx]+400,xycoords[tx]] 
        tx = 0
        if 'y' in roxy[1]: tx=2
        if '1' in roxy[1]:
            roarrow[tx:tx+2,i] = [xycoords[1+tx]-80,xycoords[1+tx]-80]
        else: 
            roarrow[tx:tx+2,i] = [xycoords[tx]+80,xycoords[tx]+80]
            
    xdeg = np.tile(xdegval, (ypixels, 1))[::-1,::-1]
    ydeg = np.tile(ydegval, (xpixels, 1))[::-1,::-1]
    
    # dont have an s so skipping to plotim part (lines 119-171 not hit)
    
    # only running with get_det so far so skipping to skip plot (missing 175-227)
    
    #|------------------------------|
    #|--- Return the mosaic grid ---|
    #|------------------------------|
    if get_det:
        return det
    
    return None

#|--------------------------|
#|--- Make mosaic header ---|
#|--------------------------|
def mosaic_hdr(hdr):
    """
    Function to make a header suitable for the mosaic images
    where the four SOLOHI panels have been combined into one
    
    Input:
        hdr: the header for one of the original panels
    
    Output:
        hdr: the updated header
    
    Note:
        Not a full port, skipped some of the header parameters that
        are not necessary for wombat
    
    """
    # Port of the essential parts of get_solohi_pointing that give 
    # the mosaic hdr
    
    projval = [2019.7,2069.24]
    hdr['naxis1'] = 4016 / hdr['NBIN1']
    hdr['naxis2'] = 4144 / hdr['NBIN2']
    hdr['rectify'] = 'T'
    
    hdr['WCSAXES'] = 2
    hdr['CRPIX1'] = (projval[0] - hdr['R1COL'] + 1) / hdr['NBIN1']
    hdr['CRPIX2'] = (projval[1] - hdr['R1ROW'] + 1) / hdr['NBIN2']
    hdr['CRPIX1A'] = hdr['CRPIX1']
    hdr['CRPIX2A'] = hdr['CRPIX2']
    hdr['CDELT1'] = 0.010305687800767 * hdr['nbin1']
    hdr['CDELT2'] = 0.010305687800767 * hdr['nbin2']
    hdr['CDELT1A'] = -hdr['cdelt1']
    hdr['CDELT2A'] = hdr['cdelt2']
    hdr['WCSNAME'] = 'Helioprojective Zenith Polynomial'  
    hdr['CTYPE1'] = 'HPLN-ZPN'
    hdr['CTYPE2'] = 'HPLT-ZPN'
    hdr['PV1_1'] = 0.0
    hdr['PV1_2'] = 90.0
    hdr['PV2_0'] = 0.000122109
    hdr['LONPOLE'] = 180.
    hdr['LATPOLE'] = 0.0    
    
    hdr['CUNIT1'] =	'deg'
    hdr['CUNIT2'] =	'deg'
    hdr['CUNIT1A'] =	'deg'
    hdr['CUNIT2A'] =	'deg'
    hdr['PV2_0A'] = hdr['PV2_0']
    hdr['PV2_1A'] = hdr['PV2_1']
    hdr['PV2_2A'] = hdr['PV2_2']
    hdr['PV2_3A'] = hdr['PV2_3']
    hdr['PV2_4A'] = hdr['PV2_4']
    hdr['PV2_5A'] = hdr['PV2_5']
    
    point = get_sunspyce_hpc_point(hdr['date-avg'], 'solo', instrument='SOLO_SOLOHI_ILS', doDeg=True)
    hdr['CRVAL1']= point[0]	#0
    hdr['CRVAL2']= point[1]	#25
    point[2] = point[2] + 0.183176
    dtor = np.pi / 180.
    hdr['PC1_1']= np.cos(point[2] * dtor)
    hdr['PC1_2']= -np.sin(point[2] * dtor)
    hdr['PC2_1']= np.sin(point[2] * dtor)
    hdr['PC2_2']= np.cos(point[2] * dtor)

    hdr['PV1_1']= 0.0 	    # deg
    hdr['PV1_2']= 90.0	    # deg
    hdr['PV1_3']= 180.0	    # deg
    hdr['PV1_1A']= 0.0 	    # deg
    hdr['PV1_2A']= 90.0	    # deg
    hdr['PV1_3A']= 180.0	    # deg
    hdr['LATPOLE']= 0.0
    hdr['LONPOLE']= 180.0
    hdr['LATPOLEA']= 0.0
    hdr['LONPOLEA']= 180.0
    
    roll, dec, ra =  get_sunspyce_roll(hdr['date-avg'], 'solo', instrument='SOLO_SOLOHI_ILS', system='GEI')
    if ra < 0: ra = ra + 360
    roll = roll + 0.183176
    hdr['CRVAL1A'] = ra
    hdr['CRVAL2A'] = dec
    hdr['CRPIX1A'] = hdr['CRPIX1']
    hdr['CRPIX2A'] = hdr['CRPIX2']
    hdr['PC1_1A'] = np.cos(roll*dtor)
    hdr['PC1_2A'] = np.sin(roll*dtor)
    hdr['PC2_1A'] = -hdr['PC1_2A']
    hdr['PC2_2A'] = hdr['PC1_1A']
    
    if not hdr['rectify']:
        sys.exit('Havent ported rectify portion of get_solohi_pointing')
    
    # Not doing coords for now
    
    return hdr
    
#|------------------------------------|
#|--- Main Function to make Mosaic ---|
#|------------------------------------|
def solohi_fits2grid(filesIn, doFull=False):
    """
    Main function to combine a set of four SOLOHI images
    into a single mosaic image. We assume the given data
    is level 2 and do not do lower level processing.
    
    Input:
        filesIn: a list of fits files
    
    Optional Input:
        doFull: a flag to keep full resolution results instead
                of downsizing for the combined image (defaults False)
    
    Output:
        binIm -or- fullIm: either a binned or full resolution mosaic image
        hdr: the corresponding header
    
    Note:
        Not a full port, skipped some of the header parameters that
        are not necessary for wombat
    
    """
    
    #|-----------------------|
    #|--- Get the binning ---|
    #|-----------------------|
    with fits.open(filesIn[0]) as hdulist:
        im  = hdulist[0].data
        hdr = hdulist[0].header
    nbin = hdr['nbin1']
    
    #|-----------------------|
    #|--- Setup empty arr ---|
    #|-----------------------|
    # Not doing prep since using L2
    det_grid = solohi_getgrid(get_det=True)
    if nbin != 1:
        det_grid = rebinIDL(det_grid, np.array([int(4144/nbin),  int(4016/nbin)]))
        imgs1 = np.zeros([2, int(1920/nbin), int(2048/nbin)])
        if hdr['rectify']:
            imgs2 = np.zeros([2,int(2048/nbin),int(1920/nbin)])
        else:
            imgs2 = np.zeros([2,int(1920/nbin),int(2048/nbin)])
    
    detcs = np.array(['1','2','3','4'])
    tr = np.array([1920,2048]) / nbin 
    bl = tr-1
    
    zs = np.zeros(4)
    hdr0 = hdr
    for i in range(len(filesIn)):
        # no prepping for now
        
        #|-----------------------|
        #|--- Open the images ---|
        #|-----------------------|
        with fits.open(filesIn[i]) as hdulist:
            im  = hdulist[0].data
            hdr = hdulist[0].header
        if not hdr['rectify']:
            sys.exit('Need to port solohi_rectify')
        if len(hdr['detector']) > 1:
            detcs = np.array(['TL','TR','BR','BL'])
        
        # ignoring bkg keyword (137-146)
        goodIm = im[np.isfinite(im)]
        if np.max(goodIm) == np.min(goodIm):
            im[0,0:3] = np.arange(3) + np.min(goodIm)
        
        
        z = np.where(detcs == hdr['detector'])[0][0]
        zs[i] = z
        
        #|------------------------------|
        #|--- Stick data in img vars ---|
        #|------------------------------|
        # IDL reasons for not going directly into fullIm?
        if hdr['detector'] == '1':
            imgs1[0,:,:] = imgs1[0,:,:] + im
        elif hdr['detector'] == '2':
            imgs2[0,:,:] = imgs2[0,:,:] + im
        elif hdr['detector'] == '3':
            imgs1[1,:,:] = imgs1[1,:,:] + im
        elif hdr['detector'] == '4':
            imgs2[1,:,:] = imgs2[1,:,:] + im

    #|--------------------------------|
    #|--- Copy from im into fullIm ---|
    #|--------------------------------|
    fullIm = np.copy(det_grid)
    idx = np.where(det_grid==1)
    fullIm[idx[0][0]:idx[0][-1]+1, idx[1][0]:idx[1][-1]+1] = imgs1[0,:,:]
    idx = np.where(det_grid==2)
    fullIm[idx[0][0]:idx[0][-1]+1, idx[1][0]:idx[1][-1]+1] = imgs2[0,:,:]
    idx = np.where(det_grid==3)
    fullIm[idx[0][0]:idx[0][-1]+1, idx[1][0]:idx[1][-1]+1] = imgs1[1,:,:]
    idx = np.where(det_grid==4)
    fullIm[idx[0][0]:idx[0][-1]+1, idx[1][0]:idx[1][-1]+1] = imgs2[1,:,:]

    mhdr = mosaic_hdr(hdr0)


    #|--------------|
    #|--- Bin it ---|
    #|--------------|
    binN = 4
    if not doFull:
        sz = fullIm.shape
        binIm = rebinIDL(fullIm, np.array([int(sz[0]/binN),  int(sz[1]/binN)]))
        mhdr['dstop1'] = mhdr['dstop1'] / binN
        mhdr['dstop2'] = mhdr['dstop2'] / binN
        mhdr['CRPIX1'] = 0.5+(mhdr['crpix1']-0.5)/binN
        mhdr['CRPIX1A']= 0.5+(mhdr['CRPIX1A']-0.5)/binN
        mhdr['CRPIX2'] = 0.5+(mhdr['crpix2']-0.5)/binN
        mhdr['CRPIX2A']= 0.5+(mhdr['CRPIX2A']-0.5)/binN
        mhdr['CDELT1'] =  mhdr['CDELT1']*binN
        mhdr['CDELT2'] =  mhdr['CDELT2']*(binN)
        mhdr['CDELT1A'] =  mhdr['CDELT1A']*(binN)
        mhdr['CDELT2A'] =  mhdr['CDELT2A']*(binN)
        
        mhdr['naxis1'] = int(sz[1] / binN) # python oppo order
        mhdr['naxis2'] = int(sz[0] / binN)
        mhdr['dstop1'] = np.min([mhdr['dstop1'], sz[0]]) # think this equiv of IDL
        mhdr['dstop2'] = np.min([mhdr['dstop2'], sz[1]])
        return binIm, mhdr
    #|--------------|
    #|--- Or not ---|
    #|--------------|
    else:
        return fullIm, mhdr
        
