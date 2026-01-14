"""
Module for WCS (World Coordinate System) routines. Astropy
does have some WCS routines but these are exact ports of the
IDL versions. The proj routines convert from intermed coords 
relative to ref pix into the given projection. The inv proj
go from given proj into the intermed coords.

"""
import numpy as np
import scipy

#|----------------------------------|
#|--- Global for unit conversion ---|
#|----------------------------------|
c = np.pi / 180.
cunit2rad = {'arcmin': c / 60.,   'arcsec': c / 3600.,  'mas': c / 3600.e3,  'rad':  1., 'deg':c}

#|-----------------------------------|
#|--- Convert IDL save to WCS hdr ---|
#|-----------------------------------|
def idlsav2wcs(pathIn):
    """
    Function to convert an IDL save file to a wcs style dictionary
    
    Input:
        pathIn: a path to the location of the IDL save file
    
    Output:
        awcs: a wcs like dictionary


    """
    idlwcso = scipy.io.readsav(pathIn)['wcso']
    awcs = {}
    awcs['COORD_TYPE'] = idlwcso['COORD_TYPE'].astype(str)[0]
    awcs['WCSNAME'] = idlwcso['WCSNAME'].astype(str)[0]
    awcs['NAXIS'] = np.array([idlwcso['NAXIS'][0][0], idlwcso['NAXIS'][0][1]]).astype(int)
    awcs['VARIATION'] = idlwcso['VARIATION'].astype(str)[0]
    awcs['COMPLIANT'] = idlwcso['COMPLIANT'].astype(int)[0]
    awcs['PROJECTION'] = idlwcso['PROJECTION'].astype(str)[0]
    awcs['IX'] = idlwcso['IX'].astype(int)[0]
    awcs['IY'] = idlwcso['IY'].astype(int)[0]
    awcs['CRPIX'] = np.array([idlwcso['CRPIX'][0][0], idlwcso['CRPIX'][0][1]]).astype(float)
    awcs['CRVAL'] = np.array([idlwcso['CRVAL'][0][0], idlwcso['CRVAL'][0][1]]).astype(float)
    awcs['CTYPE'] = np.array([idlwcso['CTYPE'][0][0], idlwcso['CTYPE'][0][1]]).astype(str)
    awcs['CNAME'] = np.array([idlwcso['CNAME'][0][0], idlwcso['CNAME'][0][1]]).astype(str)
    awcs['CUNIT'] = np.array([idlwcso['CUNIT'][0][0], idlwcso['CUNIT'][0][1]]).astype(str)
    awcs['CDELT'] = np.array([idlwcso['CDELT'][0][0], idlwcso['CDELT'][0][1]]).astype(float)
    awcs['PC'] = np.array([[idlwcso['PC'][0][0][0], idlwcso['PC'][0][0][1]], [idlwcso['PC'][0][1][0], idlwcso['PC'][0][1][1]]])
    awcs['PROJ_NAMES'] = np.array(idlwcso['PROJ_NAMES'][0]).astype(str)
    awcs['PROJ_VALUES'] = np.array(idlwcso['PROJ_VALUES'][0]).astype(float)
    awcs['ROLL_ANGLE'] = idlwcso['ROLL_ANGLE'].astype(float)[0]
    awcs['SIMPLE'] = idlwcso['SIMPLE'].astype(int)[0]
    
    timeDict = {}
    for item in idlwcso['TIME'][0][0].dtype.names:
        timeDict[item] = idlwcso['TIME'][0][0][item].decode("utf-8")
    awcs['TIME'] = timeDict
    
    pos = []
    for item in idlwcso['POSITION'][0][0]:
        pos.append(item)
    awcs['POSITION'] = pos

    return awcs 

#|-----------------------------------|
#|--- Convert fits hdr to WCS hdr ---|
#|-----------------------------------|
def fitshead2wcs(hdr,system=''):
    """
    Function to convert a fits header to a wcs style dictionary
    
    Input:
        hdr: the fits file header
    
    Optional Input:
        system: option to add a tag (probably 'A') to pull different
                set of tags from the fits header
    Output:
        wcs: a wcs like dictionary
    
    Notes:
        Astropy has version of this that doesn't always play nice with
        the ported versions of the IDL code. This version seems to work
        as expected, particularly if one is careful about pulling the 
        correct system of tags (with our without the A) for whatever the
        use case needs
    """    
    # skipping hdr check, assuming is fine/tbd
    
    #|--------------------------------|
    #|--- Make a header dictionary ---|
    #|--------------------------------|
    tags = list(hdr.keys())
    tags = [aTag.upper() for aTag in tags]
    
    # Skipping Nan/nan check (250-257)
    # Assuming column is not passed (258-325)
    column = ''
    orig_column = ''
    
    
    # Skipping a lot of the safety checks
    n_axis = hdr['naxis']
    
    compliant = True
    
    #|--------------------------------|
    #|--- Check type of wcs system ---|
    #|--------------------------------|
    crota_present = False
    pc_present    = False
    cd_present    = False
    for i in range(n_axis):
        if 'CROTA'+str(i+1) in tags: crota_present = True
        for j in range(n_axis):
            if 'PC'+str(i+1)+'_'+str(j+1) in tags: pc_present = True
            # add lowercase to work with sunpy map metadata format
            elif 'pc'+str(i+1)+'_'+str(j+1) in tags: 
                pc_present = True
                hdr['PC'+str(i+1)+'_'+str(j+1)] = hdr['pc'+str(i+1)+'_'+str(j+1)]
            # dunno what cd is so skipping for now
    
    if pc_present:
        variation = 'PC'
        if crota_present:
            compliant = False
    elif crota_present:
        variation = 'CROTA'
    else:
        print ('Issue in fitshead2wcs, might be uncoded CD version')
        print (Quit)
        
    
    #|-----------------------------|
    #|--- Pull keywords for wcs ---|
    #|-----------------------------|
    # Extract CTYPE keywords (431)
    # Assume that the ctype is found so 1 is x and 2 is y
    if 'CTYPE1' in tags:
        crp1 = hdr['CRPIX1'] 
        crp2 = hdr['CRPIX2'] 
        crval1 = hdr['CRVAL1'+system]
        crval2 = hdr['CRVAL2'+system]
        cunit1 = hdr['CUNIT1'+system]
        cunit2 = hdr['CUNIT2'+system]
    elif 'ctype1' in tags:
        crp1 = hdr['crpix1'] 
        crp2 = hdr['crpix2'] 
        crval1 = hdr['crval1'+system]
        crval2 = hdr['crval2'+system]
        cunit1 = hdr['cunit1'+system]
        cunit2 = hdr['cunit2'+system]
    else:
        print ('Issue in fitshead2wcs, missing ctype in header and havent coded alt version')
        print (Quit)
    
    
    # skipping cname and non compliant strings
    
    if variation == 'PC':
        cdelt1 = hdr['CDELT1'+system]
        cdelt2 = hdr['CDELT2'+system]
        pc = np.zeros([n_axis,n_axis])
        for i in range(n_axis):
            for j in range(n_axis):
                pc[i,j] = hdr['PC'+str(i+1)+'_'+str(j+1)+system]   
    elif variation == 'CROTA': # 594
        # un-IDLifying a lot of this bc we can
        if hdr['CROTA1'] == hdr['CROTA2']:
            roll_angle = hdr['CROTA1']
            hdr['SC_ROLL'] = roll_angle
        else:
            sys.exit('CROTA inconsistent, cannot calculate roll angle')

        cdelt1, cdelt2 = hdr['CDELT1'], hdr['CDELT1']
        pc = np.zeros([n_axis,n_axis])
        if cdelt1 * cdelt2 == 0:
            sys.exit('Zero in cdelt, cannot calculate pc matrix')
        else:
            lam = cdelt2 / cdelt1
            cosa = np.cos(roll_angle * np.pi / 180.)
            sina = np.sin(roll_angle * np.pi / 180.)
            pc[0,0] = cosa
            pc[0,1] = -lam * sina
            pc[1,0] = sina / lam
            pc[1,1] = cosa
 
                
    #|--------------------------------|
    #|--- Determine coord sys type ---|
    #|--------------------------------|
    if hdr['CTYPE1'+system][:4] == 'RA--':
        coord_type = 'Celestial-Equatorial'
        projection = hdr['CTYPE1'+system][5:]
    elif  hdr['CTYPE1'+system][:4] == 'HPLN':
        coord_type = 'Helioprojective-Cartesian'
        projection = hdr['CTYPE1'+system][5:]
    elif hdr['CTYPE1'+system][:4] in ['SOLA']:
        coord_type = 'Helioprojective-Cartesian'
        if (cunit1[:3].upper() in ['ARC', 'DEG', 'MAS']) & (cunit2[:3].upper() in ['ARC', 'DEG', 'MAS']):
            projection = 'TAN'
    else:
        print ('Issue in fitshead2wcs, havent coded non RA coord types')
        print (Quit)
        
    # skipping stuff don't think is needed
    wcsname = coord_type
    
    # Proj names/vals 
    proj_names = []
    proj_values = []
    if 'LONPOLE' in hdr:
        proj_names.append('LONPOLE')
        proj_values.append(hdr['LONPOLE'])
    if 'LATPOLE' in hdr:
        proj_names.append('LATPOLE')
        proj_values.append(hdr['LATPOLE'])
    for key in tags:
        if 'PV' in key:
            if key[-1] != 'A': # don't seem to want the A tag version, not certain here tho
                proj_names.append(key)
                proj_values.append(hdr[key])
    
    #|-------------------------------|
    #|--- Fill the wcs dictionary ---|
    #|-------------------------------|
    wcs = {}
    wcs['coord_type'] = coord_type
    wcs['wcsname'] = wcsname
    wcs['naxis'] = [hdr['naxis1'], hdr['naxis2']]
    wcs['variation'] = variation
    wcs['compliant'] = True
    wcs['projection'] = projection
    wcs['ix'] = 0
    wcs['iy'] = 1
    wcs['crpix'] = [crp1, crp2]
    wcs['crval'] = [crval1, crval2]
    wcs['ctype'] = [hdr['ctype1'+system], hdr['ctype2'+system]]
    wcs['cunit'] = [cunit1, cunit2]
    wcs['cdelt'] = [cdelt1, cdelt2]
    wcs['pc'] = pc
    wcs['proj_names'] = np.array(proj_names)
    wcs['proj_values'] = np.array(proj_values)
    wcs['roll_angle'] = hdr['SC_ROLL'+system]
    # wc['simple']
    # wcs['time']
    # wcs['position']

    return wcs
                       
#|--------------------------------|
#|--- Get Sun center in pixels ---|
#|--------------------------------|
def get_Suncent(my_wcs):
    """
    Function to get the location of the sun center in pixels
    
    Input:
        my_wcs: a wcs style dictionary
    
    Output:
        [scx, scy]: the x and y coords of the sun center (in pixels)
    
    Notes:
        This is simple version of wcs_get_coord with coord [0,0]
        !!! shouldn't use for non TAN projection !!!
    """
    c2rx = cunit2rad[my_wcs['cunit'][0].lower()]
    c2ry = cunit2rad[my_wcs['cunit'][1].lower()]
    coord = [0,0]
    cx = (coord[0] - my_wcs['crval'][0]) / my_wcs['cdelt'][0]
    cy = (coord[1] - my_wcs['crval'][1]) / my_wcs['cdelt'][1]
    pc = my_wcs['pc']
    scx = cx * pc[0,0] + cy * pc[1,0] + my_wcs['crpix'][0] - 1
    scy = cx * pc[0,1] + cy * pc[1,1] + my_wcs['crpix'][1] - 1
    
    return [scx, scy]
    
#|----------------------------|
#|--- Proj intermed to TAN ---|
#|----------------------------|
def wcs_proj_tan(my_wcs, coord, doQuick=False, force_proj=False):
    """
    Function to project intermediate coordinates into TAN coords
    
    Input:
        my_wcs: a wcs style dictionary
    
        coord: a list of intermediate coord pairs (e.g. [[x1,y1], [x2,y2], ...])
               intermed = relative to ref pix but crval not applied
    
    Optional Input:
        doQuick: flag to force the quick version of the calculation instead
                 of doing the full more accurate one (defaults to False)
    
        force_proj: flag to force the full projection calculation insted of 
                    doing the quick, even if check suggests quick is 
                    sufficient (defaults to False)

    Output:
        coord: the coordinates converted to TAN projection 
               (same units as input e.g. arc, rad, deg ) 

    """
    #|--------------------------|
    #|--- Make sure is array ---|
    #|--------------------------|
    singlePt = False
    if len(coord.shape) == 1:
        singlePt = True
        coord = coord.reshape([2,1])
        
    dtor = np.pi / 180.
    halfpi = np.pi / 2
    cx = cunit2rad[my_wcs['cunit'][0].lower()]
    cy = cunit2rad[my_wcs['cunit'][1].lower()]

    #|----------------------------|
    #|--- Check pixel location ---|
    #|----------------------------|
    # If within 3 deg of sun switch to quick proj instead of full
    # Setting force_proj to True will overwrite this
    if not force_proj:
        if my_wcs['coord_type'] == 'Helioprojective-Radial':
            ymm = np.array([np.min(coord[1,:]), np.max(coord[1,:])])
            yrange = (ymm + my_wcs['crval'][1]) * cy + halfpi
            if np.max(np.abs(yrange)) <= 3 * dtor: 
                doQuick = True
        else:
            xmm = np.array([np.min(coord[0,:]), np.max(coord[0,:])])
            xxrange = (xmm + my_wcs['crval'][0]) * cx
            ymm = np.array([np.min(coord[1,:]), np.max(coord[1,:])])
            yrange = (ymm + my_wcs['crval'][1]) * cy 
            if (np.max([np.max(np.abs(xxrange)), np.max(np.abs(yrange))])) <= 3 * dtor:
                doQuick = True

    #|--------------------------|
    #|--- Quick proj version ---|
    #|--------------------------|
    if doQuick and not force_proj:
        if my_wcs['coord_type'] == 'Helioprojective-Radial':
            x = coord[0,:] * cx
            y = (coord[1,:] + my_wcs['crval'][1]) * cy + halfpi
            coord[0,:] = my_wcs['crval'][0] + np.arctan2(x,y) / cx
            coord[1,:] = (np.sqrt(x**2 + y**2) - halfpi) / cy
            return coord
        else:
            coord[0,:] = coord[0,:] #+ my_wcs['crval'][0]         
            coord[1,:] = coord[1,:] #+ my_wcs['crval'][1]
            return coord
        
    #|--------------------------|
    #|--- Start full version ---|
    #|--------------------------|
    # assume standard phi0, theta0 for now
    phi0, theta0 = 0. * dtor, 90. * dtor
    # Get the celestial longitude and latitude of the fiducial point.
    alpha0 = my_wcs['crval'][0] * cx
    delta0 = my_wcs['crval'][1] * cy
    
    # Get the native long of the celestial pole
    # assuming default values for now
    if delta0 > theta0:
        phip = 0
    else:
        phip = 180 * dtor
        
    # Calculate native spherical coords
    phi = np.arctan2(cx*coord[0,:], -cy*coord[1,:])
    theta = np.sqrt((cx*coord[0,:])**2 + (cy*coord[1,:])**2)
    # Correct the theta
    w0 = np.where(theta == 0)[0]
    w1 = np.where(theta != 0)[0]
    theta[w0] = halfpi
    theta[w1] = np.arctan(1 / theta[w1])

    # Calculate the celestial spherical coordinates
    if delta0 >= halfpi:
        alpha = alpha0 + phi - phip - np.pi
        delta = theta
    elif delta0 <= -halfpi:
        alpha = alpha0 - phi + phip
        delta = -theta
    else:
        dphi = phi - phip
        cos_dphi = np.cos(dphi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        alpha = alpha0 + np.arctan2(-cos_theta*np.sin(dphi), sin_theta*np.cos(delta0) - cos_theta*np.sin(delta0)*cos_dphi)
        delta = np.arcsin(sin_theta*np.sin(delta0) + cos_theta * np.cos(delta0) * cos_dphi)
        
    # Convert back to og units
    coord[0,:] = alpha / cx
    coord[1,:] = delta / cy
    
    if singlePt:
        coord = coord.flatten()
    
    return coord

#|----------------------------|
#|--- Proj TAN to intermed ---|
#|----------------------------|
def wcs_inv_proj_tan(my_wcs, coord, doQuick=False, force_proj=False):
    """
    Function to project TAN coords into intermediate coordinates 
    
    Input:
        my_wcs: a wcs style dictionary

        coord: a list of TAN coord pairs (e.g. [[alp1,del1], [alp2,del2], ...])     
    
    Optional Input:
        doQuick: flag to force the quick version of the calculation instead
                 of doing the full more accurate one (defaults to False)
    
        force_proj: flag to force the full projection calculation insted of 
                    doing the quick, even if check suggests quick is 
                    sufficient (defaults to False)

    Output:
        coord: the coordinates converted to intermediate values
               intermed = relative to ref pix but crval not applied
               (same units as input e.g. arc, rad, deg ) 
    

    """
    #|--------------------------|
    #|--- Make sure is array ---|
    #|--------------------------|
    # Check shape of input array
    singlePt = False
    if len(coord.shape) == 1:
        singlePt = True
        coord = coord.reshape([2,1])
        
    #|----------------------------|
    #|--- Pull values from hdr ---|
    #|----------------------------|
    dtor = np.pi / 180.
    halfpi = np.pi / 2
    cx = cunit2rad[my_wcs['cunit'][0].lower()]
    cy = cunit2rad[my_wcs['cunit'][1].lower()]
    
    # Quick version
    if not force_proj:
        if my_wcs['coord_type'] == 'Helioprojective-Radial':
            ymm = np.array([np.min(coord[1,:]), np.max(coord[1,:])])
            yrange = (ymm + my_wcs['crval'][1]) * cy + halfpi
            if np.max(np.abs(yrange)) <= 3 * dtor: 
                doQuick = True
        else:
            xmm = np.array([np.min(coord[0,:]), np.max(coord[0,:])])
            xxrange = (xmm + my_wcs['crval'][0]) * cx
            ymm = np.array([np.min(coord[1,:]), np.max(coord[1,:])])
            yrange = (ymm + my_wcs['crval'][1]) * cy 
            if (np.max([np.max(np.abs(xxrange)), np.max(np.abs(yrange))])) <= 3 * dtor:
                doQuick = True
    
    #|--------------------------|
    #|--- Quick proj version ---|
    #|--------------------------|
    if doQuick and not force_proj:
        if my_wcs['coord_type'] == 'Helioprojective-Radial':
            r = coord[1,:] * cy + halfpi
            theta = (coord[0,:] - wcs['crval'][0]) * cx
            coord[0,:] = r * np.sin(theta) / cx
            coord[1,:] = r * np.cos(theta) / cy            
            return coord
        else:
            coord[0,:] = coord[0,:] - my_wcs['crval'][0]         
            coord[1,:] = coord[1,:] - my_wcs['crval'][1]
            return coord
            
    #|--------------------------|
    #|--- Start full version ---|
    #|--------------------------|
    phi0 = 0.
    theta0 = 90.
    if 'proj_names' in my_wcs:
        for item in my_wcs['proj_names']:
            if item == 'PV1_1':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                phi0 = my_wcs['proj_values'][idx[0]]
            if item == 'PV1_2':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                phi0 = my_wcs['proj_values'][idx[0]]
    if (phi0 != 0) or (theta0 != 90):
        print ('Non-standard PVi_1 and/or PVi_2 values -- ignored') # so why did we bother checking?
    
    # Convert to rads
    phi0, theta0 = phi0 * dtor, theta0 * dtor

    # Get the celestial longitude and latitude of the fiducial point.
    alpha0, delta0 = my_wcs['crval'][0] * cx , my_wcs['crval'][1] * cy
    
    phip = 180
    if delta0 > theta0: phip = 0
    if 'proj_names' in my_wcs:
        if 'LONPOLE' in my_wcs['proj_names']:
            phip = my_wcs['proj_values'][np.where(my_wcs['proj_names'] == 'LONPOLE')][0]
        if 'PV1_3' in my_wcs['proj_names']:
            phip = my_wcs['proj_values'][np.where(my_wcs['proj_names'] == 'PV1_3')][0]
    if (phip != 180) & (delta0 != halfpi):
        print('Non standard LONPOLE value')
    
    phip = dtor * phip
    
    # Convert from celestial to native spherical coordinates.
    alpha = cx * coord[0,:]
    delta = cy * coord[1,:]
    dalpha = alpha - alpha0
    cos_dalpha = np.cos(dalpha)
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    phi = phip + np.arctan2(-cos_delta * np.sin(dalpha), sin_delta * np.cos(delta0) - cos_delta * np.sin(delta0) * cos_dalpha)
    theta = np.arcsin(sin_delta * np.sin(delta0) + cos_delta * np.cos(delta0) * cos_dalpha)
    # Calculate the relative coords
    r_theta = np.copy(theta)
    w_good = np.where(r_theta > 0)
    r_theta[w_good] = 1. / np.tan(theta[w_good])
    x = r_theta * np.sin(phi)
    y = -r_theta * np.cos(phi)
    
    # Convert back to og units
    coord[0,:] = x / cx    
    coord[1,:] = y / cy
    
    if singlePt:
        coord = coord.flatten()
    
    
    return coord
    
#|----------------------------|
#|--- Proj intermed to AZP ---|
#|----------------------------|
def wcs_proj_azp(my_wcs, coord):
    """
    Function to project intermediate coordinates into AZP coords
    
    Input:
        my_wcs: a wcs style dictionary
    
        coord: a list of intermediate coord pairs (e.g. [[x1,y1], [x2,y2], ...])
               intermed = relative to ref pix but crval not applied
    
    Output:
        coord: the coordinates converted to AZP projection 
               (same units as input e.g. arc, rad, deg ) 

    """
    dtor = np.pi / 180.
    halfpi = np.pi / 2
    cx = cunit2rad[my_wcs['cunit'][0]]
    cy = cunit2rad[my_wcs['cunit'][1]]
    phi0 = 0.
    theta0 = 90.

    #|----------------------------|
    #|--- Pull values from hdr ---|
    #|----------------------------|
    if 'proj_names' in my_wcs:
        for item in my_wcs['proj_names']:
            if item == 'PV1_1':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                phi0 = my_wcs['proj_values'][idx[0]]
            if item == 'PV1_2':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                theta0 = my_wcs['proj_values'][idx[0]]
    if (phi0 != 0) or (theta0 != 90):
        print ('Non-standard PVi_1 and/or PVi_2 values -- ignored') # so why did we bother checking?
    
    # Convert to rads
    phi0, theta0 = phi0 * dtor, theta0 * dtor
    
    mu = 0.
    gamma = 0.
    if 'proj_names' in my_wcs:
        for item in my_wcs['proj_names']:
            if item == 'PV2_1':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                mu = my_wcs['proj_values'][idx[0]]
            if item == 'PV2_2':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                gamma = my_wcs['proj_values'][idx[0]]
    
    # Convert gamma to radians
    gamma = gamma  * dtor
    
    # Get the celestial longitude and latitude of the fiducial point.
    alpha0, delta0 = my_wcs['crval'][0] * cx , my_wcs['crval'][1] * cy
    
    phip = 180
    if delta0 > theta0: phip = 0
    if 'proj_names' in my_wcs:
        if 'LONPOLE' in my_wcs['proj_names']:
            phip = my_wcs['proj_values'][np.where(my_wcs['proj_names'] == 'LONPOLE')][0]
        if 'PV1_3' in my_wcs['proj_names']:
            phip = my_wcs['proj_values'][np.where(my_wcs['proj_names'] == 'PV1_3')][0]
    if (phip != 180) & (delta0 != halfpi):
        print('Non standard LONPOLE value')
    
    phip = dtor * phip
    
    #|------------------------|
    #|--- Full Calculation ---|
    #|------------------------|
    # Calculate the native spherical coords
    phi = np.arctan2(cx*coord[0,:], -cy*coord[1,:])
    r_theta = np.sqrt((cx*coord[0,:])**2 + (cy*coord[1,:]*np.cos(gamma))**2)
    if gamma == 0:
        rho = r_theta / (mu + 1)
    else:
        rho = r_theta / (mu + 1 + cy*coord[1,:]*np.sin(gamma))
    psi = np.atan(1/rho)
    omega = rho * mu / np.sqrt(rho**2 + 1)
    # check for values outside +/- 1
    badIdx = np.where(np.abs(omega) > 1)
    # just replace with signed 1, not sure exactly what IDL does
    omega[badIdx] = np.sign(omega[badIdx])
    omega = np.arcsin(omega)
    theta = psi - omega   
    badIdx = np.where(theta > np.pi)
    theta[badIdx] = theta[badIdx] - 2*np.pi
    theta2 = psi + omega + np.pi
    badIdx = np.where(theta2 > np.pi)
    theta2[badIdx] = theta2[badIdx] - 2*np.pi
    badIdx = np.where((np.abs(theta2-halfpi)< np.abs(theta-halfpi)) & (np.abs(theta2 < halfpi)))
    badIdx2 = np.where(np.abs(theta) > halfpi)
    theta[badIdx] = theta2[badIdx]
    theta[badIdx2] = theta2[badIdx2]
    
    
    # Calculate the celestial spherical coordinates
    if delta0 > halfpi:
        alpha = alpha0 + phi - phip - np.pi
        delta = theta
    elif delta0 < -halfpi:
        alpha = alpha0 - phi + phip
        delta = -theta
    else:
        dphi = phi - phip
        cos_dphi = np.cos(dphi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        alpha = alpha0 + np.arctan(-cos_theta * np.sin(dphi) / (sin_theta * np.cos(delta0) - cos_theta * np.sin(delta0) * cos_dphi))
        delta = np.arcsin(sin_theta * np.sin(delta0) + cos_theta * np.cos(delta0) * cos_dphi)
    
    # Convert back into og units    
    coord[0,:] = alpha / cx
    coord[1,:] = delta / cy
    return coord

#|----------------------------|
#|--- Proj AZP to intermed ---|
#|----------------------------|
def wcs_inv_proj_azp(my_wcs, coord):
    """
    Function to project AZP coordinates into intermediate coords
    
    Input:
        my_wcs: a wcs style dictionary
    
        coord: a list of AZP coord pairs (e.g. [[alp1,del1], [alp2,del2], ...])     
    
    Output:
        coord: the coordinates converted to intermediate values
               intermed = relative to ref pix but crval not applied
               (same units as input e.g. arc, rad, deg ) 

    """
    #|--------------------------|
    #|--- Make sure is array ---|
    #|--------------------------|
    # Check shape of input array
    singlePt = False
    if len(coord.shape) == 1:
        singlePt = True
        coord = coord.reshape([2,1])
    
    
    dtor = np.pi / 180.
    halfpi = np.pi / 2
    cx = cunit2rad[my_wcs['cunit'][0]]
    cy = cunit2rad[my_wcs['cunit'][1]]
    
    #|----------------------------|
    #|--- Pull values from hdr ---|
    #|----------------------------|
    phi0 = 0.
    theta0 = 90.
    if 'proj_names' in my_wcs:
        for item in my_wcs['proj_names']:
            if item == 'PV1_1':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                phi0 = my_wcs['proj_values'][idx[0]]
            if item == 'PV1_2':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                phi0 = my_wcs['proj_values'][idx[0]]
    if (phi0 != 0) or (theta0 != 90):
        print ('Non-standard PVi_1 and/or PVi_2 values -- ignored') # so why did we bother checking?
    
    # Convert to rads
    phi0, theta0 = phi0 * dtor, theta0 * dtor
    
    mu = 0.
    gamma = 0.
    if 'proj_names' in my_wcs:
        for item in my_wcs['proj_names']:
            if item == 'PV2_1':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                mu = my_wcs['proj_values'][idx[0]]
            if item == 'PV2_2':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                gamma = my_wcs['proj_values'][idx[0]]
    
    #|------------------------|
    #|--- Full Calculation ---|
    #|------------------------|
    # Convert gamma to radians
    gamma = gamma  * dtor
    
    # Get the celestial longitude and latitude of the fiducial point.
    alpha0, delta0 = my_wcs['crval'][0] * cx , my_wcs['crval'][1] * cy
    
    phip = 180
    if delta0 > theta0: phip = 0
    if 'proj_names' in my_wcs:
        if 'LONPOLE' in my_wcs['proj_names']:
            phip = my_wcs['proj_values'][np.where(my_wcs['proj_names'] == 'LONPOLE')][0]
        if 'PV1_3' in my_wcs['proj_names']:
            phip = my_wcs['proj_values'][np.where(my_wcs['proj_names'] == 'PV1_3')][0]
    if (phip != 180) & (delta0 != halfpi):
        print('Non standard LONPOLE value')
    
    phip = dtor * phip

    # Convert from celestial to native spherical coords
    alpha = cx * coord[0,:]
    delta = cy * coord[1,:]
    dalpha = alpha - alpha0
    cos_dalpha = np.cos(dalpha)
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    phi = phip + np.arctan2(-cos_delta * np.sin(dalpha), sin_delta * np.cos(delta0) - cos_delta * np.sin(delta0) * cos_dalpha)
    theta = np.arcsin(sin_delta * np.sin(delta0) + cos_delta * np.cos(delta0) * cos_dalpha)
    
    # Determine the latitude of divergence
    if mu == 0:
        thetax = 0
    elif np.abs(mu) > 1:
        thetax = np.arcsin(-1/mu)
    else:
        thetax = np.arcsin(-mu)
    
    # Calculate the relative coords
    cos_theta = np.cos(theta)
    if gamma == 0:
        denom = mu + np.sin(theta)
    else:
        denom = mu + np.sin(theta) + cos_theta * np.cos(phi) * np.tan(gamma)
    
    w_good = np.where(denom !=0) or np.where(theta <=thetax) # inverted from IDL bc want good
    if len(w_good[0]) > 0:
        theta[w_good[0]] = (mu + 1) * cos_theta[w_good] / denom[w_good]
    x = theta * np.sin(phi)
    
    if gamma == 0:
        y = -theta * np.cos(phi)
    else:
        y = -theta * np.cos(phi) / np.cos(gamma)
        
    # Convert back to og units
    coord[0,:] = x / cx    
    coord[1,:] = y / cy
    
    if singlePt:
        coord = coord.flatten()
    
    return coord

#|----------------------------|
#|--- Proj intermed to ZPN ---|
#|----------------------------|
def wcs_proj_zpn(my_wcs, coord):
    """
    Function to project ZPN coordinates into intermediate coords
    
    Input:
        my_wcs: a wcs style dictionary
    
        coord: a list of intermediate coord pairs (e.g. [[x1,y1], [x2,y2], ...])
               intermed = relative to ref pix but crval not applied
    
    Output:
        coord: the coordinates converted to ZPN projection 
               (same units as input e.g. arc, rad, deg ) 

    """
    dtor = np.pi / 180.
    halfpi = np.pi / 2

    #|----------------------------|
    #|--- Pull values from hdr ---|
    #|----------------------------|
    cx = cunit2rad[my_wcs['cunit'][0]]
    cy = cunit2rad[my_wcs['cunit'][1]]
    
    phi0 = 0.
    theta0 = 90.
    if 'proj_names' in my_wcs:
        for item in my_wcs['proj_names']:
            if item == 'PV1_1':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                phi0 = my_wcs['proj_values'][idx[0]]
            if item == 'PV1_2':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                theta0 = my_wcs['proj_values'][idx[0]]
    if (phi0 != 0) or (theta0 != 90):
        print ('Non-standard PVi_1 and/or PVi_2 values -- ignored') # so why did we bother checking?
    
    #|------------------------|
    #|--- Full Calculation ---|
    #|------------------------|
    # Convert to rads
    phi0, theta0 = phi0 * dtor, theta0 * dtor
    
    # Get the polynomial coefficients
    holdMypp = np.zeros(21)
    if 'proj_names' in my_wcs:
        for i in range(21):
            name = 'PV'+str(my_wcs['iy']+1)+'_'+str(i)
            idxs = np.where(my_wcs['proj_names'] == name)[0]
            if len(idxs) > 0:
                holdMypp[i] = my_wcs['proj_values'][idxs[0]]
    holdMypp = np.array(holdMypp)
    haspp = np.where(holdMypp !=0)[0]
    if len(haspp) > 0:
        n = haspp[-1]
    else:
        print('No polynomial coordinates specified')

    pp = holdMypp[:n+1]
    pderiv = (pp*range(n+1))[1:-1]
    
    # Get the celestial lon/lat of the fiducial point
    alpha0 = my_wcs['crval'][my_wcs['ix']] * cx
    delta0 = my_wcs['crval'][my_wcs['iy']] * cy
    
    # Get the native longitude of the celesital pole
    phip = 180.
    if delta0 >= theta0: phip = 0.
    if 'proj_names' in my_wcs:
        idx = np.where(my_wcs['proj_names'] == 'LONPOLE')[0]
        if len(idx) > 0:
            phip = my_wcs['proj_values'][idx[0]]
        name = 'PV'+str(my_wcs['ix']+1)+'_3'
        idx = np.where(my_wcs['proj_names'] == name)[0]    
        if len(idx) > 0:
            phip = my_wcs['proj_values'][idx[0]]
    
    if (phip!=180.) & (delta0 !=halfpi):
        print('Non standard LONPOLE value '+ str(phip))
    phip = phip * dtor
    
    # Calculate the native spherical coordinates
    phi = np.arctan2(cx*coord[0,:], -cy*coord[1,:])
    r_theta = np.sqrt((cx*coord[0,:])**2 + (cy*coord[1,:])**2)
    # Reiteratively solve for gamma
    tolerance = 1e-8
    max_iter = 1000
    gamma = np.arctan(r_theta)
    n_iter = 0
    
    while n_iter < max_iter:
        n_iter += 1
        diff = (np.polyval(pp[::-1], gamma) - r_theta) / np.polyval(pderiv[::-1], gamma)
        gamma = gamma - diff
        if np.max(np.abs(diff)) < tolerance: n_iter = max_iter+1
    
    theta = halfpi - gamma
    w_missing = np.where(theta < - halfpi)
    
    # Calculate the celestial spherical coordinates
    if delta0 >= halfpi:
        alpha = alpha0 + phi - phip - np.pi
        delta = theta
    elif delta0 <= -halfpi:
        alpha = alpha0 - phi + phip
        delta = -theta
    else:
        dphi = phi - phip
        cos_dphi = np.cos(dphi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        alpha = alpha0 + np.arctan2(-cos_theta * np.sin(dphi), sin_theta * np.cos(delta0) - cos_theta * np.sin(delta0) * cos_dphi)
        delta = np.arcsin(sin_theta * np.sin(delta0) + cos_theta * np.cos(delta0) * cos_dphi)

    # Convert back to original units
    coord[0,:] = alpha / cx
    coord[1,:] = delta / cy
    
    # Ignoring missing
    return coord

#|----------------------------|
#|--- Proj ZPN to intermed ---|
#|----------------------------|
def wcs_inv_proj_zpn(my_wcs, coord):
    """
    Function to project AZP coordinates into intermediate coords
    
    Input:
        my_wcs: a wcs style dictionary
    
        coord: a list of ZPN coord pairs (e.g. [[alp1,del1], [alp2,del2], ...])     
    
    Output:
        coord: the coordinates converted to intermediate values
               intermed = relative to ref pix but crval not applied
               (same units as input e.g. arc, rad, deg ) 

    """
    #|--------------------------|
    #|--- Make sure is array ---|
    #|--------------------------|
    # Check shape of input array
    singlePt = False
    if len(coord.shape) == 1:
        singlePt = True
        coord = coord.reshape([2,1])
    
    dtor = np.pi / 180.
    halfpi = np.pi / 2


    #|----------------------------|
    #|--- Pull values from hdr ---|
    #|----------------------------|
    cx = cunit2rad[my_wcs['cunit'][0]]
    cy = cunit2rad[my_wcs['cunit'][1]]
    
    phi0 = 0.
    theta0 = 90.
    if 'proj_names' in my_wcs:
        for item in my_wcs['proj_names']:
            if item == 'PV1_1':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                phi0 = my_wcs['proj_values'][idx[0]]
            if item == 'PV1_2':
                idx = np.where(my_wcs['proj_names'] == item)[0]
                theta0 = my_wcs['proj_values'][idx[0]]
    if (phi0 != 0) or (theta0 != 90):
        print ('Non-standard PVi_1 and/or PVi_2 values -- ignored') # so why did we bother checking?
    
    #|------------------------|
    #|--- Full Calculation ---|
    #|------------------------|
    # Convert to rads
    phi0, theta0 = phi0 * dtor, theta0 * dtor
       
    # Get the polynomial coefficients
    holdMypp = np.zeros(21)
    if 'proj_names' in my_wcs:
        for i in range(21):
            name = 'PV'+str(my_wcs['iy']+1)+'_'+str(i)
            idxs = np.where(my_wcs['proj_names'] == name)[0]
            if len(idxs) > 0:
                holdMypp[i] = my_wcs['proj_values'][idxs[0]]
                #if printIt:
                #    print(name, holdMypp[i])
    holdMypp = np.array(holdMypp)
    haspp = np.where(holdMypp !=0)[0]
    if len(haspp) > 0:
        n = haspp[-1]
    else:
        print('No polynomial coordinates specified')

    pp = holdMypp[:n+1]
    
    
    # Get the celestial lon/lat of the fiducial point
    alpha0 = my_wcs['crval'][my_wcs['ix']] * cx
    delta0 = my_wcs['crval'][my_wcs['iy']] * cy        
    
    # Get the native longitude of the celesital pole
    phip = 180.
    if delta0 >= theta0: phip = 0.
    if 'proj_names' in my_wcs:
        idx = np.where(my_wcs['proj_names'] == 'LONPOLE')[0]
        if len(idx) > 0:
            phip = my_wcs['proj_values'][idx[0]]
        name = 'PV'+str(my_wcs['ix']+1)+'_3'
        idx = np.where(my_wcs['proj_names'] == name)[0]    
        if len(idx) > 0:
            phip = my_wcs['proj_values'][idx[0]]
    
    if (phip!=180.) & (delta0 !=halfpi):
        print('Non standard LONPOLE value '+ str(phip))
    phip = phip * dtor
    
    # Convert from celestial to native spherical coords
    alpha = cx * coord[0,:]
    delta = cy * coord[1,:]
    
    dalpha = alpha - alpha0
    cos_dalpha = np.cos(dalpha)
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    phi = phip + np.arctan2(-cos_delta * np.sin(dalpha), sin_delta * np.cos(delta0) - cos_delta * np.sin(delta0) * cos_dalpha)
    theta = np.arcsin(sin_delta * np.sin(delta0) + cos_delta * np.cos(delta0) * cos_dalpha)
    
    # Calculate the relative coordinates
    r_theta = np.polyval(pp[::-1], halfpi-theta)
    x = r_theta * np.sin(phi)
    y = -r_theta * np.cos(phi)
    coord[0,:] = x / cx    
    coord[1,:] = y / cy    
    
    if singlePt:
        coord = coord.flatten()
    return coord

#|----------------------------|
#|--- Get coord of a pixel ---|
#|----------------------------|
def wcs_get_coord(my_wcs, pixels=None):
    """
    Function to get wcs coordinates from pixel values. It can be
    passed a list of specific pixel values or will calculated all
    the values for the full image grid
    
    Input:
        my_wcs: a wcs style dictionary
        
    Optional Input:
        pixels: a list of specific pixel values to use in the calc
                (defaults to None and does full grid calc)

    Output:
        im: the total brightness calculated from the pol images 
            if doPB is flagged it returns the polarization brightness
            if doPolAng is flagged it returns the polarized angle
    
        hdr: the corresponding header 
    
    Notes:
        The code assumes an appropriate set of images are passed

    """
    # Assuming an appropriate header
    # ignoring distortion, associate, apply for now (139-152)
    
    #|---------------------------------|
    #|--- Sort out pixels/full grid ---|
    #|---------------------------------|
    # Get the dimensions/indices
    naxis = my_wcs['naxis']
    n_axis = len(naxis) # these var names bother me but following idl
    naxis1 = naxis[0]
    naxis2 = naxis[1]
    # assuming not pixel list (158-195)
    if type(pixels) != type(None):
        if len(pixels.shape) == 1:
            pixels = pixels.reshape([2,1])
        # assume shaped fine for multipoints?
        coord = pixels
    else:
        num_elements = np.prod(naxis)
        index = np.arange(num_elements).astype(int)
        coord = np.empty([n_axis, num_elements])
        coord[0,:] = index  % naxis1
        coord[1,:] = (index / naxis1 % naxis2).astype(int)

    # Skipping distortion
    
    #|-------------------------------|
    #|--- Get intermediate coords ---|
    #|-------------------------------|
    # Adjust relative to CRPIX
    crpix = my_wcs['crpix']
    coord[0,:] = coord[0,:] - (crpix[0] -1)
    coord[1,:] = coord[1,:] - (crpix[1] -1)
        
    # Skipping distortion/associate/pixel-list (218-234)
    
    # Calcualte immedate (relative coordinates)
    # Assuming were doing pc
    coord = np.matmul(my_wcs['pc'], coord)
 
    # Skipping more distortion (264 - 284)
    coord[0,:] = coord[0,:]*my_wcs['cdelt'][0]
    coord[1,:] = coord[1,:]*my_wcs['cdelt'][1]
    # Skipping more distortion (288-301)
    
    # Assume we don't just want relative proj
    
    # Projection table - assume we don't need for now but check and bail if so
    ctypes = my_wcs['ctype']
    for item in ctypes:
        if '-TAB' in item:
            print ('Need to do wcs_proj_tab but havent ported yet')
            print (Quit)
    
    # Assume we dont hit any of the weird cases of proj and crval already dealt with (319-350)
    
    #|------------------------|
    #|--- Apply projection ---|
    #|------------------------|
    proj = my_wcs['projection']
    if proj == 'TAN':
        coord = wcs_proj_tan(my_wcs, coord)
    elif proj == 'AZP':
        coord = wcs_proj_azp(my_wcs, coord)
    elif proj == 'ZPN':
        coord = wcs_proj_zpn(my_wcs, coord)
    else:
        print('Other projections not yet ported including '+proj)
        print(Quit)
       
    # Skipping projextion, pos_long, nowrap since not hit in simple version
    
    
    #|-----------------------|
    #|--- Reformat Output ---|
    #|-----------------------|
    if type(pixels) != type(None):
        coord.reshape(pixels.shape)
    else:
        coord = coord.reshape([2, naxis2, naxis1])

    return coord    
    
    
#|----------------------------|
#|--- Get pixel of a coord ---|
#|----------------------------|
def wcs_get_pixel(my_wcs, coord,  doQuick=False, force_proj=False, noPC=False):
    """
    Function to get pixel values for wcs coordinates. 
    
    Input:
        my_wcs: a wcs style dictionary
    
        coord: a list of coordinates in the projection type of my_wcs
        
    Optional Input:
        doQuick: flag to force the quick version of the calculation instead
                 of doing the full more accurate one (defaults to False)
    
        force_proj: flag to force the full projection calculation insted of 
                    doing the quick, even if check suggests quick is 
                    sufficient (defaults to False)
    
        noPC: flag to not to use the PC matrix in calculating the coords
              (defaults to False )

    Output:
        coord: the pixel values of the coordinates
     
    """
    #|---------------------|
    #|--- Process Input ---|
    #|---------------------|
    if isinstance(coord, list):
        coord = np.array(coord)
    # Check shape of input array
    singlePt = False
    if len(coord.shape) == 1:
        singlePt = True
        coord = coord.reshape([2,1])
        
    #|--------------------------|
    #|--- Inverse Projection ---|
    #|--------------------------|
    if my_wcs['projection'] == 'TAN':
        outpix = wcs_inv_proj_tan(my_wcs, coord, doQuick=doQuick)
    elif my_wcs['projection'] == 'AZP':
        outpix = wcs_inv_proj_azp(my_wcs, coord)
    elif my_wcs['projection'] == 'ZPN':
        outpix = wcs_inv_proj_zpn(my_wcs, coord)
    else:
        print ('Other projections not ported')
        print (Quit)

    #|-----------------------------------|
    #|--- Convert from intermed coord ---|
    #|-----------------------------------|
    # Skipping subtract ref values for non-spherical and de-app tablular
    coord[0,:] = outpix[0,:] / my_wcs['cdelt'][0]
    coord[1,:] = outpix[1,:] / my_wcs['cdelt'][1]
    
    #temp = np.copy(coord)
    pc = my_wcs['pc']
    
    
    if not noPC:
        pc = my_wcs['pc']
        pci  = np.linalg.inv(pc)
        coord = np.matmul(pci, coord)
    
    # Add in reference pixel
    coord[0,:] = coord[0,:] + my_wcs['crpix'][0] -1
    coord[1,:] = coord[1,:] + my_wcs['crpix'][1] -1
    
    if singlePt:
        coord = coord.flatten()
    return coord
        
    
    
    
    
    