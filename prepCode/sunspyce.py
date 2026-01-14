"""
Module for functions related to SPICE kernels. The spiceypy
package already exists and performs most of the IDL SPICE
operations so we just port the portions that interface with
spiceypy. The function names match the IDL versions but are
spycy instead of spicy 

"""

import numpy as np
import sys
from sunpy.time import parse_time
import os
import spiceypy as spice
import math
from sunpy.coordinates import sun


#|--------------------------------|
#|--- Global for s/c ID number ---|
#|--------------------------------|
global scDict
scDict = {'sta':'-234', 'stereoa':'-234', 'stereoahead':'-234', 'stb':'-235', 'stereob':'-235', 'stereobehind':'-235', 'solo':'-144', 'solarorbiter':'-144', 'psp':'-96','parkersolarprobe':'-96', 'EARTH':'399', 'Earth':'399', 'earth':'399'}

#|-----------------------------------------------|
#|--- Calc helioprojective cartesian pointing ---|
#|-----------------------------------------------|
def get_sunspyce_hpc_point(date, spacecraft, instrument=None, doDeg=False, doRad=False):
    """
    Function to get the helioprojective cartesian pointing of a 
    spacecraft using sunspyce routines. This returns the yaw,
    pitch, and roll angle with the first two in arcsec and the 
    last in degrees unless doDeg or doRad is flagged 
    
    Input:
        date: a date string in a format suitable for spiceypy.str2et
              (this is fairly flexible in formatting)
    
        spacecraft: a string tag representing the spacecraft of interest
                    (current options are sta, stb, solo, psp, earth)
    
    Optional Input:
        instrument: option to pass a specific instrument string that
                    will be used by get_sunspyce_cmat
                    (defaults to None)
    
        doDeg: return all values in degrees
    
        doRad: return all values in radians

    Output:
        pointing: a vector with [yaw, pitch, roll] as [arcsec, arcsec, deg]

    """
    # returns yaw (arcsec), pitch (arcsec), and roll angle (deg)
    # If doDeg then all three params returned in deg
    
    #|--------------------|
    #|--- Set up units ---|
    #|--------------------|
    roll_units = 180 / np.pi
    xy_units   = roll_units * 3600
    if doDeg: xy_units = roll_units
    if doRad:
        roll_units = 1
        xy_units   = 1
        
    #|--------------------------|
    #|--- Get s/c spice code ---|
    #|--------------------------|
    if spacecraft.lower() in scDict:
        sc = scDict[spacecraft.lower()]
    else:
        sys.exit(spacecraft.lower() + ' not in scDict for sunspyce. Pick from sta, stb, solo, psp, earth.')
    
    sc_stereo = False
    if sc in ['-234', '-235']: sc_stereo = True
    
    #|----------------|
    #|--- Get cmat ---|
    #|----------------|
    cmat = get_sunspyce_cmat(date, spacecraft, system='HPC', instrument=instrument)

    # Skipping error stuff
    # don't need to predefine pointing
    
    halfpi = np.pi / 2.
    twopi  = np.pi * 2.
    
    if sc == '-96':
        cmat0 = np.matmul([[0,0,1],[-1,0,0],[0,-1,0]], cmat)
        roll, pitch, yaw = spice.m2eul(cmat0, 1,3,2)
        yaw = halfpi - yaw
        roll = roll + halfpi
        if np.abs(roll) > np.pi:
            roll = roll - math.copysign(twopi, roll)
    else:
        roll, pitch, yaw = spice.m2eul(cmat, 1,3,2)
        yaw = halfpi - yaw
        if sc == scDict['solo']:
            roll = roll + halfpi
        if sc == scDict['stb']:
            roll = roll + np.pi
            if roll > np.pi: roll = roll - twopi

    # Ignoring stereo post conjunction bc haven't used keyword
    
    #|--------------------------------|
    #|--- Correct ranges and units ---|
    #|--------------------------------|
    # correct any cases where pitch is greater than 90
    if np.abs(pitch) > halfpi:
        pitch = math.copysign(np.pi, pitch) - pitch
        yaw = yaw - math.copysign(np.pi, yaw)
        roll = roll - math.copysign(np.pi, roll)
        
    # Apply the units
    pointing = np.zeros(3)
    pointing[0] = yaw * xy_units
    pointing[1] = pitch * xy_units
    pointing[2] = -roll * roll_units   
    
    return pointing


#|-------------------------------|
#|--- Get c matrix from SPICE ---|
#|-------------------------------|
def get_sunspyce_cmat(date, spacecraft, system=None, instrument=None, tolerance=None, sixVec=False):
    """
    Function to get the camera matrix using spice
    
    Input:
        date: a date string in a format suitable for spiceypy.str2et
              (this is fairly flexible in formatting)
    
        spacecraft: a string tag representing the spacecraft of interest
                    (current options are sta, stb, solo, psp, earth)
    
    Optional Input:
        system: the coordinate sys (e.g. HEEQ, Carrington...)
    
        instrument: a specific instrument to use when getting cmat
    
        tolerance: a tolerance value for spice.ckgp to use
                   (defaults to None but ultimately set to 1000)
    
        sixVec: flag to return a 6x6 matrix instead of the standard 3x3
                (defaults to False)

    Output:
        ccmat: the camera matrix

    """
    #|--------------------------|
    #|--- Get s/c spice code ---|
    #|--------------------------|
    if spacecraft.lower() in scDict:
        sc = scDict[spacecraft.lower()]
    else:
        sys.exit(spacecraft.lower() + ' not in scDict for sunspyce. Pick from sta, stb, solo, psp.')
    
    sc_base = ''
    if sc == '-96':
        sc_base = 'SPP_SPACECRAFT'
    elif sc == '-144':
        sc_base = 'SOLO_SRF'
        
    #|------------------|
    #|--- Parse Time ---|
    #|------------------|
    time = parse_time(date).utc
    
    # Determine which coord system is specified
    # assuming single value for now 
    if system:
        system = system.upper()
        if system == 'HEQ': system = 'HEEQ'
        elif system in 'CARRINGTON': system = 'CARRINGTON'
    else:
        system == 'RTN'
        
    #|------------------------|
    #|--- Get coord system ---|
    #|------------------------|
    # Assume not passed frame bc don't give it the option yet...
    frame = None
    if system in ['HGRTN', 'RTN', 'HPC']:
        if sc == '-96': frame = 'PSPHGRTN'
        elif sc == '-144': frame = 'SOLOHGRTN'
        elif sc == '-234': frame = 'STAHGRTN'
        elif sc == '-235': frame = 'STBHGRTN'
        else:
            sys.exit('Cannot pull frame from sc in get_sunspyce_cmat')
    # ignoring the other systems
    
    # Determine the tolerance
    if tolerance:
        tol = tolerance
    else:
        tol = 1000
        
    # Determine if use ITRF93 kernels - skipping bc don't give keyword
    
    # Convert date/time to eph time and then to sc clock time
    et = spice.str2et(date)
    
    nVec = 3
    if sixVec:
        nVec = 6
    # again single time val for now
    cmat = np.zeros([nVec, nVec])
    
    sclkdp = spice.sce2c(int(sc), et)
    
    # Adding frcode that gets hit by roll GEI code 
    if not frame:
        if system == 'GEI':
            frame = 'J2000' 

    #|--------------------------|
    #|--- Pass to spice func ---|
    #|--------------------------|
    cmat, clkout = spice.ckgp(int(sc)*1000, sclkdp, tol, frame)
    
    #|-----------------------------|
    #|--- Modify for instrument ---|
    #|-----------------------------|
    if instrument:
        rotMat = spice.pxform(sc_base, instrument, et)
        if np.abs(np.linalg.det(rotMat) - 1) > 1e-5:
            sys.exit('Invalid rotation matrix for instrument')
        
        # Solar orbiter thing ignoring for now
        if sc == '-144':
            rotMat = np.matmul([[-1,0,0],[0,-1,0],[0,0,1]], rotMat)
        ccmat = np.matmul(rotMat, cmat)
        # Assume c matrix was found
    else:
        ccmat = cmat
    
    #|-----------------------|
    #|--- HPC adjustement ---|
    #|-----------------------|
    if system == 'HPC':
        ccmat = np.matmul(ccmat, [[0, 0, 1.], [1., 0, 0], [0, 1., 0]])

    # ignoring weird storing stuff
    return ccmat    
        
#|---------------------------|
#|--- Get spacecraft roll ---|
#|---------------------------|
def get_sunspyce_roll(date, spacecraft, system=None, instrument=None, doRad=False, tolerance=None):
    """
    Function to get the spacecraft roll using spice
    
    Input:
        date: a date string in a format suitable for spiceypy.str2et
              (this is fairly flexible in formatting)
    
        spacecraft: a string tag representing the spacecraft of interest
                    (current options are sta, stb, solo, psp, earth)
    
    Optional Input:
        system: the coordinate sys (e.g. HEEQ, Carrington...)
    
        instrument: a specific instrument to use when getting roll
    
        doRad: flag to return the result in radians

        tolerance: a tolerance value for spice.ckgp to use
                   (defaults to None but ultimately set to 1000)
    
    Output:
        roll: the spacecraft roll (in degrees unless doRad flagged)

        pitch: the spacecraft pitch (in degrees unless doRad flagged)

        yaw: the spacecraft yaw (in degrees unless doRad flagged)

    """
    
    # Assuming passed correct things
    units = 180. / np.pi
    
    #|----------------------|
    #|--- Process Inputs ---|
    #|----------------------|
    if doRad:
        units = 1.
        
    if spacecraft in scDict:
        sc = scDict[spacecraft]
    else:
        sys.exit('Spacecraft not in spice codes')    
    
    sc_stereo = sc in ['-234', '-235']
    
    if system:
        system = system.upper()
    else:
        system = 'RTN'
    
    #|----------------------|
    #|--- Calls to Spice ---|
    #|----------------------|
    cmat = get_sunspyce_cmat(date, spacecraft, system=system, instrument=instrument, tolerance=tolerance)
    roll, pitch, yaw = 0., 0., 0.
    twopi = np.pi * 2.
    halfpi = np.pi / 2.
    
    if sc == '-96':
        cmat0 = np.matmul([[0,0,1],[-1,0,0],[0,-1,0]], cmat)
        roll, pitch, yaw = spice.m2eul(cmat0, 1,2,3)
        roll = -roll
        pitch = -pitch
        if np.abs(roll) > np.pi:
            roll = roll - math.copysign(twopi, roll)
    else:
        roll, pitch, yaw = spice.m2eul(cmat, 1,2,3)
        pitch = - pitch
        if sc in ['-234', '-235']:
            roll = roll - halfpi
        if sc == '-235':
            roll = roll + np.pi
        if np.abs(roll) > np.pi:
            roll = roll - math.copysign(twopi, roll)

    # Skipping post conjuction
    
    #|------------------------|
    #|--- Ranges and Units ---|
    #|------------------------|
    # Correct any cases where pitch > 90 deg
    if np.abs(pitch) > halfpi:
        pitch = math.copysign(np.pi, pitch) - pitch
        yaw   = yaw - math.copysign(np.pi, yaw) 
        roll  = roll - math.copysign(np.pi, roll) 
    
    # Apply the units
    roll  = units * roll
    pitch = units * pitch
    yaw   = units * yaw        
        
    return roll, pitch, yaw

#|---------------------------|
#|--- Get distance of s/c ---|
#|---------------------------|
def get_sunspyce_coord(date, spacecraft, system=None, instrument=None, target=None, doMeters=False, doAU=False, doVelocity=True):
    """
    Function to get the orbital location of a spacecraft. Typically the
    sun is the center and the spacecraft is the target unless keywords
    are set otherwise
    
    Input:
        date: a date string in a format suitable for spiceypy.str2et
              (this is fairly flexible in formatting)
    
        spacecraft: a string tag representing the spacecraft of interest
                    (current options are sta, stb, solo, psp, earth)
    
    Optional Input:
        system: the coordinate sys (e.g. HEEQ, Carrington..., defaults to None)
    
        instrument: a specific instrument to use when getting the value (defaults to None)
    
        target: the point being measured to (typically the spacecraft, defaults to None)
    
        doMeters: flag to return the result in meters (defaults to False)
    
        doAU: flag to return the result in AU (defaults to False)
    
        doVelocity: flag to return the velocity vector in addition to the dist (defaults to True)
    
    Output:
        state: a length 3 array of the state (6 if the velocity is included)

    """
    #|----------------------|
    #|--- Process Inputs ---|
    #|----------------------|
    # Determine which spacecraft was requested and make it spicy    
    if spacecraft.lower() in scDict:
        sc = scDict[spacecraft.lower()]
    else:
        sys.exit(spacecraft.lower() + ' not in scDict for sunspyce. Pick from sta, stb, solo, psp, earth.')
    
    # Convert time to utc
    time = parse_time(date).utc
    
    # Convert date/time to eph time and then to sc clock time
    et = spice.str2et(date)
    
    # use instruments keyword if provided
    if type(instrument) != type(None):
        print ("need to code this part")
        print (Quit)
        
    # Determine which coordinate system was specified
    if type(system) == type(None):
        system = 'HCI'
    if system == 'HEQ': system = 'HEEQ'
    elif system in 'CARRINGTON': system = 'CARRINGTON'
    
    if system in ['HGRTN', 'RTN']:
        if sc == '-96': frame = 'PSPHGRTN'
        elif sc == '-144': frame = 'SOLOHGRTN'
        elif sc == '-234': frame = 'STAHGRTN'
        elif sc == '-235': frame = 'STBHGRTN'
        else:
            sys.exit('Cannot pull frame from sc in get_sunspyce_cmat')
            
    if system == 'SCI':
        print('Havent ported STEREO pointing frame code')
        print (Quit)
        
    if system == 'HERTN':
        print('Havent ported HERTN pointing frame code')
        print (Quit)
        
        
    # Assuming conic parameters aren't avail bc aren't in the test case
    
    # Assume not doing ITRF93 kernels for Earth
    
    #|----------------------------|
    #|--- Get state from spice ---|
    #|----------------------------|
    # Get the state and light travel time
    if system == 'HAE':
        if type(target) == type(None):
            target = sc
            center = 'Sun'
        else:
            target = target
            center = sc
        state, ltime = spice.spkezr(target, et, 'ECLIPJ2000','None', center)
    elif system == 'HCI':
        if type(target) == type(None):
            target = sc
            center = 'Sun'
        else:
            target = target
            center = sc                        
        state, ltime = spice.spkezr(target, et, 'HCI','None', center)
    elif system == 'HEE':
        if type(target) == type(None):
            target = sc
            center = 'Sun'
        else:
            target = target
            center = sc                        
        state, ltime = spice.spkezr(target, et, 'HEE','None', center)
    elif system == 'HEEQ':
        if type(target) == type(None):
            target = sc
            center = 'Sun'
        else:
            target = target
            center = sc
        state, ltime = spice.spkezr(target, et, 'HEEQ','None', center)
    elif system == 'CARRINGTON':
        if type(target) == type(None):
            target = sc
            center = 'Sun'
        else:
            target = target
            center = sc
        state, ltime = spice.spkezr(target, et, 'IAU_SUN','None', center)
            
        
    else:
        sys.exit('Other systems not ported yet')   
        
    # Assuming no times beyond the range (and we have no conics anyway) 
    if not doVelocity:
        state = state[:3]
        
    #|---------------------|
    #|--- Process Units ---|
    #|---------------------|
    # Units - spice res in km
    if doMeters:
        state = state * 1000
    elif doAU:
        state = state * 1.496e8
    
    return state

#|-------------------------------------|
#|--- Get Spherical location of s/c ---|
#|-------------------------------------|
def get_sunspyce_lonlat(date, spacecraft, system=None, instrument=None, target=None, doMeters=False, doAU=False, doDegrees=False, pos_long=False, lt_carr=False):
    """
    Function to get the spherical location of the spacecraft
    
    Input:
        date: a date string in a format suitable for spiceypy.str2et
              (this is fairly flexible in formatting)
    
        spacecraft: a string tag representing the spacecraft of interest
                    (current options are sta, stb, solo, psp, earth)
    
    Optional Input:
        system: the coordinate sys (e.g. HEEQ, Carrington..., defaults to None)
    
        instrument: a specific instrument to use when getting the value (defaults to None)
    
        target: the point being measured to (typically the spacecraft, defaults to None)
    
        doMeters: flag to return the result in meters (defaults to False)
    
        doAU: flag to return the result in AU (defaults to False)
    
        doDegrees: flag to return the velocity vector in addition to the dist (defaults to False)
    
        pos_long: flag to force longitude to positive values (defaults to False)
    
        lt_carr: flag to apply light time travel correction (defaults to False)
    
    Output:
        [rad, lat, long]: the spacecraft radius (km), latitude (radians), and longitude (radians)

    """

    #|----------------------|
    #|--- Process Inputs ---|
    #|----------------------|
    if type(system) != type(None):
        system = system.upper()
    else:
        system = 'HCI'
        
    if system == 'HEQ': system = 'HEEQ'
    elif system in 'CARRINGTON': system = 'CARRINGTON'
    if type(instrument) != type(None):
        system = ''
    
    if system == 'HPC':
        system = 'RTN'
        hpc_conv = True
    else:
        hpc_conv = False
        
    #|-----------------------|
    #|--- Get rect coords ---|
    #|-----------------------|
    # Call get_sunspice_coord
    state = get_sunspyce_coord(date, spacecraft, system=system, instrument=instrument, doMeters=doMeters, doAU=doAU, doVelocity=False)
   
    # Ignoring planetographic
    
    #|----------------------------|
    #|--- Convert to spherical ---|
    #|----------------------------|
    # Use reclat to convert rect coords into rad, lon, lat
    rad, lon, lat = spice.reclat(state)
    
    #|---------------------------|
    #|--- Various Corrections ---|
    #|---------------------------|
    # If HPC apply a correction to RTN coords
    if hpc_conv:
        print ('hit untested code, should double check')
        twopi = 2 * np.pi
        lon = np.pi - lon
        if lon > np.pi: lon = lon - twopi
        if lon < -np.pi: lon = lon + twopi
        
    # Adjust lon range if carrington or pos_long keyword 
    if (system == 'CARRINGTON') or pos_long:
        if lon < 0: lon = lon + 2 * np.pi
        
    # If Carrington or lt_carr set apply light-travel-time-correction
    if (system == 'CARRINGTON') & lt_carr:
        print ('hit untested code, should double check')
        conv = 1.
        if doMeters: conv = 1e-3
        elif doAU: conv = 1.4959787e08
        rsun = 695508.00 / conv
        dtime = (rad - rsun) * (conv / 299792.458)
        rate = 14.1844 * np.pi / (180. * 86400.)
        lon = lon + rate * dtime
    
    #|------------------------|
    #|--- Unit conversions ---|
    #|------------------------|
    # Conversion to degrees
    if doDegrees:
        lon = lon * 180. / np. pi
        lat = lat * 180. / np. pi
    return [rad, lon, lat]

#|-------------------------------|
#|--- Get the p0 angle of s/c ---|
#|-------------------------------|
def get_sunspyce_p0_angle(date, spacecraft, doDegrees=False):
    """
    Function to get the p0 angle using spice. This is the angle of the
    projection of solar north onto the plane of the sky relative to
    celestial north
    
    Input:
        date: a date string in a format suitable for spiceypy.str2et
              (this is fairly flexible in formatting)
    
        spacecraft: a string tag representing the spacecraft of interest
                    (current options are sta, stb, solo, psp, earth)
    
    Optional Input:
        doDegrees: flag to return result in degrees
    
    Output:
        p0: the p0 angle 


    """
    #|----------------------|
    #|--- Process inputs ---|
    #|----------------------|
    # Determine which spacecraft was requested and make it spicy    
    if spacecraft.lower() in scDict:
        sc = scDict[spacecraft.lower()]
    else:
        sys.exit(spacecraft.lower() + ' not in scDict for sunspyce. Pick from sta, stb, solo, psp, earth.')
    
    
    #|--------------------------------|
    #|--- Get solar axis direction ---|
    #|--------------------------------|
    # get the orientation of the sun axis in J2000
    et = spice.str2et(date)
    rota = spice.pxform('IAU_SUN', 'J2000', et)
    sun_north = rota[:,2]
    j2000_north = [0., 0., 1.]

    
    #|-----------------|
    #|--- Reproject ---|
    #|-----------------|
    # Assuming doing single date for now
    state, ltime = spice.spkezr('Sun', et, 'J2000','None', sc)
    rad = state[:3]
    rad = rad / np.sqrt(np.sum(rad**2))
    
    # Calc the parts of solar and j2000 that are perp to Sun-sc and renorm
    snproj = sun_north - np.sum(rad*sun_north) * rad
    snproj = snproj / np.sqrt(np.sum(snproj**2))
    jnproj = j2000_north - np.sum(rad*j2000_north) * rad
    jnproj = jnproj / np.sqrt(np.sum(jnproj**2))
    
    # Split the solar N proj ax into parts parallel and perp to J200 proj
    sxproj = np.sum(snproj * jnproj)
    vecproj = np.cross(j2000_north, rad)
    vecproj = vecproj / np.sqrt(np.sum(vecproj**2))
    syproj  = np.sum(snproj * vecproj)
    
    # Calculate the p0 angle
    p0 = np.atan2(syproj, sxproj)
    
    # convert to degrees
    if doDegrees:
        p0 = p0 * 180. / np.pi
    
    return p0

#|---------------------------------------|
#|--- Get decimal Carrington Rotation ---|
#|---------------------------------------|
def get_sunspyce_carr_rot(date, spacecraft=None):
    """
    Function to calculate the carrington rotation number corresponding
    to a date
    
    Input:
        date: a date string in a format suitable for spiceypy.str2et
              (this is fairly flexible in formatting)
    
    Optional Input:
        spacecraft: a string tag representing the spacecraft of interest
                    (current options are sta, stb, solo, psp, earth)
     
    Output:
        carr_rot: the decimal carrington rotation corresponding to date

    """
    twopi = 2 * np.pi
    
    # Convert time to utc
    time = parse_time(date).utc
    
    if type(spacecraft) != type(None): 
        if spacecraft.lower() in scDict:
            sc = scDict[spacecraft.lower()]
        else:
            sys.exit(spacecraft.lower() + ' not in scDict for sunspyce. Pick from sta, stb, solo, psp, earth.')
    
    # not dealing with anytim buried in tim2carr, this is a match within 0.0001
    carr_rot = sun.carrington_rotation_number(time) 
    earth_lon = get_sunspyce_lonlat(date, 'Earth', system='Carrington')[1]
    if earth_lon < 0: earth_lon += twopi
    frac = 1 - earth_lon / twopi
    
    # Subtract the fractional part and round off to get integer Carrington rot num
    n_carr = int(np.round(carr_rot - frac))
    diff = carr_rot - frac - n_carr
    if np.max(np.abs(diff)) > 0.1:
        print('Excessive residual in get_sunspyce_carr_rot')
    carr_rot = n_carr + frac
    
    if type(spacecraft) != type(None): 
        body_lon = get_sunspyce_lonlat(date, spacecraft, system='HEEQ')[1]
        carr_rot = carr_rot - body_lon / twopi
    
    return carr_rot
    
#|------------------------------------|
#|--- Load the basic spice kernels ---|
#|------------------------------------|
def load_common_kernels(pathIn):
    """
    Function to load all the basic kernels that are needed by spice 
    for most cases regardless of spacecraft choice. These kernels are:
        de421.bsp
        naif0012.tls
        heliospheric.tf
        pck00011_n0066.tpc    
    
    Input:
        pathIn: path to where these kernels live.

    """
    # Load the kernels that most satellites use
    kerns = ['de421.bsp', 'naif0012.tls', 'heliospheric.tf', 'pck00011_n0066.tpc']    
    
    # Get the loaded kernels to check against so
    # we can avoid reloading
    num_kernels = spice.ktotal('ALL')
    loadKerns = []
    for i in range(num_kernels):
        filename, kind, source, handle,  = spice.kdata(i, 'ALL')
        loadKerns.append(filename)
    loadKerns = np.array(loadKerns)
    
    # Load em up
    for aKern in kerns:
        fullName = pathIn + aKern
        if fullName not in loadKerns:
            spice.furnsh(fullName)
            
#|---------------------------------------|
#|--- Load PSP specific spice kernels ---|
#|---------------------------------------|
def load_psp_kernels(pathIn):
    """
    Function to load kernels related to Parker Solar Probe
    calculations via spice 
    
    Input:
        pathIn: path to where these kernels live.

    """
    # Get the loaded kernels to check against so
    # we can avoid reloading
    num_kernels = spice.ktotal('ALL')
    loadKerns = []
    for i in range(num_kernels):
        filename, kind, source, handle,  = spice.kdata(i, 'ALL')
        loadKerns.append(filename)
    loadKerns = np.array(loadKerns)
    
    # All the psp kernel folders
    pspFolds = ['attitude_long_term_predict', 'attitude_short_term_predict', 'gen', 'operations_sclk_kernel', 'orbit']
    
    # Search through and load anyone in these folders
    for aFold in pspFolds:
        files = os.listdir(pathIn+aFold)
        for aF in files:
            fullName = pathIn + aFold +'/' + aF
            if fullName not in loadKerns:
                spice.furnsh(fullName)
        
#|----------------------------------------|
#|--- Load SolO specific spice kernels ---|
#|----------------------------------------|
def load_solo_kernels(pathIn):
    """
    Function to load kernels related to Solar Orbiter
    calculations via spice 
    
    Input:
        pathIn: path to where these kernels live.

    """
    # Get the loaded kernels to check against so
    # we can avoid reloading
    num_kernels = spice.ktotal('ALL')
    loadKerns = []
    for i in range(num_kernels):
        filename, kind, source, handle,  = spice.kdata(i, 'ALL')
        loadKerns.append(filename)
    loadKerns = np.array(loadKerns)
    
    # All the solo kernel folders
    soloFolds = ['att', 'gen', 'operations_sclk_kernel', 'orbit', 'sclk']
    
    # Search through and load anyone in these folders
    for aFold in soloFolds:
        files = os.listdir(pathIn+aFold)
        for aF in files:
            fullName = pathIn + aFold +'/' + aF
            if fullName not in loadKerns:
                spice.furnsh(fullName)    
