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
    # returns yaw (arcsec), pitch (arcsec), and roll angle (deg)
    # If doDeg then all three params returned in deg
    
    # Define the unilts
    roll_units = 180 / np.pi
    xy_units   = roll_units * 3600
    if doDeg: xy_units = roll_units
    if doRad:
        roll_units = 1
        xy_units   = 1
        
    # Determine which spacecraft was requested and make it spicy    
    if spacecraft.lower() in scDict:
        sc = scDict[spacecraft.lower()]
    else:
        sys.exit(spacecraft.lower() + ' not in scDict for sunspyce. Pick from sta, stb, solo, psp, earth.')
    
    sc_stereo = False
    if sc in ['-234', '-235']: sc_stereo = True
    
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
    # Determine which spacecraft was requested and make it spicy    
    if spacecraft.lower() in scDict:
        sc = scDict[spacecraft.lower()]
    else:
        sys.exit(spacecraft.lower() + ' not in scDict for sunspyce. Pick from sta, stb, solo, psp.')
    
    sc_base = ''
    if sc == '-96':
        sc_base = 'SPP_SPACECRAFT'
    elif sc == '-144':
        sc_base = 'SOLO_SRF'
        
    time = parse_time(date).utc
    
    # Determine which coord system is specified
    # assuming single value for now 
    if system:
        system = system.upper()
        if system == 'HEQ': system = 'HEEQ'
        elif system in 'CARRINGTON': system = 'CARRINGTON'
    else:
        system == 'RTN'
        
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

    cmat, clkout = spice.ckgp(int(sc)*1000, sclkdp, tol, frame)
    
    # Modify the c-matrix based on the instrument keyword
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
    
    # Apply any additional processing
    if system == 'HPC':
        ccmat = np.matmul(ccmat, [[0, 0, 1.], [1., 0, 0], [0, 1., 0]])

    # ignoring weird storing stuff
    return ccmat    
        
#|---------------------------|
#|--- Get spacecraft roll ---|
#|---------------------------|
def get_sunspyce_roll(date, spacecraft, system=None, instrument=None, doRad=False, tolerance=None):
    # Assuming passed correct things
    units = 180. / np.pi
    
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

#|-----------------------------------------|
#|--- Get Cartesian (?) location of s/c ---|
#|-----------------------------------------|
def get_sunspyce_coord(date, spacecraft, system=None, instrument=None, target=None, doMeters=False, doAU=False, doVelocity=True):
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
        
    # Call get_sunspice_coord
    state = get_sunspyce_coord(date, spacecraft, system=system, instrument=instrument, doMeters=doMeters, doAU=doAU, doVelocity=False)
   
    # Ignoring planetographic
    
    # Use reclat to convert rect coords into rad, lon, lat
    rad, lon, lat = spice.reclat(state)
    
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
    
    # Conversion to degrees
    if doDegrees:
        lon = lon * 180. / np. pi
        lat = lat * 180. / np. pi
    return [rad, lon, lat]

#|-------------------------------|
#|--- Get the p0 angle of s/c ---|
#|-------------------------------|
def get_sunspyce_p0_angle(date, spacecraft, doDegrees=False):
    # Determine which spacecraft was requested and make it spicy    
    if spacecraft.lower() in scDict:
        sc = scDict[spacecraft.lower()]
    else:
        sys.exit(spacecraft.lower() + ' not in scDict for sunspyce. Pick from sta, stb, solo, psp, earth.')
    
    
    # get the orientation of the sun axis in J2000
    et = spice.str2et(date)
    rota = spice.pxform('IAU_SUN', 'J2000', et)
    sun_north = rota[:,2]
    j2000_north = [0., 0., 1.]

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
