"""
Module to process AIA observations

This script was designed to take in Level 1 AIA data pulled from
Sunpy Fido and process it for the WOMBAT GUI. It is just a wrapper
of existing funtions from aiapy. The basic steps were taken from
online documentation, but the respike and deconvolve steps do not
function as suggested so we have left them out. The current code gets
the pointing, registers it, and normalizes the image by the exposure
time. It returns a list of processed maps corresponding to the input
list of fits files

External Calls:
    aiapy


"""

import sunpy.map
import astropy.units as u
from aiapy.calibrate import register, update_pointing, respike, correct_degradation
from aiapy.calibrate.util import get_pointing_table
from aiapy.psf import deconvolve
                
#|-------------------------|
#|--- Main Prep Routine ---|
#|-------------------------|
def aia_prep(filesIn, downSize=1024):
    """
    Function that processes a list of level 1 AIA fits files into 
    astropy/sunpy maps. It was set up to use fits files pulled using
    Sunpy Fido. Most of the other prep functions are ports of IDL 
    routines to enable matching mass calculations but we do not support
    that for AIA so we can use the existing routines to prep the data

    Input:
        filesIn: a list of fits files path+names
    
    Optional Input:
        downSize: size of the output image (in pixels)
                  *** assuming a square output

    Output:
        maps_out: a list of maps corresponding to the input fits files


    """    
    # |------------------------------------------------------|
    # |------------- Loop to process each image -------------|
    # |------------------------------------------------------|
    num = len(filesIn)
    maps_out = []

    # Assume were working from files and not something loaded from read_sdo
    for i in range(num):
        #print ('Processing AIA image '+str(i+1) + ' out of '+str(num))
        # aia files are compressed/different from secchi so doensn't work with
        # straight up fits read, but using sunpy map equiv to read_sdo
        aia_map = sunpy.map.Map(filesIn[i])
                
        # Make range wide enough to get closest 3-hour pointing
        pointing_table = get_pointing_table("JSOC", time_range=(aia_map.date - 12 * u.h, aia_map.date + 12 * u.h))
        aia_map_updated_pointing = update_pointing(aia_map, pointing_table=pointing_table)
        
        # Respike
        # This fails bc of a JSOC TLS/SSL certificate error
        #aia_map_respike = respike(aia_map_updated_pointing)
        
        # Deconvolve
        # This either runs inifinitely slow or catches in loop so not running
        #aia_map_decon = deconvolve(aia_map_updated_pointing)
        
        # Registration
        aia_map_registered = register(aia_map_updated_pointing)
        
        if aia_map_registered.dimensions.x.to_value() > downSize:
            new_dimensions = [downSize, downSize] * u.pixel
            aia_map_registered = aia_map_registered.resample(new_dimensions)
        
        # Normalize by exposure time
        aia_map_normed = aia_map_registered / aia_map_registered.exposure_time
        
        # Add to the output list
        maps_out.append(aia_map_normed)
    return maps_out 