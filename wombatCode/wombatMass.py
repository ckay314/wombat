"""
Module with all the mass calculations for wombat

More words, eventually

External calls:
    wcs_funs
"""

import numpy as np
import sys
from scipy.spatial import ConvexHull
from skimage.draw import polygon

sys.path.append('prepCode/') 
from wcs_funs import fitshead2wcs, wcs_get_coord


#|------------------------------|
#|---- Electron Theory Calc ----|
#|------------------------------|
def elTheory(Rin,theta, center=False, limb=0.63, returnAll=False):
    """
    Function that will calculate the expected brightness of a single electron
    based on the distance and plane of sky separation. 
    
    Inputs:
        Rin: the projected distance from electron to the sun center. this is typically 
             found using wcs_get_coord then converting from arcsec to rsun and using 
             that as the input (in Rsun)
    
        theta: the separation of the electron from the plane of sky (in deg)

    Optional Inputs:
        center: flag to return central disk brightness instead of mean solar 
                brightness (defaults to False)
        
        limb:  limb darkening coefficient (defaults to 0.63 as in IDL)
    
        returnAll: return Bt, Br, and pol in addition to R, B (see below)
                    
    
    Outputs: 
        R: unprojected radial distance from the sun center
        B: total brightness for one electron at Rin, theta
    
    Optional Outputs:
        Bt: tangential brightness for one electron at Rin, theta
        Br: radial brightness for one electron at Rin, theta
        pol: polarization for one electron at Rin, theta
    
    Notes:
        This is a minimal port of the IDL code of the same name with some 
        clean up and pythonification
 
    """
    radeg = 180. / np.pi

    # 0.63 is default for limb darkening
    u = limb
    
    const = 1.24878e-25
    if not center:
        const = const/(1-u/3) # convert to mean solar brightness
    
    if theta >= 90:
        theta = 180 - theta
    if theta <= -90:
        theta = 180 + theta
    if np.abs(theta) > 90:
        sys.exit('Theta greater than 90 in elTheory, should check why')
    
    # Deproject distance    
    R = Rin/np.cos(theta/radeg)
    # set min value of 1 so it doesn't complain, will be occulted anywat
    R[np.where(R<1)] = 1.0001
    sinchi2 = (Rin/R)**2	# angle between sun center, electron, observer
    s = 1./R
    s2 = s**2
    c2 = (1.-s2)
    c = np.sqrt(c2)			# cos(omega)
    g = c2*(np.log((1.+s)/c))/s
    
    #  Compute Van de Hulst Coefficients
    #  Expressions are given in Billings (1968) after Minnaert (1930)
    ael = c*s2
    cel = (4.-c*(3.+c2))/3.
    bel = -(1.-3.*s2-g*(1.+3.*s2))/8.
    del0 = (5.+s2-g*(5.-s2))/8.
    
    #  Compute electron brightness
    #  pB is polarized brightness (Bt-Br)
    Bt = const*( cel + u*(del0-cel) )
    pB = const* sinchi2 *( (ael + u*(bel-ael) ) )
    Br = Bt-pB
    B = Bt+Br
    Pol = pB/B
    
    if returnAll:
        return R,B,Bt,Br,Pol
    else:
        return R,B
        
#|----------------------------------|
#|---- Total Brightness to Mass ----|
#|----------------------------------|
def TB2mass(img, hdr, onlyNe=False, doPB=False):
    """
    Function that will convert a total brightness image into a mass (density)
    image using elTheory and header information.
    
    Inputs:
        img: a total brightness image or an excess brightness image
    
        hdr: the corresponding header

    Optional Inputs:
        onlyNe: a flag to return the number of electrons instead of 
                the mass per pixel (defaults to false)
                
        doPB: a flag to account for the polarization brightness and 
              use (Bt-Br) from elTheory instead of B (defaults to False) 
    
    Outputs: 
        mass: the calculated image with mass per pixel instead of brightness
        hdr: the updated header
    
   Notes:
        This is a minimal port of the IDL code of the same name with some 
        clean up and pythonification
 
    """
    # |---------------------------------------|
    # |------ Set up the distance array ------|
    # |---------------------------------------|
    # First part a little different order than IDL but we have code to 
    # get sunc from wcs already
    wcs = fitshead2wcs(hdr) # not system = A here it seems
    
    # Get the distance factor over the full grid so we can use that in eltheory
    dist = wcs_get_coord(wcs) #[2,naxis,naxis] with axis having usual swap from idl
    if hdr['cunit1'] == 'deg':
        dist = dist * 3600.
    # SECCHI Version
    if 'rsun' in hdr:
        dist = np.sqrt(dist[0,:,:]**2 + dist[1,:,:]**2) / hdr['rsun']
    # PSP version
    elif 'RSUN_ARC' in hdr:
        hdr['rsun'] = hdr['RSUN_ARC']
        dist = (np.sqrt(dist[0,:,:]**2 + dist[1,:,:]**2) / hdr['RSUN_ARC'])
        
    # |---------------------------------------|
    # |----------- Apply el Theory -----------|
    # |---------------------------------------|
    # Assume no pos keyword or cmelonlat for now (1-6-126)
    pos_angle = 0.
    if not doPB:
        R,B = elTheory(dist,0)
    else:
        R,B,Bt,Br,Pol = elTheory(dist,0, returnAll=True)
    B[np.where(B == 0)] = 1   
    
    
    # |---------------------------------------|
    # |--------- Various Conversions ---------|
    # |---------------------------------------|
    # A good portion of IDL seems to be commented out so ignoring that
    solar_radius = hdr['rsun']
    cm_per_arcsec = 6.96e10 / solar_radius
    if hdr['cunit1'] == 'deg':
        cm2_per_pixel= (cm_per_arcsec * hdr['cdelt1']*3600.)**2
    else:
        cm2_per_pixel = (cm_per_arcsec * hdr['cdelt1'])**2
    
    # Electron density or mass? 
    if onlyNe:
        conv = cm2_per_pixel
    else:
        conv = cm2_per_pixel * 1.974e-24 # why is this ne2mass separate function in IDL...
    
    if doPB:    
        mass = img / (Bt - Br)
    else:
        mass = img / B
    
    mass = conv * mass     
    # Not doing ROI here
    
    # Clean out NaNs
    mass[~np.isfinite(mass)] = 0
  
    # add a tag into header
    hdr['history'] = 'Converted to mass units using calcCMEmass.py'
    
    return mass, hdr

#|------------------------|
#|---- Points to Mask ----|
#|------------------------|
def pts2mask(imShape, scats):
    """
    Function that will take a list of the projected wireframe points and
    convert it to a mask to be used in the mass summing
    
    Inputs: 
        imShape: the shape of the image on which the pts are projected
    
        scats: the projected scatter points as [x_positions, y_positions]
    
    Outputs: 
        mask: a mask of the wireframe (same shape as im)

    """
    x_positions, y_positions = scats[0], scats[1]
    # Make sure we have a wf, and also enough of a wf
    if len(x_positions) > 50: 
        points = np.transpose(np.array([x_positions, y_positions]))
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
    
        mask = np.zeros(imShape, dtype=int)
        rr, cc = polygon(vertices[:, 0], vertices[:, 1], shape=(imShape))   
        mask[rr,cc] = 1
        return mask
    else:
        return None