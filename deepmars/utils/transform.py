"""Coordinate Transform Functions

Functions for coordinate transforms, used by input_data_gen.py functions.
"""
import numpy as np


def estimate_longlatdiamkm(dim, llbd, distcoeff, coords):
    """First-order estimation of long/lat, and radius (km) from
    (Orthographic) x/y position and radius (pix).

    For images transformed from ~6000 pixel crops of the 30,000 pixel
    LROC-Kaguya DEM, this results in < ~0.4 degree latitude, <~0.2
    longitude offsets (~2% and ~1% of the image, respectively) and ~2% error in
    radius. Larger images thus may require an exact inverse transform,
    depending on the accuracy demanded by the user.

    Parameters
    ----------
    dim : tuple or list
        (width, height) of input images.
    llbd : tuple or list
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    distcoeff : float
        Ratio between the central heights of the transformed image and original
        image.
    coords : numpy.ndarray
        Array of crater x coordinates, y coordinates, and pixel radii.

    Returns
    -------
    craters_longlatdiamkm : numpy.ndarray
        Array of crater longitude, latitude and radii in km.
    """
    # Expand coords.
    long_pix, lat_pix, radii_pix = coords.T

    # Determine radius (km).
    km_per_pix = 1. / km2pix(dim[1], llbd[3] - llbd[2], dc=distcoeff)
    radii_km = radii_pix * km_per_pix

    # Determine long/lat.
    deg_per_pix = km_per_pix * 180. / (np.pi * 3389.0)
    long_central = 0.5 * (llbd[0] + llbd[1])
    lat_central = 0.5 * (llbd[2] + llbd[3])

    # Iterative method for determining latitude.
    lat_deg_firstest = lat_central - deg_per_pix * (lat_pix - dim[1] / 2.)
    latdiff = abs(lat_central - lat_deg_firstest)
    # Protect against latdiff = 0 situation.
    latdiff[latdiff < 1e-7] = 1e-7
    lat_deg = lat_central - (deg_per_pix * (lat_pix - dim[1] / 2.) *
                             (np.pi * latdiff / 180.) /
                             np.sin(np.pi * latdiff / 180.))
    # Determine longitude using determined latitude.
    long_deg = long_central + (deg_per_pix * (long_pix - dim[0] / 2.) /
                               np.cos(np.pi * lat_deg / 180.))

    # Return combined long/lat/radius array.
    return np.column_stack((long_deg, lat_deg, radii_km))


def coord2pix(cx, cy, cdim, imgdim, origin="upper"):
    """Converts coordinate x/y to image pixel locations.

    Parameters
    ----------
    cx : float or ndarray
        Coordinate x.
    cy : float or ndarray
        Coordinate y.
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image.
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels.
    origin : 'upper' or 'lower', optional
        Based on imshow convention for displaying image y-axis. 'upper' means
        that [0, 0] is upper-left corner of image; 'lower' means it is
        bottom-left.

    Returns
    -------
    x : float or ndarray
        Pixel x positions.
    y : float or ndarray
        Pixel y positions.
    """

    x = imgdim[0] * (cx - cdim[0]) / (cdim[1] - cdim[0])

    if origin == "lower":
        y = imgdim[1] * (cy - cdim[2]) / (cdim[3] - cdim[2])
    else:
        y = imgdim[1] * (cdim[3] - cy) / (cdim[3] - cdim[2])

    return x, y


def pix2coord(x, y, cdim, imgdim, origin="upper"):
    """Converts image pixel locations to Plate Carree lat/long.  Assumes
    central meridian is at 0 (so long in [-180, 180)).

    Parameters
    ----------
    x : float or ndarray
        Pixel x positions.
    y : float or ndarray
        Pixel y positions.
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image.
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels.
    origin : 'upper' or 'lower', optional
        Based on imshow convention for displaying image y-axis. 'upper' means
        that [0, 0] is upper-left corner of image; 'lower' means it is
        bottom-left.

    Returns
    -------
    cx : float or ndarray
        Coordinate x.
    cy : float or ndarray
        Coordinate y.
    """

    cx = (x / imgdim[0]) * (cdim[1] - cdim[0]) + cdim[0]

    if origin == "lower":
        cy = (y / imgdim[1]) * (cdim[3] - cdim[2]) + cdim[2]
    else:
        cy = cdim[3] - (y / imgdim[1]) * (cdim[3] - cdim[2])

    return cx, cy


def km2pix(imgheight, latextent, dc=1., a=3389.0):
    """Returns conversion from km to pixels (i.e. pix / km).

    Parameters
    ----------
    imgheight : float
        Height of image in pixels.
    latextent : float
        Latitude extent of image in degrees.
    dc : float from 0 to 1, optional
        Scaling factor for distortions.
    a : float, optional
        World radius in km.  Default is Moon (1737.4 km).

    Returns
    -------
    km2pix : float
        Conversion factor pix/km
    """
    return (180. / np.pi) * imgheight * dc / latextent / a
