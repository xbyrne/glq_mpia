"""
Some utilities for the project
"""
import re
import numpy as np
from astropy.io import fits
from pyvo.dal import sia

## Converting Magnitudes and Fluxes

# See https://www.eso.org/rm/api/v1/public/releaseDescriptions/144
#  and https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
delta_m = {"J": 0.916, "K": 1.827, "W1": 2.699, "W2": 3.339}


def vega_to_AB(vega_mag, band):
    """
    Converts Vega magnitudes to AB system
    """
    return vega_mag + delta_m[band]


def AB_to_uJy(mag_AB):
    """
    Converts an AB magnitude to a flux in uJy.
    """
    return 10 ** (29 - (48.60 / 2.5)) * 10 ** (-mag_AB / 2.5)


## Downloading and Processing Images


def fov2d(dec, fov=7.5 / 3600):
    """
    Returns a tuple with a field of view centred on a direction with
    declination `dec` and of radius `fov`.
    `fov` and `dec` given in degrees.
    Default `fov` is 7.5 arcsec.
    """
    return (fov / np.cos(np.deg2rad(dec)), fov)


SIA_SERVICE_URL = "https://datalab.noirlab.edu/sia/des_dr2"  # DES DR2 SIA service URL
SIA_SERVICE = sia.SIAService(SIA_SERVICE_URL)


def fetch_object_urls(ra, dec, sia_service=SIA_SERVICE):
    """
    Fetches the SIA URLs for a given RA and Dec, from a given sia_service
    """
    url_long_list = sia_service.search((ra, dec), fov2d(dec)).to_table()["access_url"]
    url_list = [
        url for url in url_long_list if re.match(".+_[grizY].fits.fz&extn=1.+", url)
    ]
    return url_list


def fetch_image(filename):
    """Fetches image from URL without printing to stdout"""
    # TODO: What to do if image opening fails? In what ways does it fail?
    img = fits.open(filename, cache=False)[0].data
    return img


def crop_image(raw_img, px=28):
    """
    Crops an input numpy array to a `px`x`px` square
    """
    if any(
        dim < px for dim in raw_img.shape[:2]
    ):  # Ensures array bigger than target size (shouldn't be problems here)
        return None

    centre_x, centre_y = (dim // 2 for dim in raw_img.shape[:2])
    cropped_img = raw_img[
        centre_x - px // 2 : centre_x + px // 2, centre_y - px // 2 : centre_y + px // 2
    ]
    return cropped_img


def band_from_url(url):
    """
    Hacks the band (g/r/i/z/y) from a SIA URL name
    """
    return url.split(".fits.fz")[0][-1]
