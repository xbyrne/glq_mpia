"""
Some utilities for the project
"""
import re
import numpy as np
from astropy.io import fits
from pyvo.dal import sia
from pyphot import unit, Filter

## Converting Magnitudes and Fluxes

# See https://www.eso.org/rm/api/v1/public/releaseDescriptions/144
#  and https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
band_names = ["g", "r", "i", "z", "Y", "J", "K", "W1", "W2"]
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
    return 10 ** (6 + (8.9 / 2.5)) * 10 ** (-mag_AB / 2.5)


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


## BAGPIPES utils


def package_model_components(t0, t1, mass, metallicity, dust_av, zgal):
    """
    Converts a series of galactic parameters into a model_components dictionary
    with a constant star formation rate
    """
    constant = {}  # Star formation - tophat function
    constant["age_max"] = t0  # Time since SF switched on: Gyr
    constant["age_min"] = t1  # Time since SF switched off: Gyr; t1<t0
    constant["massformed"] = mass  # vary log_10(M*/M_solar) between 1 and 15
    constant["metallicity"] = metallicity  # vary Z between 0 and 2.5 Z_oldsolar

    dust = {}  # Dust component
    dust["type"] = "Calzetti"  # Define the shape of the attenuation curve
    dust["Av"] = dust_av  # magnitudes

    nebular = {}  # Nebular emission component
    nebular["logU"] = -3

    model_components = {}  # The model components dictionary
    model_components["redshift"] = zgal  # Observed redshift
    model_components["constant"] = constant
    model_components["dust"] = dust
    model_components["nebular"] = nebular

    return model_components


filters_pyphot = {}  # For deducing photometry from spectra
for band_name, file_name in zip(
    band_names,
    [
        "CTIO_DECam.g",
        "CTIO_DECam.r",
        "CTIO_DECam.i",
        "CTIO_DECam.z",
        "CTIO_DECam.Y",
        "Paranal_VISTA.J",
        "Paranal_VISTA.Ks",
        "WISE_WISE.W1",
        "WISE_WISE.W2",
    ],
):
    file = np.loadtxt(f"./data/sed_fitting/filters/{file_name}.dat")

    wave = file[:, 0] * unit["AA"]
    transmit = file[:, 1]
    filters_pyphot[band_name] = Filter(
        wave, transmit, name=band_name, dtype="photon", unit="Angstrom"
    )


def spectrum_to_photometry(wavelengths, fluxes):
    """
    Converts a spectrum to a photometry in grizYJKW12, using pyphot.
    Requires wavelengths to be in angstrom and fluxes to be in jansky
    Returns a 9d vector of the magnitudes in each band, in muJy
    """
    wavelengths *= unit["AA"]
    fluxes *= unit["Jy"]

    band_flxs = np.zeros(len(filters_pyphot))
    for i, band_name in enumerate(band_names):
        band_flxs[i] = filters_pyphot[band_name].get_flux(wavelengths, fluxes)  # Jy

    return band_flxs * 1e6  # muJy
