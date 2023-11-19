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
    Where mag=99., returns 0.
    """

    def uJy(mag):
        """Calculates a single flux"""
        return 10 ** (6 + (8.9 / 2.5)) * 10 ** (-mag / 2.5)

    if np.isscalar(mag_AB):
        if mag_AB == 99.0:
            return 0.0
        return uJy(mag_AB)

    fluxes = uJy(mag_AB)
    fluxes[mag_AB == 99.0] = 0.0  # Flooring mag=99
    return fluxes


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
    return fits.open(filename, cache=False)[0].data


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


## SED fitting utils


def find_best_model(coi, model_type):
    """
    Finds the model of type `model_type` with the highest logprob
    """
    assert model_type in ["G", "Q", "GQ"]
    mcmc_fl = np.load(f"./data/sed_fitting/mcmc_results/{model_type}/{coi}.npz")
    best_model_index = np.argmax(mcmc_fl["logprobs"])
    samples = mcmc_fl["samples"]
    flat_samples = samples.reshape((np.prod(samples.shape[:2]), samples.shape[-1]))
    return flat_samples[best_model_index]


# Handling LePHARE output
def unpack_lephare_spectra(coi):
    """
    Extracts the useful information from a LePHARE .spec output file
    This includes:
        Best-fitting galaxy spectrum
        Best-fitting quasar spectrum
        Best-fitting stellar spectrum
    """
    filename = f"./lephare/lephare_dev/output_spectra/Id{str(coi)[-9:]}.spec"
    spectra_array = np.loadtxt(filename, skiprows=193).T
    sep1, sep2 = (
        np.argwhere(np.diff(spectra_array[0, :]) < 0).reshape(2) + 1
    )  # Separating the different spectra

    galaxy_wavs = spectra_array[0, :sep1]  # wavs, in AA (?)
    galaxy_spectrum = AB_to_uJy(
        spectra_array[1, :sep1]
    )  # spectrum, from AB (? Not Vega?)
    quasar_wavs = spectra_array[0, sep1:sep2]
    quasar_spectrum = AB_to_uJy(spectra_array[1, sep1:sep2])
    stellar_wavs = spectra_array[0, sep2:]
    stellar_spectrum = AB_to_uJy(spectra_array[1, sep2:])

    return (
        galaxy_wavs,
        galaxy_spectrum,
        quasar_wavs,
        quasar_spectrum,
        stellar_wavs,
        stellar_spectrum,
    )
