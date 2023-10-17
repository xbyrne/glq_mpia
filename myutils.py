"""
Some utilities for the project
"""
import re
import numpy as np
from astropy import units as u, constants as const
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
    Where mag=99., returns 0.
    """

    def uJy(mag):
        """Calculates a single flux"""
        return 10 ** (6 + (8.9 / 2.5)) * 10 ** (-mag / 2.5)

    if np.isscalar(mag_AB):
        if mag_AB==99.:
            return 0.
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


## SED fitting utils


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


filters_list = np.loadtxt(
    "./data/sed_fitting/filters/filters_list_grizYJKW12.txt", dtype="str"
)


def galaxy_BAGPIPES_spectroscopy(t0, t1, mass, metallicity, dust_av, zgal):
    """
    Generates a galaxy spectrum, based on a range of parameters.
    This assumes a constant star formation rate. Require t1<t0
    Outputs are in AA, Jy
    """
    import bagpipes as pipes

    model_components = package_model_components(
        t0, t1, mass, metallicity, dust_av, zgal
    )
    model_components = package_model_components(1, 0.5, 10, 0.2, 0.2, 0.5)

    bagpipes_galaxy_model = pipes.model_galaxy(
        model_components, filt_list=filters_list, spec_wavs=np.arange(4e3, 6e4, 5.0)
    )

    bagpipes_galaxy_model.update(model_components)
    wavs = bagpipes_galaxy_model.wavelengths  # Rest frame
    flxs = bagpipes_galaxy_model.spectrum_full  # ergscma

    wavs = wavs * u.AA * (1 + zgal)  # Redshifting
    flxs = (
        flxs * (u.erg / u.s / (u.cm**2) / u.AA) * (wavs**2) / const.c
    )  # Converting F_lambda to F_nu
    flxs = flxs.to(u.Jy).value  # Jy
    wavs = wavs.value  # AA
    return wavs, flxs


model_qso = np.loadtxt(
    "./data/sed_fitting/vandenberk2001_z=0_fnu_noscale.txt", skiprows=1
)
filter_m_1450_file = np.loadtxt("data/sed_fitting/filters/filter_1450.txt")


def get_1450_filter(z_QSO):
    """
    Retrieves a pyphot filter which is a tophat function around 1450AA
    """
    wave = filter_m_1450_file[:, 0] * unit["AA"] * (1 + z_QSO)
    transmit = filter_m_1450_file[:, 1]
    filter_m_1450 = Filter(
        wave, transmit, name="1450_tophat", dtype="photon", unit="Angstrom"
    )
    return filter_m_1450


def quasar_spectroscopy(M_QSO, z_QSO):
    """
    Generates a quasar spectrum from a 1450A magnitude and a redshift.
    This will be based on the model chosen (default: vdb)
    Outputs are in AA, Jy
    """
    filter_m_1450 = get_1450_filter(z_QSO)
    # load quasar model, truncate Lyman-alpha forest
    spec_qso = model_qso[:, 1]
    spec_qso[model_qso[:, 0] * 1e4 < 1215.16] = 0.0
    flux_qso = spec_qso * 1e-3 * unit["Jy"]  # Models are apparently given in mJy...
    wavelength = (
        model_qso[:, 0] * 1e4 * (1 + z_QSO) * unit["AA"]
    )  # ...and wavelengths in microns

    # rescale to desired apparent magnitude 1450 AA
    mag_1450 = -2.5 * np.log10(filter_m_1450.get_flux(wavelength, flux_qso) / 3631)
    flux_qso *= 10 ** ((M_QSO - mag_1450) / -2.5)
    return wavelength.value, flux_qso.value  # in Jy


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


def spectrum_from_params(model):
    """
    Generates a spectrum from a parameter list
    """
    if len(model) == 2:
        M_QSO, z_QSO = model
        wavs, flxs = quasar_spectroscopy(M_QSO, z_QSO)
    elif len(model) == 6:
        t0, t1, mass, metallicity, dust_av, zgal = model
        wavs, flxs = galaxy_BAGPIPES_spectroscopy(
            t0, t1, mass, metallicity, dust_av, zgal
        )
    elif len(model) == 8:
        t0, t1, mass, metallicity, dust_av, zgal, M_QSO, z_QSO = model
        wavs, flxs = galaxy_BAGPIPES_spectroscopy(
            t0, t1, mass, metallicity, dust_av, zgal
        )
        quasar_wavs, quasar_flxs = quasar_spectroscopy(M_QSO, z_QSO)
        quasar_flxs = np.interp(wavs, quasar_wavs, quasar_flxs, left=0)
        flxs += quasar_flxs

    return wavs, flxs * 1e6  # to uJy


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
