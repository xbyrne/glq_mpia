"""
spectra_utils.py
Some utils for making spectra from models
"""

import numpy as np
import bagpipes as pipes
from astropy import units as u, constants as const
from pyphot import unit, Filter

band_names = ["g", "r", "i", "z", "Y", "J", "K", "W1", "W2"]
eff_wavs = [
    4862.24,
    6460.63,
    7850.77,
    9199.28,
    9906.83,
    12681.01,
    21588.53,
    33526,
    46028,
]


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


filters_list = np.loadtxt(
    "../data/sed_fitting/filters/filters_list_grizYJKW12.txt", dtype="str"
)

model_components = package_model_components(1, 0.5, 10, 0.2, 0.2, 0.5)
bagpipes_galaxy_model = pipes.model_galaxy(
    model_components, filt_list=filters_list, spec_wavs=np.arange(4e3, 6e4, 5.0)
)


def galaxy_BAGPIPES_spectroscopy(t0, t1, mass, metallicity, dust_av, zgal):
    """
    Generates a galaxy spectrum, based on a range of parameters.
    This assumes a constant star formation rate. Require t1<t0
    Outputs are in AA, Jy
    """
    model_components = package_model_components(
        t0, t1, mass, metallicity, dust_av, zgal
    )
    bagpipes_galaxy_model.update(model_components)
    wavs = bagpipes_galaxy_model.wavelengths  # Rest frame
    flxs = bagpipes_galaxy_model.spectrum_full  # ergscma

    wavs = wavs * u.AA * (1 + zgal)  # Redshifting
    flxs = (
        flxs * (u.erg / u.s / (u.cm ** 2) / u.AA) * (wavs ** 2) / const.c
    )  # Converting F_lambda to F_nu
    flxs = flxs.to(u.Jy).value  # Jy
    wavs = wavs.value  # AA
    return wavs, flxs


model_qso = np.loadtxt(
    "../data/sed_fitting/vandenberk2001_z=0_fnu_noscale.txt", skiprows=1
)
filter_m_1450_file = np.loadtxt("../data/sed_fitting/filters/filter_1450.txt")


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


def extract_lephare_spectra(coi):
    """
    Extracts the LePHARE spectra from the .spec files
    The spectra are returned in the order Galaxy, Quasar, Star
    """
    coi = str(coi)[-9:]  # LePHARE not liking 10-digit IDs
    all_spectra = np.loadtxt(
        f"../lephare/lephare_dev/output_spectra/Id{coi}.spec", skiprows=193
    ) * [1, u.ABmag]
    all_spectra[:, 1] = [mag.to(u.Jy).value * 1e6 for mag in all_spectra[:, 1]]
    sb1, sb2 = np.where(np.diff(all_spectra[:, 0]) < 0)[0] + 1
    wavs = [
        all_spectra[:sb1, 0],  # Galaxy
        all_spectra[sb1:sb2, 0],  # Quasar
        all_spectra[sb2:, 0],  # Star
    ]
    specs = [
        all_spectra[:sb1, 1],  # Galaxy
        all_spectra[sb1:sb2, 0],  # Quasar
        all_spectra[sb2:, 0],  # Star
    ]
    return wavs, specs
