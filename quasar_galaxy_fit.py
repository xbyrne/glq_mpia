"""
quasar_galaxy_fit.py
SED fitting to galaxy, quasar, and galaxy+quasar templates
"""

import sys
from multiprocessing import Pool, cpu_count
import os
import numpy as np
import pandas as pd
from astropy import units as u, constants as const
from pyphot import unit, Filter
import bagpipes as pipes
import emcee

#### your path for QSOGEN code, Temple+2021 DOI: 10.1093/mnras/stab2586
sys.path.append("./qsogen/")
from qsosed import Quasar_sed

os.environ["OMP_NUM_THREADS"] = "1"
print(cpu_count())

quasar_ids = np.load("./data/processed/quasar_ids.npz")["ids"].astype(int)
df = pd.read_csv("./data/processed/cut_crossmatched_objects.csv", index_col=0).loc[
    quasar_ids
]

band_names = ["g", "r", "i", "z", "Y", "J", "K", "W1", "W2"]
mag_df = df[[f"{band}_mag" for band in band_names]]
magerr_df = df[[f"{band}_magerr" for band in band_names]]
flux_df = df[[f"{band}_flux" for band in band_names]]
fluxerr_df = df[[f"{band}_fluxerr" for band in band_names]]


def load_grizYJKW12(ID):
    "Loads the photometry for a particular coadd id"
    ID = int(ID)
    photometry = np.vstack((flux_df.loc[ID], fluxerr_df.loc[ID])).T
    return photometry


filters_list = np.loadtxt(
    "./data/sed_fitting/filters/filters_list_grizYJKW12.txt", dtype="str"
)

filters_pyphot = {}
for band_name, file_name in zip(band_names, filters_list):
    file = np.loadtxt(f"./data/sed_fitting/{file_name}")

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


### BAGPIPES Galaxy Modelling ###


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


model_components = package_model_components(1, 0.5, 10, 0.2, 0.2, 0.5)  # Require t1<t0
bagpipes_galaxy_model = pipes.model_galaxy(
    model_components, filt_list=filters_list, spec_wavs=np.arange(4000.0, 60000.0, 5.0)
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
        flxs * (u.erg / u.s / (u.cm**2) / u.AA) * (wavs**2) / const.c
    )  # Converting F_lambda to F_nu
    flxs = flxs.to(u.Jy).value  # Jy
    wavs = wavs.value  # AA
    return wavs, flxs


### Quasar Modelling

model_qso = np.loadtxt(
    "./data/sed_fitting/vandenberk2001_z=0_fnu_noscale.txt", skiprows=1
)
filter_m_1450_file = np.loadtxt("./data/sed_fitting/filters/filter_1450.txt")


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


def tau_eff(z):
    """
    Computes the effective opacity to Lyman-alpha in the IGM
    following Bosman+2022 parametrization at 4.8<z<6.0.
    The effective opacity is clipped between (0,1).
    :param z:
    :return: tau_eff
    """

    tau_0 = 0.3
    beta = 13.7
    C = 1.35
    z0 = 4.8

    tau_eff = tau_0 * ((1 + z) / (1 + z0)) ** beta + C

    return tau_eff


def quasar_spectroscopy(M_QSO, z_QSO, ebv=0, vandenberk_template=False):
    """
    Generates a quasar spectrum from a 1450A magnitude and a redshift.
    This will be based on the model chosen (default: vdb)
    Outputs are in AA, Jy
    """
    filter_m_1450 = get_1450_filter(z_QSO)
    if vandenberk_template:
        # load quasar model
        spec_qso = np.copy(model_qso[:, 1])
        # apply IGM attenuation, Lya, LyB, LyG
        # NOTE: The attenuation of the forest at z<4.8 is overestimated
        ind_LyA = np.where((model_qso[:, 0] * 1e4 < 1215.16))[0]
        ind_LyB = np.where((model_qso[:, 0] * 1e4 < 1025.72))[0]
        ind_LyGplus = np.where((model_qso[:, 0] * 1e4 < 972.53))[0]
        ind_LymanLim = np.where((model_qso[:, 0] * 1e4 < 911.7))[0]
        spec_qso[ind_LyA] *= np.exp(
            -tau_eff(model_qso[ind_LyA, 0] * 1e4 * (1 + z_QSO) / 1215.67 - 1)
        )
        spec_qso[ind_LyB] *= np.exp(
            -0.16 * tau_eff(model_qso[ind_LyB, 0] * 1e4 * (1 + z_QSO) / 1025.72 - 1)
        )
        spec_qso[ind_LyGplus] *= np.exp(
            -0.056 * tau_eff(model_qso[ind_LyGplus, 0] * 1e4 * (1 + z_QSO) / 972.53 - 1)
        )
        spec_qso[ind_LymanLim] = 0
        flux_qso = spec_qso * 1e-3 * unit["Jy"]  # Template spectrum is in mJy..
        wavelength = (
            np.copy(model_qso[:, 0]) * 1e4 * (1 + z_QSO) * unit["AA"]
        )  # ...and wavelengths in microns
    else:
        sed = Quasar_sed(
            z=z_QSO, ebv=ebv, wavlen=np.logspace(2.5, 4, num=20001, endpoint=True)
        )
        wavelength = sed.wavred * unit["AA"]
        # doing the conversion with astropy units and then
        flux_qso_flambda = sed.flux * (u.erg / u.s / u.cm**2 / u.AA) * (1 + z_QSO)
        flux_qso = (flux_qso_flambda * (sed.wavred * u.AA) ** 2 / const.c).to(
            u.Jy
        ).value * unit["Jy"]

    # rescale to desired apparent magnitude 1450 AA
    # flux_qso expected in Jy
    mag_1450 = -2.5 * np.log10(filter_m_1450.get_flux(wavelength, flux_qso) / 3631)
    flux_qso *= 10 ** ((M_QSO - mag_1450) / -2.5)
    return wavelength.value, flux_qso.value  # in AA, Jy


### Generating spectra from model parameters


def spectrum_from_params(theta, z_QSO=6):
    """
    Generates a spectrum from a parameter list
    """
    if len(theta) == 2:
        M_QSO, ebv = theta
        wavs, flxs = quasar_spectroscopy(
            M_QSO=M_QSO, z_QSO=z_QSO, ebv=ebv, vandenberk_template=False
        )
    elif len(theta) == 6:
        t0, t1, mass, metallicity, dust_av, zgal = theta
        wavs, flxs = galaxy_BAGPIPES_spectroscopy(
            t0, t1, mass, metallicity, dust_av, zgal
        )
    elif len(theta) == 8:
        t0, t1, mass, metallicity, dust_av, zgal, M_QSO, ebv = theta
        wavs, flxs = galaxy_BAGPIPES_spectroscopy(
            t0, t1, mass, metallicity, dust_av, zgal
        )
        quasar_wavs, quasar_flxs = quasar_spectroscopy(
            M_QSO=M_QSO, z_QSO=z_QSO, ebv=ebv, vandenberk_template=False
        )
        quasar_flxs = np.interp(wavs, quasar_wavs, quasar_flxs, left=0)
        flxs += quasar_flxs

    return wavs, flxs * 1e6  # to uJy


### MCMC fitting


def log_prior(theta, obj_type):
    """
    A uniform log prior with boundaries in a realistic range.
    Returns 0 if inside the good range
    Returns -np.inf if outside
    """
    if obj_type == "GQ":
        t0, t1, mass, metallicity, dust_av, zgal, M_QSO, ebv = theta
        if (
            (0 < t0 < 13)
            & (0 < t1 < t0)
            & (8 < mass < 15)
            & (0 < metallicity < 2)
            & (0 < dust_av < 2)
            & (0 < zgal < 4)
            & (18 < M_QSO < 24)
            & (0.0 <= ebv < 2)
        ):
            return 0.0

    elif obj_type == "Q":
        M_QSO, ebv = theta
        if (18 < M_QSO < 24) & (0.0 <= ebv < 2):
            return 0.0

    elif obj_type == "G":
        t0, t1, mass, metallicity, dust_av, zgal = theta
        if (
            (0 < t0 < 13)
            & (0 < t1 < t0)
            & (8 < mass < 15)
            & (0 < metallicity < 2)
            & (0 < dust_av < 2)
            & (0 < zgal < 10)
        ):
            return 0.0
    else:
        raise ValueError(
            f"Invalid obj_type: {obj_type}. Must be either `G`, `Q`, or `GQ`."
        )

    return -np.inf


def log_likelihood(theta, y, yerr, z_QSO):
    "Log likelihood for a given model"
    fluxes_model = np.zeros(y.shape)  # 9d vector

    wavs, flxs = spectrum_from_params(theta, z_QSO=z_QSO)
    fluxes_model = spectrum_to_photometry(wavs, flxs)

    if np.sum(fluxes_model) == 0.0:
        # Invalid galaxy params => BAGPIPES gives a blank spectrum
        # Could this happen with GQ?
        return -np.inf

    sigma2 = yerr**2
    return -0.5 * np.sum((y - fluxes_model) ** 2 / sigma2 + np.log(sigma2))


def log_probability(theta, y, yerr, obj_type, z_QSO):
    "Bayesian probability update"
    lp = log_prior(theta, obj_type)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, yerr, z_QSO)


def suggest_init(obj_type):
    """
    Generates an initial set of parameters, randomly distributed around mean_init_row
    """
    mean_init_row = np.array([1, 0.5, 10, 0.2, 0.2, 0.5, 20, 0.2])
    if obj_type == "G":
        mean_init_row = mean_init_row[:6]
    elif obj_type == "Q":
        mean_init_row = mean_init_row[6:]
    elif obj_type != "GQ":
        raise ValueError(
            f"Invalid obj_type: {obj_type}. Must be either `G`, `Q`, or `GQ`."
        )
    ndim = len(mean_init_row)
    return mean_init_row + np.random.uniform(low=-0.1, high=0.1, size=ndim)


def initialise_chains(nwalkers, flux, err_flux, obj_type, z_QSO):
    """
    Returns a set of initial vectors in parameter space, all of which within the prior.
    This is a np array of shape (nwalkers, nparams)
    """
    base_init = suggest_init(obj_type)
    pos = np.tile(base_init, (nwalkers, 1))
    i = 0
    while np.all(pos[-1, :] == base_init) and i < nwalkers:
        new_row = suggest_init(obj_type)

        # Ensure that suggested initial point is actually in the prior
        if (log_prior(new_row, obj_type) != -np.inf) & (
            log_likelihood(new_row, flux, err_flux, z_QSO) != -np.inf
        ):
            pos[i, :] = new_row
            i += 1
    return pos


def fit_sed(
    flux,
    err_flux,
    obj_type,
    nsteps=5000,
    discard=1000,
    thin=10,
    nwalkers=32,
    z_QSO=6,
    **kwargs,
):  # kwargs to be passed to EnsembleSampler
    """
    Do the MCMC sampling process to SED-fit the fluxes (and their errors) to a G, Q, or GQ model
    Returns each of the samples, and their log_probs, in a big array.
    """

    init = initialise_chains(nwalkers, flux, err_flux, obj_type, z_QSO=z_QSO)
    nwalkers, ndim = init.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(flux, err_flux, obj_type, z_QSO),
            pool=pool,
            **kwargs,
        )
        sampler.run_mcmc(init, nsteps, progress=True)

    samples = sampler.get_chain(flat=False, thin=thin, discard=discard)
    log_prob = sampler.get_log_prob(flat=False, thin=thin, discard=discard)

    return samples, log_prob


def fit_sed_id(quasar_id, z_QSO):
    photometry = load_grizYJKW12(quasar_id)
    flxs = photometry[:, 0]
    flxerrs = photometry[:, 1]
    for ot in ["GQ", "Q", "G"]:
        samples, log_prob = fit_sed(
            flux=flxs,
            err_flux=flxerrs,
            obj_type=ot,
            nsteps=n,
            discard=d,
            thin=t,
            z_QSO=z_QSO,
            nwalkers=nwalkers,
            a=2.0,
        )  # samples.shape=(3000,nwalkers,{2/8/6}); log_prob.shape=(3000,nwalkers)
        np.savez_compressed(
            f"./data/sed_fitting/mcmc_results/{ot}/{quasar_id}.npz",
            samples=samples,
            logprobs=log_prob,
        )


if __name__ == "__main__":
    ### Running SED fitting ###

    nwalkers = 32
    n = 10000  # nsteps
    d = 4000  # discard
    t = 2  # thin
    num_samples = int((n - d) / t)

    all_samples = [
        np.zeros((len(quasar_ids), num_samples, nwalkers, nparams))
        for nparams in [8, 2, 6]  # GQ, Q, G respectively
    ]
    all_log_probs = [
        np.zeros((len(quasar_ids), num_samples, nwalkers)) for i in range(3)
    ]

    fit_sed_id(quasar_id=1599741416, z_QSO=5.941)  # J0109
    fit_sed_id(quasar_id=1143273115, z_QSO=6.074)  # J0603
    fit_sed_id(quasar_id=1695974542, z_QSO=5.986)  # J0122
