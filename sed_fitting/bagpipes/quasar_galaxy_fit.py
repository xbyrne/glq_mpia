from multiprocessing import Pool
import os
import numpy as np
import pandas as pd
from astropy import units as u, constants as const
from pyphot import unit, Filter
import bagpipes as pipes
import emcee

os.environ["OMP_NUM_THREADS"] = "1"

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

sylt_ids = np.load("../../contrastive_learning/sylt_ids.npz")["ids"]
df = pd.read_csv("../../selecting_data/objs_7102.csv", index_col=0).loc[sylt_ids]
mag_df = df[[f"{band}_mag" for band in band_names]]
magerr_df = df[[f"{band}_magerr" for band in band_names]]
flux_df = df[[f"{band}_flux" for band in band_names]]
fluxerr_df = df[[f"{band}_fluxerr" for band in band_names]]

### BAGPIPES Galaxy Modelling ###


def load_grizYJKW12(ID):
    ID = int(ID)
    photometry = np.vstack(
        (
            flux_df.loc[ID],
            fluxerr_df.loc[ID]
            / 3,  # Table has the 3sigma error; bagpipes probably wants the 1sigma
        )
    ).T
    return photometry


filters_list = np.loadtxt("./filters_list_grizYJKW12.txt", dtype="str")  # For bagpipes


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

modeldir_dict = {
    "vdb": "./quasar_templates/vandenberk2001_z=0_fnu_noscale.txt",
    "sls": "./quasar_templates/selsing2016_z=0_fnu_noscale.txt",
    "cln": "./quasar_templates/Colina_Quasar_z=0_ETC-Glikman+Hernan-Caballero.txt",
    "mcg": "./quasar_templates/mcgreer_z75qso.dat",
}
model_qso = np.loadtxt(modeldir_dict["vdb"], skiprows=1)

filter_m_1450_file = np.loadtxt("../filters/filter_1450.txt")


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


### Deducing photometry from a spectrum ###

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
    file = np.loadtxt(f"../filters/{file_name}.dat")

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


### MCMC fitting


def log_prior(theta, obj_type):
    """
    A uniform log prior with boundaries in a realistic range.
    Returns 0 if inside the good range
    Returns -np.inf if outside
    """
    if obj_type == "GQ":
        t0, t1, mass, metallicity, dust_av, zgal, M_QSO, z_QSO = theta
        if (
            (0 < t0 < 13)
            & (0 < t1 < t0)
            & (8 < mass < 15)
            & (0 < metallicity < 2)
            & (0 < dust_av < 2)
            & (0 < zgal < 4)
            & (18 < M_QSO < 24)
            & (5.5 < z_QSO < 7)
        ):
            return 0.0

    elif obj_type == "Q":
        M_QSO, z_QSO = theta
        if (18 < M_QSO < 24) & (5.5 < z_QSO < 7):
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


def log_likelihood(theta, y, yerr, obj_type):
    fluxes_model = np.zeros(y.shape)  # 9d vector

    if obj_type == "GQ":
        t0, t1, mass, metallicity, dust_av, zgal, M_QSO, z_QSO = theta
    elif obj_type == "Q":
        M_QSO, z_QSO = theta
    elif obj_type == "G":
        t0, t1, mass, metallicity, dust_av, zgal = theta
    else:
        raise ValueError(
            f"Invalid obj_type: {obj_type}. Must be either `G`, `Q`, or `GQ`."
        )

    if "G" in obj_type:
        wavs, flxs = galaxy_BAGPIPES_spectroscopy(
            t0, t1, mass, metallicity, dust_av, zgal
        )
        fluxes_model_galaxy = spectrum_to_photometry(wavs, flxs)
        if (
            np.sum(fluxes_model_galaxy) == 0.0
        ):  # Invalid galaxy params => BAGPIPES gives a blank spectrum
            return -np.inf

        fluxes_model += fluxes_model_galaxy

    if "Q" in obj_type:
        wavs, flxs = quasar_spectroscopy(M_QSO, z_QSO)
        fluxes_model_quasar = spectrum_to_photometry(wavs, flxs)
        fluxes_model += fluxes_model_quasar

    sigma2 = yerr**2
    return -0.5 * np.sum((y - fluxes_model) ** 2 / sigma2 + np.log(sigma2))


def log_probability(theta, y, yerr, obj_type):
    """
    Bayesian probability update
    """
    lp = log_prior(theta, obj_type)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, yerr, obj_type)


def suggest_init(obj_type):
    """
    Generates an initial set of parameters, randomly distributed around mean_init_row
    """
    mean_init_row = np.array([1, 0.5, 10, 0.2, 0.2, 0.5, 20, 6.0])
    if obj_type == "G":
        mean_init_row = mean_init_row[:6]
    elif obj_type == "Q":
        mean_init_row = mean_init_row[6:]
    elif obj_type != "GQ":
        raise ValueError(
            f"Invalid obj_type: {obj_type}. Must be either `G`, `Q`, or `GQ`."
        )
    ndim = len(mean_init_row)
    return mean_init_row + np.random.uniform(low=-0.01, high=0.01, size=ndim)


def initialise_chains(nwalkers, flux, err_flux, obj_type):
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
            log_likelihood(new_row, flux, err_flux, obj_type) != -np.inf
        ):
            pos[i, :] = new_row
            i += 1
    return pos


def fit_sed(
    flux, err_flux, obj_type, nsteps=5000, discard=1000, thin=10, nwalkers=32, **kwargs
):  # kwargs to be passed to EnsembleSampler
    """
    Do the MCMC sampling process to SED-fit the fluxes (and their errors) to a G, Q, or GQ model
    Returns each of the samples, and their log_probs, in a big array.
    """

    init = initialise_chains(nwalkers, flux, err_flux, obj_type)
    nwalkers, ndim = init.shape

    with Pool() as pool:  # Multicore processing
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(flux, err_flux, obj_type),
            pool=pool,
            **kwargs,
        )
        sampler.run_mcmc(init, nsteps, progress=True)

    samples = sampler.get_chain(flat=False, thin=thin, discard=discard)
    log_prob = sampler.get_log_prob(flat=False, thin=thin, discard=discard)

    return samples, log_prob


### Running SED fitting ###

nwalkers = 32
n = 10000  # nsteps
d = 4000  # discard
t = 2  # thin
num_samples = int((n - d) / t)

all_samples = [
    np.zeros((len(sylt_ids), num_samples, nwalkers, nparams))
    for nparams in [6, 2, 8]  # G, Q, GQ respectively
]
all_log_probs = [np.zeros((len(sylt_ids), num_samples, nwalkers)) for i in range(3)]

for i, idd in enumerate(sylt_ids):
    photometry = load_grizYJKW12(idd)
    flxs = photometry[:, 0]
    flxerrs = photometry[:, 1]
    for j, ot in enumerate(["G", "Q", "GQ"]):
        samples, log_prob = fit_sed(
            flux=flxs,
            err_flux=flxerrs,
            obj_type=ot,
            nsteps=n,
            discard=d,
            thin=t,
            nwalkers=nwalkers,
            a=2.0,
        )  # samples will be shape (3000,nwalkers,{6/2/8}), log_prob will be (3000,nwalkers)
        all_samples[j][i, :, :, :] = samples
        all_log_probs[j][i, :, :] = log_prob

np.savez_compressed(
    "./mcmc_results.npz",
    ids=sylt_ids,
    samples_G=all_samples[0],
    samples_Q=all_samples[1],
    samples_GQ=all_samples[2],
    log_probs_G=all_log_probs[0],
    log_probs_Q=all_log_probs[1],
    log_probs_GQ=all_log_probs[2],
)
