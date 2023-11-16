"""
Process the data which results from the cross-matching,
for easier subsequent usage.
Isolates and calculates fluxes, magnitudes, and their errors
in each band.
data/interim/des_wise_vhs_objects.csv
--> data/processed/cut_crossmatched_objects.csv
"""
import numpy as np
import pandas as pd
import myutils

bands = ["g", "r", "i", "z", "Y", "J", "K", "W1", "W2"]

original_df = pd.read_csv(
    "./data/interim/des_wise_vhs_objects.csv", index_col=0
)  # 116499 entries
old_names = [  # All the fields which are useful for selection processing
    "RA",
    "DEC",
    "MAG_APER_4_G",
    "MAGERR_APER_4_G",
    "MAG_APER_4_R",
    "MAGERR_APER_4_R",
    "MAG_APER_4_I",
    "MAGERR_APER_4_I",
    "MAG_APER_4_Z",
    "MAGERR_APER_4_Z",
    "MAG_APER_4_Y",
    "MAGERR_APER_4_Y",
    "Jpmag",
    "e_Jpmag",
    "Kspmag",
    "e_Kspmag",
    "W1mag",
    "e_W1mag",
    "W2mag",
    "e_W2mag",
]
interim_df = original_df.loc[:, old_names]

new_names = [  # Neatening up column names
    # Unless otherwise specified, magnitudes are in AB system and fluxes are in uJy
    "ra_des",
    "dec_des",
    "g_mag",
    "g_magerr",
    "r_mag",
    "r_magerr",
    "i_mag",
    "i_magerr",
    "z_mag",
    "z_magerr",
    "Y_mag",
    "Y_magerr",
    "J_mag_vg",
    "J_magerr",
    "K_mag_vg",
    "K_magerr",
    "W1_mag_vg",
    "W1_magerr",
    "W2_mag_vg",
    "W2_magerr",
]
interim_df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

# Initial cuts: making W1-W2 cut and removing bad detections
interim_df = interim_df[  # 116499
    (interim_df["W1_mag_vg"] - interim_df["W2_mag_vg"] > 0.5)  # 12168
    & ~(interim_df["W1_magerr"].isna())  # 12167
    & ~(interim_df["W2_magerr"].isna())  # 7659
    & ~(interim_df["J_mag_vg"].isna())  # 7641
    & ~(interim_df["K_mag_vg"].isna())  # 7527
]


# Converting Vega magnitudes to AB
for band in ["J", "K", "W1", "W2"]:
    interim_df[f"{band}_mag"] = myutils.vega_to_AB(interim_df[f"{band}_mag_vg"], band)

# Discarding Vega mags
interim_df = interim_df[
    ["ra_des", "dec_des"]
    + np.array([[f"{band}_{param}" for param in ["mag", "magerr"]] for band in bands])
    .ravel()
    .tolist()
]  # ['g_mag','g_magerr','g_flux','g_fluxerr','r_mag'...]


# Converting AB magnitudes to fluxes
# Flux errors calculated using m = -2.5 log(f/f0) => |dm| = 2.5 log(e) df / f
for band in bands:
    mags = interim_df[f"{band}_mag"]
    interim_df.loc[:, f"{band}_flux"] = myutils.AB_to_uJy(mags)

    interim_df[f"{band}_fluxerr"] = (
        interim_df[f"{band}_magerr"]
        * interim_df[f"{band}_flux"]
        / (2.5 * np.log10(np.e))
    )


## Flooring objects below 3sigma detection
# If an object is detected below a 3sigma level,
#  magnitude and error are set to 99;
#  flux set to 0; fluxerr set to 3sigma / 3
# NB nondetections are at this stage at flux=fluxerr=0

# DES, VHS, and WISE all give their sensitivities in different ways

# DES gives 10sigma magnitude depths n the abstract of:
#  https://ui.adsabs.harvard.edu/abs/2021ApJS..255...20A/abstract
des_m_10s = {"g": 24.7, "r": 24.4, "i": 23.8, "z": 23.1, "Y": 21.7}
# The relationship between magnitudes at different significance levels is
# m_asigma - m_bsigma = -2.5 log(F_asigma / F_bsigma) = -2.5 log(a/b)
des_m_3s = {key: m_10s - 2.5 * np.log10(3 / 10) for key, m_10s in des_m_10s.items()}
des_F_3s = {key: myutils.AB_to_uJy(m_3s) for key, m_3s in des_m_3s.items()}

# VHS gives its 5sigma depths here:
#  http://www.eso.org/rm/api/v1/public/releaseDescriptions/144
vhs_m_5s = {"J": 20.8, "K": 20.0}
vhs_m_3s = {key: m_5s - 2.5 * np.log10(3 / 5) for key, m_5s in vhs_m_5s.items()}
vhs_F_3s = {key: myutils.AB_to_uJy(m_3s) for key, m_3s in vhs_m_3s.items()}

# WISE gives 5sigma flux sensitivities in the abstract of:
#  https://ui.adsabs.harvard.edu/abs/2010AJ....140.1868W/abstract
wise_F_5s = {"W1": 0.08e3, "W2": 0.11e3}
# WISE gives 5sigma flux sensitivities here:
#  https://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec2_3a.html#tbl1
wise_F_5s = {"W1": 54, "W2": 71}
wise_F_3s = {key: (3 / 5) * F_5s for key, F_5s in wise_F_5s.items()}
F_3s = {**des_F_3s, **vhs_F_3s, **wise_F_3s}

# We now floor objects which are below {S_N}sigma
S_N = 2
for band in bands:
    to_floor = (
        interim_df[f"{band}_flux"] <= S_N * interim_df[f"{band}_fluxerr"]
    )  # `>=`` accounts for flux=fluxerr=0
    interim_df.loc[
        to_floor, [f"{band}_mag", f"{band}_magerr", f"{band}_flux", f"{band}_fluxerr"]
    ] = [99.0, 99.0, 0.0, F_3s[band] / 3]


## We now remove objects that were not detected (at 2sigma) in JKW12
# HZQs should be detected in these bands!

processed_df = interim_df[  # 7527
    (interim_df["J_flux"] != 0.0)  # 7521
    & (interim_df["K_flux"] != 0.0)  # 7438
    & (interim_df["W1_flux"] != 0)  # 7438
    & (interim_df["W2_flux"] != 0)  # 7438
]

processed_df.to_csv("./data/processed/cut_crossmatched_objects.csv")
