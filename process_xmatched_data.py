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

original_df = pd.read_csv("./data/interim/des_wise_vhs_objects.csv", index_col=0)
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
interim_df = original_df[old_names]

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


# Converting Vega magnitudes to AB
for band in ["J", "K", "W1", "W2"]:
    interim_df[f"{band}_mag"] = myutils.vega_to_AB(interim_df[f"{band}_mag_vg"], band)

# Converting AB magnitudes to fluxes
# Flux errors calculated using m = -2.5 log(f/f0) => |dm| = 2.5 log(e) df / f
for band in bands:
    interim_df[f"{band}_flux"] = myutils.AB_to_uJy(interim_df[f"{band}_mag"])
    interim_df[f"{band}_fluxerr"] = (
        interim_df[f"{band}_magerr"]
        * interim_df[f"{band}_flux"]
        / (2.5 * np.log10(np.e))
    )

# Slicing again to select useful columns
interim_df = interim_df[
    ["ra_des", "dec_des"]
    + np.array(
        [
            [f"{band}_{param}" for param in ["mag", "magerr", "flux", "fluxerr"]]
            for band in bands
        ]
    )
    .ravel()
    .tolist()
]  # ['g_mag','g_magerr','g_flux','g_fluxerr','r_mag'...]

## Flooring objects below 3sigma detection
# If an object is detected below a 3sigma level, magnitude and error
#  are set to 99.; the flux is set to 0

# We now floor objects which are below {S_N}sigma
S_N = 3
for band in bands:
    # to_floor = interim_df[f"{band}_flux"] < F_3s[band]
    to_floor = interim_df[f"{band}_flux"] < S_N * interim_df[f"{band}_fluxerr"]
    # The below might throw some SettingWithCopyWarnings but it seems to work fine
    interim_df.loc[to_floor, [f"{band}_mag", f"{band}_magerr", f"{band}_flux"]] = [
        99.0,
        99.0,
        0.0,
    ]


## We now make some more cuts, to ensure
# (W1, W2) != 99.             [ensuring detected in WISE]
# (dW1, dW2) < 2.5 log(e) / 3 [ensuring S/N>3]
# W1 - W2 > 0.5 [Vega]        [Removes many dwarf stars]
# (J, K) != NaN.              [ensuring detected in VHS]

cut = (
    (interim_df["W1_mag"] != 99.0)  # 116473
    & (interim_df["W2_mag"] != 99.0)  # 106509
    & (interim_df["W1_magerr"] < 2.5 * np.log10(np.e) / 3)  # 106486
    & (interim_df["W2_magerr"] < 2.5 * np.log10(np.e) / 3)  # 92954
    & (interim_df["W1_mag"] - interim_df["W2_mag"] > 0.5 + 2.699 - 3.339)  # 7038
    & ~(interim_df["J_mag"].isna())  # 7020
    & ~(interim_df["K_mag"].isna())  # 6923
)

processed_df = interim_df[cut]
# processed_df.to_csv("./data/processed/cut_crossmatched_objects.csv")
