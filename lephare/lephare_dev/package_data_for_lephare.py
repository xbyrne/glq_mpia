"""
package_data_for_lephare.py
Packages flux data in a lephare input file,
called glq_quasar_data.in
"""
import numpy as np
import pandas as pd

quasar_ids = np.load("../../data/processed/quasar_ids.npz")["ids"]

df = pd.read_csv("../../data/processed/cut_crossmatched_objects.csv", index_col=0).loc[
    quasar_ids
]

bands = ["g", "r", "i", "z", "Y", "J", "K", "W1", "W2"]


def AB_to_uJy(mag_AB):
    """
    Converts an AB magnitude to a flux in uJy.
    """
    return 10 ** (29 - (48.60 / 2.5)) * 10 ** (-mag_AB / 2.5)


upper_limit_3sigmas = [
    AB_to_uJy(mag)
    for mag in [
        24.7 - 2.5 * np.log10(3 / 10),
        24.4 - 2.5 * np.log10(3 / 10),
        23.8 - 2.5 * np.log10(3 / 10),
        23.1 - 2.5 * np.log10(3 / 10),
        21.7 - 2.5 * np.log10(3 / 10),
        20.8 - 2.5 * np.log10(3 / 5),
        20.0 - 2.5 * np.log10(3 / 5),
    ]
] + [0.08e-3 * 3 / 5, 0.11e-3 * 3 / 5]

input_filetext = ""

for coi in quasar_ids:
    input_filetext += f"{str(coi)[-9:]} "
    mag_data = df.loc[coi]
    for band in bands:
        mag = mag_data[f"{band}_mag"]
        magerr = mag_data[f"{band}_magerr"]
        if mag == 99.0:
            input_filetext += "-99 -99 "
        else:
            input_filetext += f"{mag} {magerr} "
    input_filetext = input_filetext[:-1]  # Removing trailing space
    input_filetext += "\n"
input_filetext = input_filetext[:-1]  # Removing trailing \n

with open("./glq_quasar_data.in", "w", encoding="utf-8") as f:
    f.write(input_filetext)
