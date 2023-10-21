"""
package_data_for_lephare.py
Packages flux data in a lephare input file,
called glq_quasar_data.in
"""
import numpy as np
import pandas as pd

quasar_ids = np.load("./data/processed/quasar_ids.npz")["ids"].astype(int)

df = pd.read_csv("./data/processed/cut_crossmatched_objects.csv", index_col=0).loc[
    quasar_ids
]

bands = ["g", "r", "i", "z", "Y", "J", "K", "W1", "W2"]


def AB_to_uJy(mag_AB):
    """
    Converts an AB magnitude to a flux in uJy.
    """
    return 10 ** (29 - (48.60 / 2.5)) * 10 ** (-mag_AB / 2.5)


def uJy_to_ergscm2Hz(flx_uJy):
    """Converts microJansky to erg/s/cm2/Hz"""
    return flx_uJy * 1e-29


input_filetext = ""

for coi in quasar_ids:
    input_filetext += f"{str(coi)[-9:]} "
    flux_data = df.loc[coi]
    for band in bands:
        mag = flux_data[f"{band}_mag"]
        flx = flux_data[f"{band}_flux"]
        flxerr = flux_data[f"{band}_fluxerr"]
        if mag == 99.0:
            # Use 3sigma as upper bound to flux; -1.0 for the error
            input_filetext += f"{3*uJy_to_ergscm2Hz(flxerr)} -1.0 "
        else:
            input_filetext += f"{uJy_to_ergscm2Hz(flx)} {uJy_to_ergscm2Hz(flxerr)} "
    input_filetext = input_filetext[:-1]  # Removing trailing space
    input_filetext += "\n"
input_filetext = input_filetext[:-1]  # Removing trailing \n

with open("./lephare/lephare_dev/glq_quasar_data.in", "w", encoding="utf-8") as f:
    f.write(input_filetext)
