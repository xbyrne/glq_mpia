"""
package_data_for_lephare.py
Packages flux data in a lephare input file,
called glq_quasar_data.in
"""
import numpy as np
import pandas as pd

quasar_ids = np.load(
    '../../data/processed/quasar_ids.npz'
)['ids']

df = pd.read_csv(
    '../../data/processed/cut_crossmatched_objects.csv',
    index_col=0
).loc[quasar_ids]

bands = ['g','r','i','z','Y','J','K','W1','W2']
input_filetext = ''
for coi in quasar_ids:
    input_filetext += f'{str(coi)[-9:]} '
    mag_data = df.loc[coi]
    for band in bands:
        mag = mag_data[f'{band}_mag']
        magerr = mag_data[f'{band}_magerr']
        if mag==99.:
            input_filetext += '-99 -99 '
        else:
            input_filetext += f"{mag} {magerr} "
    input_filetext += '\n'

with open('./glq_quasar_data.in', 'w') as f:
    f.write(input_filetext)