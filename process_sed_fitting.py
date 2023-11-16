"""
process_sed_fitting.py
Assembles data from LePHARE- and BAGPIPES-based SED fitting
codes to quantify goodness of fit for all.
Saves to /data/processed/sed_fitting_scores.csv
"""

import numpy as np
import pandas as pd

# COADD IDs of objects isolated by ML
quasar_ids = np.load('./data/processed/quasar_ids.npz')['ids']

## LePHARE gives chi^2 for galaxy, quasar, and star models

lephare_df = pd.read_csv(
    './lephare/lephare_dev/glq_quasar_output.out',
    skiprows=55, index_col=0,
    delim_whitespace=True, header=None
)
lephare_cois = [int(str(coi)[-9:]) for coi in quasar_ids]
chi2_g = lephare_df.loc[lephare_cois][5] # Galaxy chi^2
chi2_q = lephare_df.loc[lephare_cois][18] # Quasar
chi2_s = lephare_df.loc[lephare_cois][22] # Star

## BAGPIPES-based MCMC code gives log likelihoods, converted into BIC through

# BIC = k ln n - 2 ln L
# where k is the number of parameters, n the number of fitting points
# n = 9 always: grizYJKW1W2
# k depends on the model
def calc_BIC(coi, model_type):
    """Calculates the BIC for an MCMC-fit model"""
    k = {'G':6,'Q':2,'GQ':8}[model_type]

    fl = np.load(f'./data/sed_fitting/mcmc_results/{model_type}/{coi}.npz')
    BIC = k * np.log(9) - 2*np.max(fl['logprobs'])
    return BIC

BIC_g = np.array([
    calc_BIC(coi, 'G') for coi in quasar_ids
])
BIC_q = np.array([
    calc_BIC(coi, 'Q') for coi in quasar_ids
])
BIC_gq = np.array([
    calc_BIC(coi, 'GQ') for coi in quasar_ids
])

## Assembling dataframe
fit_scores_df = pd.DataFrame(
    index = quasar_ids,
    columns = [
        'LePHARE_chi2_G',
        'LePHARE_chi2_Q',
        'LePHARE_chi2_S',
        'BAGPIPES_BIC_G',
        'BAGPIPES_BIC_Q',
        'BAGPIPES_BIC_GQ',
    ]
)

fit_scores_df['LePHARE_chi2_G'] = chi2_g.values
fit_scores_df['LePHARE_chi2_Q'] = chi2_q.values
fit_scores_df['LePHARE_chi2_S'] = chi2_s.values

fit_scores_df['BAGPIPES_BIC_G'] = BIC_g
fit_scores_df['BAGPIPES_BIC_Q'] = BIC_q
fit_scores_df['BAGPIPES_BIC_GQ'] = BIC_gq

fit_scores_df.to_csv('./data/sed_fitting/sed_fitting_scores.csv')