import pandas as pd
from astropy import units as u
from tqdm import tqdm

bands = ['g','r','i','z','Y','J','K','W1','W2']

full_df = pd.read_csv('./data/interim/des_wise_vhs_objects.csv', index_col=0)
old_names = [ # All the fields which are useful for selection processing
    'RA','DEC',
    'MAG_APER_4_G','MAGERR_APER_4_G',
    'MAG_APER_4_R','MAGERR_APER_4_R',
    'MAG_APER_4_I','MAGERR_APER_4_I',
    'MAG_APER_4_Z','MAGERR_APER_4_Z',
    'MAG_APER_4_Y','MAGERR_APER_4_Y',
    'Jpmag','e_Jpmag',
    'Kspmag','e_Kspmag',
    'FW1lbs','e_FW1lbs',
    'FW2lbs','e_FW2lbs'
]
neat_df = full_df[old_names]

new_names = [
    'ra_des','dec_des',
    'g_mag','g_magerr',
    'r_mag','r_magerr',
    'i_mag','i_magerr',
    'z_mag','z_magerr',
    'Y_mag','Y_magerr',
    'J_mag','J_magerr',
    'K_mag','K_magerr',
    'W1_flux','W1_fluxerr',
    'W2_flux','W2_fluxerr'
]
neat_df.rename(columns = dict(zip(old_names, new_names)), inplace=True)