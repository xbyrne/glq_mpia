"""
fetch_urls.py
Fetches the urls of the objects selected from DES
Said objects are stored in `data/processed/cut_crossmatched_objects.csv`
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import myutils


coords_df = pd.read_csv("./data/processed/cut_crossmatched_objects.csv", index_col=0)[
    ["ra_des", "dec_des"]
]

for coadd_id, (ra, dec) in tqdm(
    coords_df.iterrows(), total=len(coords_df)
):
    # Fetch URLs from SIA service
    url_list = myutils.fetch_object_urls(ra, dec)
    if len(url_list) == 5:
        with open('./data/external/img_url_list.txt', 'a') as f:
            f.write(f'# {coadd_id}\n')
            for url in url_list:
                f.write(f'{url}\n')