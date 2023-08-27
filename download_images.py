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
all_imgs_array = np.zeros((len(coords_df), 28, 28, 5))
failed_object_mask = np.zeros(len(coords_df)).astype(bool)

BAND_ORDER = {"g": 0, "r": 1, "i": 2, "z": 3, "Y": 4}

for i, (coadd_id, (ra, dec)) in tqdm(
    enumerate(coords_df.iterrows()), total=len(coords_df)
):
    FAILED_OBJECT_FLAG = False  # Innocent until proven guilty
    # Fetch URLs from SIA service
    url_list = myutils.fetch_object_urls(ra, dec)
    if len(url_list) != 5:
        # e.g. if 0 because of some failure;
        #      or 10 because on edge of tile; etc.
        FAILED_OBJECT_FLAG = True
        failed_object_mask[i] = True
        break

    obj_array = np.zeros(
        (28, 28, 5), dtype=np.float32
    )  # Initialising datacube for this object
    for url in url_list:  # Iterating over the bands
        if FAILED_OBJECT_FLAG:
            break

        raw_img = myutils.fetch_image(url)  # Download image from SIA

        processed_img = myutils.crop_image(raw_img, px=28)  # Crop image to 28*28
        if processed_img is None:  # Image had wrong dimension
            FAILED_OBJECT_FLAG = True
            failed_object_mask[i] = True
            break

        band = myutils.band_from_url(url)  # Ordering by wavelength
        obj_array[:, :, BAND_ORDER[band]] = processed_img

    if not FAILED_OBJECT_FLAG:  # Sending datacube for this object to big array
        all_imgs_array[i, :, :, :] = obj_array

success_coadd_ids = coords_df.index[~failed_object_mask]  # Removing failed objects
success_imgs_array = all_imgs_array[~failed_object_mask]

np.savez_compressed(  # Saving...
    "./data/external/images.npz", ids=success_coadd_ids, imgs=success_imgs_array
)
