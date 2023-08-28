"""
download_images.py
Downloads and compiles the images from data/external/img_url_list.txt
"""
import os
import numpy as np
import myutils

BAND_ORDER = {"g": 0, "r": 1, "i": 2, "z": 3, "Y": 4}

with open("./data/external/img_url_list.txt", "r") as f:
    url_filetext = f.read()

urls_list_by_object = url_filetext.split("#")[1:]
num_objs = len(urls_list_by_object)
coadd_ids = np.zeros(num_objs).astype(str)
failed_mask = np.zeros(num_objs).astype(bool)
imgs = np.zeros((num_objs, 28, 28, 5))

DOWNLOADED_FILENAME = "./data/external/wget_file"

for i, object_url_string in tqdm(enumerate(urls_list_by_object)):
    FAILED_OBJECT_FLAG = False  # Innocent until proven guilty

    lines = object_url_string.split()  # ID & list of URLs for a particular object
    coadd_ids[i] = lines[0]  # Extra char at the start?
    urls = lines[1:]

    for url in urls:
        band = myutils.band_from_url(url)  # g/r/i/z/Y

        os.system(
            f"wget -nv -O {DOWNLOADED_FILENAME} '{url}' > /dev/null"
        )  # Downloads file TODO: Could go wrong?
        raw_img = myutils.fetch_image(
            DOWNLOADED_FILENAME
        )  # Extracts image from fits file TODO: Could go wrong?
        processed_img = myutils.crop_image(raw_img)  # Crops image to 28x28
        if processed_img is None:
            FAILED_OBJECT_FLAG = True
            failed_mask[i] = True
            break

        imgs[i, :, :, BAND_ORDER[band]] = processed_img

success_coadd_ids = coadd_ids[~failed_mask]
success_imgs_array = imgs[~failed_mask]
np.savez_compressed(
    "./data/processed/ids_images.npz", ids=success_coadd_ids, imgs=success_imgs_array
)
