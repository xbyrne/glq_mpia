"""
combine_img_files.py
Combines the files `ids_images_{1,2}.npz`
"""
import numpy as np

fls = [np.load(f"./data/processed/ids_images_{i}.npz") for i in [1, 2]]

ids = np.concatenate([fl["ids"] for fl in fls], axis=0)
imgs = np.concatenate([fl["imgs"] for fl in fls], axis=0)
np.savez_compressed("./data/processed/ids_images.npz", ids=ids, imgs=imgs)
