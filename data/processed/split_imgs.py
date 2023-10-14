"""
split_imgs.py
Splits the images file into two so that it can be uploaded
without git LFS
"""
import numpy as np

fl = np.load('./imgs.npz')
imgs = fl['imgs']

splitpoint = imgs.shape[0] // 2

imgs1 = imgs[:splitpoint]
imgs2 = imgs[splitpoint:]

np.savez_compressed('./imgs1.npz', imgs=imgs1)
np.savez_compressed('./imgs2.npz', imgs=imgs2)