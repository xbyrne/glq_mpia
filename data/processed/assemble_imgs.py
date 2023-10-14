"""
assemble_imgs.py
Reassembles split image files
"""
import numpy as np

imgs1 = np.load('./imgs1.npz')['imgs']
imgs2 = np.load('./imgs2.npz')['imgs']

# Combine img arrays
imgs = np.concatenate((imgs1, imgs2))

np.savez_compressed('./imgs.npz', imgs=imgs)
