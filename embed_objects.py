"""
embed_objects.py
Uses t-distributed Stochastic Neighbor Embedding (t-SNE)
to project the grouping of images into 2D, for better
clustering and visualisation
"""
import numpy as np
from sklearn import manifold

fl = np.load("./data/processed/encoded_imgs.npz")
ids = fl["ids"]  # coadd object ids
enc_imgs = fl["encoded_imgs"]  # 1024D encoded images

embedding = manifold.TSNE(perplexity=40).fit_transform(
    enc_imgs
)  # 2D embedding of the 1024D encoded images.

np.savez_compressed("./data/processed/embedding.npz", ids=ids, embedding=embedding)
