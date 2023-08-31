"""
contrastive_learning.py
Trains a neural network to separate quasars from other objects using contrastive learning
"""

import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import umap
import contrastive_utils

fl = np.load("./data/processed/ids_images.npz")
ids = fl["ids"]
imgs = fl["imgs"]

# Loading model
contrastor = contrastive_utils.Contrastor(
    contrastive_utils.load_augmentor(),
    contrastive_utils.load_encoder(),
    contrastive_utils.load_projector(),
    temperature=0.015,
)
# Compiling model
contrastor.compile(
    optimizer=optimizers.SGD(learning_rate=1e-4, momentum=1e-4), run_eagerly=True
)
# Training model
history = contrastor.fit(
    contrastive_utils.normalise_imgs(imgs),
    batch_size=128,
    callbacks=[EarlyStopping(monitor="loss", mode="min", patience=5, verbose=1)],
)

# Forward passing the original images
encoded_imgs = contrastor.encoder(
    contrastive_utils.normalise_imgs(imgs[:, 2:-2, 2:-2, :])
).numpy()
# Normalising them; can't remember why. This is probably not a good way to do it either.
encoded_imgs = np.array([enc_img / np.linalg.norm(enc_img) for enc_img in encoded_imgs])

embedding = umap.UMAP().fit(encoded_imgs)
points = embedding.embedding_

np.savez_compressed("./data/processed/umap_embedding.npz", ids=ids, points=points)
