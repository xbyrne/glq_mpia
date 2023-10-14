"""
contrastive_utils.py
Utils for contrastive learning
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Reshape,
    Concatenate,
    MaxPooling2D as MP,
    RandomFlip,
    RandomRotation,
    RandomCrop,
    RandomTranslation,
)
from tensorflow.keras.models import Model


def normalise_imgs(imgs):
    """Normalises each image to its brightest pixel."""
    normalised_imgs = (
        imgs / np.max(imgs, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    )
    return normalised_imgs


## Model
def load_augmentor():
    """
    Loads the augmentor for the contrastive learning network.
    Takes as input a 28x28x5 image and makes several random transformations:
    1. Randomly crops to 24x24x5
    2. Random chance of flipping the image vertically/horizontally
    3. Randomly translates the image by 2 pixels (wrapping)
    4. Randomly rotates by any angle
    """
    input_images = Input(shape=(28, 28, 5), name="img_input")
    x = RandomCrop(24, 24, name="cropper")(input_images)
    x = RandomFlip(name="flipper")(x)
    translate_px = 2
    x = RandomTranslation(
        translate_px / 24, translate_px / 24, fill_mode="wrap", name="translator"
    )(x)
    augmented_images = RandomRotation(0.5, name="rotator")(x)
    augmentor = Model(input_images, augmented_images, name="augmentor")
    return augmentor


def load_encoder(n_filters_1=256, n_filters_2=512, n_filters_3=1024):
    """
    Loads the encoder of the contrastive learning network.
    Essentially a classic ConvNet, converting an image from
    the cropped 24x24x5 size to a 1024-dimensional vector.
    """
    augmented_images = Input(shape=(24, 24, 5), name="augmented_input")
    x = Conv2D(n_filters_1, 5, activation="relu", name="Conv2D_0")(augmented_images)
    x = MP(pool_size=2)(x)
    x = Conv2D(n_filters_2, 3, activation="relu", name="Conv2D_1")(x)
    x = MP(pool_size=2)(x)
    x = Conv2D(n_filters_3, 3, activation="relu", name="Conv2D_2")(x)
    x = MP(pool_size=2)(x)
    features = Reshape((n_filters_3,), name="Reshape")(x)
    encoder = Model(augmented_images, features, name="encoder")
    return encoder


def load_projector(input_size=1024, n_nodes_1=512, n_nodes_2=128, n_nodes_3=64):
    """
    Loads the projector for the contrastive learning network.
    This gets thrown away after training because for some reason that makes it work better.
    Simply a few dense layers, converting 1024D vectors to 64D
    """
    features = Input(shape=(input_size,), name="features")
    x = Dense(n_nodes_1, activation="relu", name="Dense_0")(features)
    x = Dense(n_nodes_2, activation="relu", name="Dense_1")(x)
    projection = Dense(n_nodes_3, name="Dense_2")(x)
    projector = Model(features, projection, name="projector")
    return projector


@tf.function
def boltz_matrix(normed_mini_batch, temp=1):
    """
    An intermediate step in calculating the loss for a given batch
    Called by `contrastive_loss`
    Inputs: A mini-batch of (normalised) vectors z_i
            A temperature parameter, default=1
    Output: A matrix with the values B_ij = exp(sim(z_i, z_j)/temp) if i!=j, 0 if i==j.
                This matrix has the shape (2*batch_size, 2*batch_size)
    """
    raw_bm = tf.math.exp(
        tf.linalg.matmul(normed_mini_batch, normed_mini_batch, transpose_b=True) / temp
    )  # exp(sim(z_i, z_j)/temp) forall i, j
    return tf.linalg.set_diag(
        raw_bm, tf.zeros_like(normed_mini_batch[:, 0])
    )  # zeroing out where i==j


@tf.function
def contrastive_loss(
    mini_batch, temp=1
):  # Minibatch(i) has shape (batch_size*2, projection_dim)
    """
    Calculates contrastive loss for a given mini-batch
    Inputs: A mini-batch of N-dimensional vectors output by the projector of the CNN
            A temperature parameter, default=1
    Output: The scalar contrastive loss for that mini-batch
    """
    batch_size = tf.shape(mini_batch)[0] // 2  # Should be even anyway
    normed_mini_batch, _ = tf.linalg.normalize(
        mini_batch, axis=1
    )  # Loss only needs directions, not magnitudes
    boltz_mat = boltz_matrix(normed_mini_batch, temp=temp)
    positive_boltzs = tf.linalg.diag_part(
        tf.roll(boltz_mat, batch_size, 0)
    )  # Positive samples
    boltz_norms = tf.math.reduce_sum(boltz_mat, axis=0)
    return tf.math.reduce_mean(
        -tf.math.log(tf.math.divide(positive_boltzs, boltz_norms))
    )


class Contrastor(Model):
    """
    Contrastor class, a subclass of the TF Model class
    """

    def __init__(self, augmentor, encoder, projector, temperature=1, **kwargs):
        super().__init__(**kwargs)
        self.augmentor = augmentor
        self.encoder = encoder
        self.projector = projector
        self.loss_tracker = tf.keras.metrics.Mean(name="contrastive_loss")
        self.temperature = temperature

    @property
    def metrics(self):
        """Loss metrics"""
        return [self.loss_tracker]

    def train_step(self, data):
        """
        Forward pass and one training step
        """
        with tf.GradientTape() as tape:
            # Forward Pass
            projection1 = self.projector(self.encoder(self.augmentor(data)))
            projection2 = self.projector(self.encoder(self.augmentor(data)))
            mini_batch = Concatenate(axis=0)([projection1, projection2])

            cl = contrastive_loss(mini_batch, temp=self.temperature)

        grads = tape.gradient(cl, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(cl)
        return {"loss": self.loss_tracker.result()}
