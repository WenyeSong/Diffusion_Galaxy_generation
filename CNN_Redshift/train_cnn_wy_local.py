"""
CNN Redshift Predictor - local Mac version (64x64 images).
Based on Li et al. 2024 (galaxiesml_examples).

Architecture: dual-branch CNN + NN
- CNN branch: 5-channel 64x64 images
- NN branch:  5 photometric magnitudes (g,r,i,z,y cmodel_mag)
"""

import os
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D,
                                     Flatten, Input, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import tensorflow.keras.backend as K

# ─── Paths (local) ────────────────────────────────────────────────────────────
TRAIN_PATH = '/Users/wen/Desktop/project/archive/5x64x64_training_with_morphology.hdf5'
VAL_PATH   = '/Users/wen/Desktop/project/archive/5x64x64_validation_with_morphology.hdf5'

MODEL_DIR  = '/Users/wen/Desktop/project/lizarraga_2024-main/CNN_Redshift/model_output'
CKPT_PATH  = os.path.join(MODEL_DIR, 'checkpoints/cp.weights.h5')
LOG_DIR    = os.path.join(MODEL_DIR, 'logs')
CSV_LOG    = os.path.join(MODEL_DIR, 'training_log.csv')
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE  = 64       # smaller for local Mac
NUM_EPOCHS  = 20
LR          = 1e-4
MAG_KEYS    = ['g_cmodel_mag', 'r_cmodel_mag', 'i_cmodel_mag',
               'z_cmodel_mag', 'y_cmodel_mag']


# ─── Data Generator ───────────────────────────────────────────────────────────
class HDF5Generator(tf.keras.utils.Sequence):
    def __init__(self, path, batch_size=64, shuffle=True, mode='train'):
        self.path       = path
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.mode       = mode
        with h5py.File(path, 'r') as f:
            self.n = f['image'].shape[0]
        self.indices = np.arange(self.n)
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = np.sort(self.indices[idx * self.batch_size:(idx + 1) * self.batch_size])
        with h5py.File(self.path, 'r') as f:
            images = f['image'][batch_idx].astype('float32')
            labels = f['specz_redshift'][batch_idx].astype('float32')
            mags   = np.stack([f[k][batch_idx] for k in MAG_KEYS], axis=-1).astype('float32')

        # per-image min-max normalisation to [0, 1]
        mins = images.min(axis=(1, 2, 3), keepdims=True)
        maxs = images.max(axis=(1, 2, 3), keepdims=True)
        images = (images - mins) / (maxs - mins + 1e-8)

        # normalise magnitudes
        mags = (mags - 20.0) / 5.0

        if self.mode == 'train':
            return (images, mags), labels
        else:
            return (images, mags)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ─── Loss function (HSC metric) ───────────────────────────────────────────────
# HSC photometric redshift loss (Nishizawa et al. 2020):
#
#   L(z_spec, z_photo) = 1 - 1 / (1 + (dz / gamma)^2)
#
# where dz = z_photo - z_spec and gamma = 0.15.
# This loss is bounded in [0, 1] and is less sensitive to large outliers
# than MSE, which suits photometric redshift estimation well.
def hsc_loss(z_spec, z_photo):
    dz    = z_photo - z_spec
    gamma = 0.15
    L     = 1 - 1.0 / (1.0 + K.square(dz / gamma))
    return K.mean(L)


# ─── Model (64x64 version) ────────────────────────────────────────────────────
def build_model():
    # CNN branch
    input_cnn = Input(shape=(5, 64, 64), name='image_input')
    x = Conv2D(32,  (3,3), activation='tanh', padding='same', data_format='channels_first')(input_cnn)
    x = MaxPooling2D((2,2), data_format='channels_first')(x)
    x = Conv2D(64,  (3,3), activation='tanh', padding='same', data_format='channels_first')(x)
    x = MaxPooling2D((2,2), data_format='channels_first')(x)
    x = Conv2D(128, (3,3), activation='tanh', padding='same', data_format='channels_first')(x)
    x = MaxPooling2D((2,2), data_format='channels_first')(x)
    x = Conv2D(256, (3,3), activation='tanh', padding='same', data_format='channels_first')(x)
    x = MaxPooling2D((2,2), data_format='channels_first')(x)
    x = Conv2D(256, (3,3), activation='tanh', padding='same', data_format='channels_first')(x)
    x = MaxPooling2D((2,2), data_format='channels_first')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', data_format='channels_first')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', data_format='channels_first')(x)
    x = Flatten()(x)
    x = Dense(512, activation='tanh')(x)
    x = Dense(128, activation='tanh')(x)
    cnn_out = Dense(32, activation='tanh')(x)

    # NN branch
    input_nn = Input(shape=(5,), name='mag_input')
    y = Dense(200, activation='relu')(input_nn)
    y = Dense(200, activation='relu')(y)
    y = Dense(200, activation='relu')(y)
    y = Dense(200, activation='relu')(y)
    y = Dense(200, activation='relu')(y)
    nn_out = Dense(200, activation='relu')(y)

    merged = Concatenate()([cnn_out, nn_out])
    output = Dense(1)(merged)

    return Model(inputs=[input_cnn, input_nn], outputs=output)


# ─── Train ────────────────────────────────────────────────────────────────────
print("Building model...")
model = build_model()
model.compile(optimizer=Adam(learning_rate=LR),
              loss=hsc_loss,
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()

train_gen = HDF5Generator(TRAIN_PATH, batch_size=BATCH_SIZE, shuffle=True,  mode='train')
val_gen   = HDF5Generator(VAL_PATH,   batch_size=BATCH_SIZE, shuffle=False, mode='train')

callbacks = [
    ModelCheckpoint(CKPT_PATH, save_weights_only=True,
                    monitor='val_loss', mode='min',
                    save_best_only=True, verbose=1),
    TensorBoard(log_dir=LOG_DIR),
    CSVLogger(CSV_LOG, append=True),
]

print(f"Training on {train_gen.n} images, validating on {val_gen.n} images")
model.fit(train_gen, epochs=NUM_EPOCHS, validation_data=val_gen,
          callbacks=callbacks, verbose=1)

print(f"Done. Best model saved to {CKPT_PATH}")
print(f"Training log: {CSV_LOG}")
