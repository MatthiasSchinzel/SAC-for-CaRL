from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class Encoder:
    def __init__(self):
        # adapt this if using `channels_first` image data format
        input_img = Input(shape=(320, 176, 1))

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((4, 1), padding='same')(x)

        x = Conv2D(1, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((4, 1))(x)
        x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adadelta',
                                 loss='binary_crossentropy')
        self.autoencoder.load_weights(
            "SavedWeights/AutoencoderBDD100kCombination.h5")
        self.encoder = Model(input_img, encoded)

    def Encode_img(self, img):
        img = np.expand_dims(np.expand_dims(img / 255, axis=0), axis=3)
        encoded_imgs = self.encoder.predict(img)
        return np.squeeze(encoded_imgs.flatten())

    def Encode_img_not_flatted(self, img):
        img = np.expand_dims(np.expand_dims(img / 255, axis=0), axis=3)
        encoded_imgs = self.encoder.predict(img)
        return np.squeeze(encoded_imgs)

    def Autoencode(self, img):
        img = np.expand_dims(np.expand_dims(img / 255, axis=0), axis=3)
        encoded_imgs = self.autoencoder.predict(img)
        return np.squeeze(encoded_imgs)
