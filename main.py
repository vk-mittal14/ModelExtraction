import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import tensorflow_hub as hub
hub_url = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"
``
import argparse
parser = argparse.
parser.add_argument(
    "--lr",
    default=5e-4,
    type=float,
    help="Learning Rate (default: 5e-4)",
)
parser.add_argument(
    "--weight-decay",
    default=1e-5,
    type=float,
    help="Weight Decay (default: 1e-5)",
)
parser.add_argument(
    "--bs",
    default=1,
    type=int,
    help="Batch Size (default: 1)",
)

parser.add_argument(
    "--num-epochs",
    default=100,
    type=int,
    help="Number of Epochs (default: 100)",
)

encoder = hub.KerasLayer(hub_url, trainable=False)

inputs = tf.keras.layers.Input(
    shape=[None, None, None, 3],
    dtype=tf.float32,
    name='image')

# [batch_size, 600]
outputs = encoder(dict(image=inputs))

victim = tf.keras.Model(inputs, outputs, name='movinet')

ind__ = 0
batch_size = 8

c = np.zeros(600).reshape(1,-1)
c[0,ind__] = 1 
sample = tf.ones((1,8,224,224,3))
results = np.array(victim(sample))

print(np.argmax(results))
cce = tf.keras.losses.CategoricalCrossentropy()

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv3D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling3D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv3D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling3D(2)(x)

        # Project residual
        residual = layers.UpSampling3D(2)(previous_block_activation)
        residual = layers.Conv3D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, tf.expand_dims(victim(outputs), 1))
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model((8,224,224), 3)
model.summary()

def iitmandi_loss(X, y):
  c_hat = np.array(victim(X))
  return cce(y,c_hat)

print(iitmandi_loss(sample, c))
points = [0]*1000
AUTO = tf.data.experimental.AUTOTUNE
def data_gen(i):
  return tf.random.uniform((8,224,224,3)), c

data = tf.data.Dataset.from_tensor_slices(points)

data = (
    data
    .map(data_gen, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(batch_size)
    .prefetch(AUTO)
)

model.compile(optimizer='adam', loss = 'categorical_crossentropy')

z = model.predict(sample)
model.fit(data, epochs = 10, verbose = 2)
