import tensorflow as tf
from tensorflow import keras

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
generator_optimizer = keras.optimizers.RMSprop()
discriminator_optimizer = keras.optimizers.RMSprop()

def generator_objective(dx_of_gx):
    # Labels are true here because generator thinks he produces real images.
    return cross_entropy(tf.ones_like(dx_of_gx), dx_of_gx)

def discriminator_objective(d_x, g_z, smoothing_factor=0.9):
    "d_x: real output & g_z: fake_output"
    real_loss = cross_entropy(
        tf.ones_like(d_x) * smoothing_factor, d_x
    )
    fake_loss = cross_entropy(
        tf.zeros_like(g_z), g_z
    )
    total_loss = real_loss + fake_loss
    return total_loss