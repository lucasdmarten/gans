from app.models.generator import Generator
from app.models.discriminator import Discriminator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


generator = Generator()

noise = np.random.uniform(-1,1, size=(1,100))
fake_image = generator(noise)
fake_image = tf.reshape(fake_image, shape=(28,28))
plt.imshow(fake_image)
plt.show()