from app.settings import BATCH_SIZE, OUTPUT_DIR, EPOCHES, BUFFER_SIZE
from app.helpers import load_data
from app.models.discriminator import Discriminator
from app.models.generator import Generator
from app.models.loss_function import (cross_entropy,
                                      generator_optimizer,
                                      discriminator_optimizer,
                                      generator_objective,
                                      discriminator_objective)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

@tf.function()
def training_step(generator, discriminator, images, k=1, batch_size=32):
	print("training_step(generator, discriminator, images, k=1, batch_size=32)")
	for _ in range(k):
		print(_)
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			print('inside gradients')
			noise = generator.generate_noise(batch_size, 784)
			g_z = generator(noise)
			print(g_z.shape)
			print(noise.shape)
			d_x_true = discriminator(images)  # Trainable?
			d_x_fake = discriminator(g_z)  # dx_of_gx

			discriminator_loss = discriminator_objective(d_x_true, d_x_fake)
			# Adjusting Gradient of Discriminator
			gradients_of_discriminator = disc_tape.gradient(
				discriminator_loss, discriminator.trainable_variables
			)
			discriminator_optimizer.apply_gradients(
				zip(
					gradients_of_discriminator,
					discriminator.trainable_variables
				)
			)  # Takes a list of gradient and variables pairs

			generator_loss = generator_objective(d_x_fake)
			# Adjusting Gradient of Generator
			gradients_of_generator = gen_tape.gradient(
				generator_loss, generator.trainable_variables
			)
			generator_optimizer.apply_gradients(
				zip(
					gradients_of_generator,
					generator.trainable_variables
				)
			)


def training(dataset, epoches):
	for epoch in range(epoches):
		for batch in dataset:
			print(f"batch: {batch}")
			training_step(
				generator, discriminator, batch, batch_size=BATCH_SIZE, k=1
			)

		## After ith epoch plot image
		if (epoch % 50) == 0:
			fake_image = tf.reshape(generator(seed), shape=(28, 28))
			print("{}/{} epoches".format(epoch, epoches))
			# plt.imshow(fake_image, cmap = "gray")
			plt.imsave("{}/{}.png".format(OUTPUT_DIR, epoch), fake_image, cmap="gray")


if __name__ == '__main__':
	seed = np.random.uniform(-1,1, size = (1, 784))
	generator = Generator()
	discriminator = Discriminator()
	(train_images, train_labels), \
	(test_images, test_labels) = load_data.mnist
	train_images = train_images.astype("float32")
	train_images = (train_images - 127.5) / 127.5

	train_dataset = \
		tf.data.Dataset.from_tensor_slices(
			train_images.reshape(train_images.shape[0], 784)
			)\
			.shuffle(BUFFER_SIZE)\
			.batch(BATCH_SIZE)

	training(train_dataset, EPOCHES)
	fake_image = generator(np.random.uniform(-1, 1, size=(1, 784)))
	plt.imshow(tf.reshape(fake_image, shape=(28, 28)), cmap="gray")
