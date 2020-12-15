"""
Author: Sanidhya Mangal
Github: sanidhyamangal
"""
import os  # for os related ops
import time  # for time related ops

import matplotlib.pyplot as plt  # for plotting
import PIL  # for image related ops
import tensorflow as tf  # for deep learning related steps
from IPython import display

from models import (WasserstienDiscriminative,  # load models
                     WasserstienGenerative)
from utils import generate_and_save_images # function to generate and save images


# Get traning dataset
(X_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(X_train, tf.float32))).map(lambda x: (x-127.5)/127.5).shuffle(50000).batch(256)


# intansiate generator
generator = WasserstienGenerative()

print(generator.model.summary())
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])

discriminator = WasserstienDiscriminative()
decision = discriminator(generated_image)
print(decision)

# create a loss for generative
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


# create a loss function for discriminator
def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

# optimizers for the generative models
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# creation of checkpoint dirs
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

# all the epochs for the training
EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 16
BATCH_SIZE = 256

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    """
    TF Graph function to be compiled into a graph
    """
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    """
    Function to perform training ops on given set of epochs
    """
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1,
                                                   time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


train(train_dataset, EPOCHS)