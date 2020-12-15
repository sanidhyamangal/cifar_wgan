from models import WasserstienGenerative  # generative models
import tensorflow as tf  # for deep learning based ops
from utils import generate_and_save_images  # function to generate and save images

import os  # for os related ops


class CifarImagesGenerator:
    """
    Class for generating cifar images
    """
    def __init__(self,
                 path_to_checkpoint: str = './training_checkpoints',
                 *args,
                 **kwargs):
        self.generator = WasserstienGenerative()
        # creation of checkpoint dirs

        checkpoint_prefix = os.path.join(path_to_checkpoint, "ckpt")
        checkpoint = tf.train.Checkpoint(generator=self.generator)

        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def generate_images(self, image_name: str = "generated"):
        seed = tf.random.normal([16, 100])
        generate_and_save_images(self.generator, image_name, seed)
