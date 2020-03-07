"""
Create dataset for training, inspired by https://github.com/surya501/loc2vec/tree/master/openstreetmap-tile-server
"""
import os
from pathlib import Path
import pandas as pd
import re
import tensorflow as tf

from src.utils.geo import *

def get_files_from_path(pathstring="../data/tiles/", zoom=14):
    """retrives file names from the folder and returns a pandas dataframe with
    four columns: path, filesize, lat, long

    Arguments:
        pathstring {string} -- relative location of file

    Returns:
        [pandas dataframe] -- sorted by the filesize
    """

    filenames = []
    for file in Path(os.path.join(pathstring, str(zoom))).glob("**/*.png"):
        filenames.append((str(file), file.stat().st_size, *re.search(".*_\d*_(\d*)_(\d*)", file.stem).groups()))
    files_df = pd.DataFrame(list(filenames),
                            columns=["path", "filesize", "x", "y"])
    sorted_files = files_df.sort_values("filesize")
    result_df = sorted_files.reset_index(drop=True)
    result_df['id'] = result_df.apply(lambda row: tile_id(row['x'], row['y'], zoom=zoom), axis=1)

    return result_df


def load_and_preprocess_image(x, label):
    """Load image and return a (cropped) original and augmented version."""

    image = tf.io.read_file(x)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, dtype='float32')

    # Normalize pixel values
    image = tf.math.divide(image, tf.constant(255, dtype=tf.float32))

    # Original: central crop
    original = tf.image.central_crop(image, central_fraction=0.5) # TODO: should depend on image size

    # Augmented: random crop and rotation
    augmented = tf.image.rot90(image, k=tf.random.uniform(shape=[],
                                                          minval=0, maxval=4,
                                                          dtype=tf.int32, seed=42))
    augmented = tf.image.random_crop(augmented, size=[128, 128, 3])

    return (original, label), (augmented, label)





def get_dataset(df):
    """Create dataset of tiles."""
    dataset = tf.data.Dataset.from_tensor_slices((df['path'], df['id']))
    dataset = dataset.shuffle(1000)

    dataset = dataset.map(lambda f, l: load_and_preprocess_image(f, l))
    dataset = dataset.batch(1)
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.
                               from_tensor_slices(x).
                               concatenate(
        tf.data.Dataset.from_tensor_slices(y)))
    dataset = dataset.batch(16)

    return dataset
