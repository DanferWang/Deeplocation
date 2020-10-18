import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Partitioning:
    """ Partitioning class to map class labels (i.e. s2 cells) to class indexes.

    """

    def __init__(self, csv_file: Path, shortname=None, skiprows=None):
        """Read partitioning CSV and save number of classes and name of the partitioning

        Args:
            csv_file: path to partitioning CSV
            shortname: short name otherwise shortname is filename without extension
            skiprows: skip the first n rows of the CSV
        """

        logging.info(f'Loading partitioning from file: {csv_file}')
        self._df = pd.read_csv(csv_file, index_col='hex_id', skiprows=skiprows)
        self.nclasses = len(self._df.index)
        self.name = csv_file.stem  # filename without extension
        if shortname:
            self.shortname = shortname
        else:
            self.shortname = self.name

        self._hexids = self._df.index.tolist()

    def __repr__(self):
        return f'{self.name} short: {self.shortname} n: {self.nclasses}'

    def get_class_idx(self, class_label) -> int:
        """Map the class label i.e. hexid to an int index.

        Args:
            class_label: class label

        Returns: class index

        """
        return self._df.loc[class_label, 'class_label']

    def get_lat_lng(self, class_label: str) -> Tuple[float, float]:
        """Return latitude and longitude given the class label

        :param idx: class label
        :return: mean latitude and longitude for given cell
        """
        row = self._df.loc[class_label]
        lat = float(row['latitude_mean'])
        lng = float(row['longitude_mean'])
        return lat, lng

    def get_hexid(self, idx):
        return self._hexids[idx]


def get_partitionings(base_folder: str, filenames: List[str], shortnames: List[str], skiprows=None) -> List[Partitioning]:
    return [Partitioning(Path(base_folder) / fname, shortname, skiprows=skiprows) for fname, shortname in zip(filenames, shortnames)]


class ImageDataset:

    def __init__(self,
                 dataset_path: str,
                 image_base_dir: str,
                 partitionings: List[Partitioning],
                 nrows=None,
                 batch_size=32,
                 target_image_size=224,
                 shuffle=False,
                 validation=False,
                 ):
        """Wrapper class for a tf.data.Dataset

        Args:
            dataset_path: path to a CSV including img_path and one column with class indexes for each partitioning
            image_base_dir: base folder where images are stored
            partitionings: list of partitionings
            nrows: consider only the first n rows from the dataset CSV (dataset_path)
            batch_size: batch size
            target_image_size: network input size
            shuffle: whether the dataset will be shuffled (per instance)
            validation: use a resized center crop as image pre-processing step instead of random crop
        """

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_image_size = target_image_size
        self._image_base_dir = Path(image_base_dir)
        self._validation = validation

        if not self._image_base_dir.exists():
            raise FileNotFoundError(image_base_dir + ' is not a dataset directory')

        logging.info(f'Read CSV from file: {dataset_path}')
        self.df = pd.read_csv(dataset_path, nrows=nrows)
        self.df['img_path'] = self.df['img_path'].apply(lambda p: str(self._image_base_dir / p))
        self.nbatches = int(np.ceil(len(self) / batch_size))

        logging.info('Prepare dataset...')
        with tf.device('/cpu:0'):
            ds_images = tf.data.Dataset.from_tensor_slices(self.df['img_path'].values)
            #ds_p = [tf.data.Dataset.from_tensor_slices(self.df[p].values) for p in self.df.columns[1:].to_list()]
            ds_p = [tf.data.Dataset.from_tensor_slices(self.df[p.name].values) for p in partitionings]

            ds = tf.data.Dataset.zip((ds_images, *ds_p))
            ds = ds.map(self._process_item, num_parallel_calls=AUTOTUNE)
            self.ds = self._prepare_for_iterate(ds)

    def __len__(self):
        return len(self.df.index)

    def _image_preprocessing(self, img_path):
        """Load an JPEG image from file and process it

        Distinguish between training and validation mode.

        Args:
            img_path: full image path to a JPEG encoded image

        Returns: preprocessed image as Tensor

        """

        img_encoded = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_encoded, channels=3)  # decode_image does not return a tensor with shape
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.multiply(tf.subtract(img, 0.5), 2)

        if self._validation:
            # center crop after
            # resizing the image preserving the aspect ratio
            height = tf.cast(tf.shape(img)[0], dtype=tf.float32)
            width = tf.cast(tf.shape(img)[1], dtype=tf.float32)
            max_side_len = tf.maximum(width, height)
            min_side_len = tf.minimum(width, height)
            is_w, is_h = tf.cond(tf.less(width, height), lambda: (0, 1), lambda: (1, 0))
            ratio = self.target_image_size / min_side_len
            offset = (tf.cast(max_side_len * ratio + 0.5, dtype=tf.int32) - self.target_image_size) // 2
            img = tf.image.resize(img,
                                size=[tf.cast(height * ratio + 0.5, dtype=tf.int32),
                                     tf.cast(width * ratio + 0.5, dtype=tf.int32)],
                                     antialias=True)

            bbox = [
                is_h * offset, is_w * offset,
                tf.constant(self.target_image_size),
                tf.constant(self.target_image_size)
            ]
            img_crop = tf.image.crop_to_bounding_box(img, bbox[0], bbox[1], bbox[2], bbox[3])

            return img_crop
        else:
            # random distorted crop
            bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(img),
                bounding_boxes=tf.constant(0, shape=[1, 0, 4]),
                min_object_covered=0.5,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.7, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, distort_bbox = bbox

            img = tf.slice(img, bbox_begin, bbox_size,)
            img = tf.image.random_flip_left_right(img)

        # resize crop to network size
        img = tf.image.resize(img, [self.target_image_size, self.target_image_size])

        return img

    def _process_item(self, img_path, *class_indexes):
        """Wrapper function that delivers a preprocessed image and target values

        Args:
            img_path: full path to an image file
            *class_indexes: list of target values (one class index per partitionign)

        Returns: (X, y)

        """
        img = self._image_preprocessing(img_path)
        return img, class_indexes

    def _prepare_for_iterate(self, ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
        else:
            ds = ds.cache()

        ds = ds.repeat()
        if self.shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.batch(self.batch_size)

        # `prefetch` lets the dataset fetch batches in the background
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
