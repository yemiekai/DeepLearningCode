

import random
import os
import tensorflow as tf
import sys
import math
import dataset.dataset_utils as dataset_utils

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of images in the validation set.
_NUM_VALIDATION = 3000

# The number of shards per dataset split.
_NUM_SHARDS = 50  # 数据集碎片数


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'VGGFace2_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _get_filenames_and_classes(dataset_dir):
    """
    返回所有图片的文件名(列表)和其分类

    Args:
        dataset_dir: A directory containing a set of subdirectories representing
            class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
        A list of image file paths, relative to `dataset_dir` and the list of
        subdirectories, representing class names.
    """

    directories = []  # 文件夹路径
    class_names = []

    for classname in os.listdir(dataset_dir):  # dataset_dir所有文件名
        path = os.path.join(dataset_dir, classname)
        if os.path.isdir(path):
            directories.append(path)    # 每种分类的文件夹
            class_names.append(classname)    # 分类类型

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)    # 每张图片的路径

    return photo_filenames, class_names


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
            (integers).
        dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))  # ceil:向下取整

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        # 根据路径获得该图片分类
                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def run(dataset_dir):

    # 获取分好类的所有文件名, 分类类型
    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)

    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    # zip(class_names, range(len(class_names)))的结果是:  [(daisy,0), (dandelion,1), ... }

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)

    # 切分训练集和测试集
    # training_filenames = photo_filenames[_NUM_VALIDATION:]
    # print("length of training:{}".format(len(training_filenames)))
    # validation_filenames = photo_filenames[:_NUM_VALIDATION]
    # print("length of validation:{}".format(len(validation_filenames)))

    # First, convert the training and validation sets.
    _convert_dataset('train', photo_filenames, class_names_to_ids, dataset_dir)
    # _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    print('\nFinished converting the Exp dataset!')


if __name__ == "__main__":
    run(r"E:\DataSets\VGGFace2\VGGFace2_test_mtcnnpy_224")

