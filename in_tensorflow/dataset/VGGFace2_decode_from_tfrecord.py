import os
import time
import tensorflow as tf
from dataset.dataset_utils import *

# reference : https://www.tensorflow.org/guide/datasets?hl=zh-cn#preprocessing_data_with_datasetmap

# 从tf_record读出VGGFace2
if __name__ == "__main__":

    class Argument:
        def __init__(self):
            self.train_datasets_dir = r'F:\DeepLearning_DataSet\VGGFace2_train_mtcnnpy_224_tfrecord'
            self.buffer_size = 12800
            self.batch_size = 120
            self.lfw_test_list = r'E:\DataSets\LFW\lfw_test_pair.txt'

    args = Argument()

    tfrecord_files = []
    for record_file in os.listdir(args.train_datasets_dir):  # dataset_dir所有文件名
        path = os.path.join(args.train_datasets_dir, record_file)
        if path.endswith('.tfrecord'):
            tfrecord_files.append(path)
    dataset = tf.data.TFRecordDataset(tfrecord_files)  # 从.tfrecord文件创建dataset
    dataset = dataset.map(parse_function_VGGFace2)
    # dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.shuffle(buffer_size=args.buffer_size)
    dataset = dataset.batch(batch_size=args.batch_size)
    dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        # 测一下读取数据集的速度
        samples = 0
        start = time.time()
        while True:
            try:

                next_element = iterator.get_next()
                next_element = sess.run(next_element)
                samples += args.batch_size

                t = time.time() - start
                sys.stdout.write('\r>>getting batch Data [%d], time [%.2f]s, speed: [%.2f]/sec' % (samples, t, samples/t))
                sys.stdout.flush()

            except tf.errors.OutOfRangeError:
                print("\nEnd of epoch\n")
                samples = 0
                start = time.time()
                break

