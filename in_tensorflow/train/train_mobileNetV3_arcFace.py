import tensorflow as tf
import tensorlayer as tl

import argparse
import os
import time

from dataset.conver_VGGFace2 import *

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--train_datasets_dir', default=r'F:\DeepLearning_DataSet\VGGFace2_train_mtcnnpy_224_tfrecord', help='train datasets base path')
    parser.add_argument('--batch_size', default=64, help='batch size to train network')
    parser.add_argument('--buffer_size', default=12800, help='tf dataset api buffer size')
    # parser.add_argument('--net_depth', default=100, help='resnet depth, default is 50')
    # parser.add_argument('--epoch', default=100000, help='epoch to train the network')
    # parser.add_argument('--lr_steps', default=[40000, 60000, 80000], help='learning rate to train network')
    # parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    # parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    # parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    # parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    # parser.add_argument('--num_output', default=85164, help='the image size')
    # parser.add_argument('--tfrecords_file_path', default='./datasets/tfrecords', type=str,
    #                     help='path to the output of tfrecords file path')
    # parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    # parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    # parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    # parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    # parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    # parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    # parser.add_argument('--ckpt_interval', default=10000, help='intervals to save ckpt file')
    # parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    # parser.add_argument('--show_info_interval', default=20, help='intervals to save ckpt file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 1. define global parameters
    args = get_parser()
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images = tf.placeholder(name='img_inputs', shape=[None, 224, 224, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    # 找到所有.tfrecord文件
    tfrecord_files = []
    for record_file in os.listdir(args.train_datasets_dir):  # dataset_dir所有文件名
        path = os.path.join(args.train_datasets_dir, record_file)
        if path.endswith('.tfrecord'):
            tfrecord_files.append(path)
    # 创建dataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_function)
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.shuffle(buffer_size=args.buffer_size)
    dataset = dataset.batch(batch_size=args.batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()





    # 3.10 define sess
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    images_train, labels_train = sess.run(next_element)

