import tensorflow as tf

import argparse
import os
import time

from dataset.conver_VGGFace2 import *
from models.mobilenet_v3 import *


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--train_datasets_dir', default=r'F:\DeepLearning_DataSet\VGGFace2_train_mtcnnpy_224_tfrecord', help='train datasets base path')
    parser.add_argument('--batch_size', default=64, help='batch size to train network')
    parser.add_argument('--buffer_size', default=12800, help='tf dataset api buffer size')
    parser.add_argument('--num_classes', default=8631, help='classes')
    parser.add_argument('--embedding', default=512, help='classes')

    parser.add_argument('--epoch', default=30, help='epoch to train the network')
    parser.add_argument('--lr_boundaries', default=[10000, 20000, 40000, 80000], help='learning rate to train network')
    parser.add_argument('--lr_values', default=[0.5, 0.1, 0.05, 0.001, 0.0001], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    # parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')

    parser.add_argument('--log_file_path', default=r'E:\TrainingCache\mobileNetV3_arcFace_VGGFace_tensorflow', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=1000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=50, help='intervals to save ckpt file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 设置路径--保存训练产生的数据
    date = time.strftime("%Y-%m-%d", time.localtime())
    save_path = os.path.join(args.log_file_path, date)  # 保存的文件夹路径
    ckpt_path = os.path.join(save_path, r'\output\ckpt')  # 保存ckpt的路径
    summary_path = os.path.join(save_path, r'\output\summary')  # 保存summary的路径


    # 1. define global parameters
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
    # 从.tfrecord文件创建dataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_function)
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.shuffle(buffer_size=args.buffer_size)
    dataset = dataset.batch(batch_size=args.batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # 2.2 prepare validate datasets
    # ver_list = []
    # ver_name_list = []
    # for db in args.eval_datasets:
    #     print('begin db %s convert.' % db)
    #     data_set = load_bin(db, args.image_size, args)
    #     ver_list.append(data_set)
    #     ver_name_list.append(db)


    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    model_out, end_points = mobilenet_v3_small(images, args.embedding, multiplier=1.0, is_training=True, reuse=None)

    # 3.2 get arcface loss
    arcface_logit = arcface_loss(embedding=model_out, labels=labels, w_init=w_init_method, out_num=args.num_classes)

    # 3.3 define the cross entropy
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arcface_logit, labels=labels))

    # 3.6 define the learning rate schedule
    lr = tf.train.piecewise_constant(global_step, boundaries=args.lr_boundaries, values=args.lr_values, name='lr_schedule')

    # 3.7 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)

    # 3.8 get train op
    grads = opt.compute_gradients(inference_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)
    # train_op = opt.minimize(inference_loss, global_step=global_step)

    # 3.9 define the inference accuracy used during validate or test
    pred = tf.nn.softmax(arcface_logit)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))

    # 3.10 define sess
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # 3.11 summary writer
    summary = tf.summary.FileWriter(summary_path, sess.graph)
    summaries = []

    # # 3.11.1 add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # 3.11.2 add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # 3.11.3 add loss summary
    summaries.append(tf.summary.scalar('inference_loss', inference_loss))
    # 3.11.4 add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)
    # 3.12 saver
    saver = tf.train.Saver(max_to_keep=args.saver_maxkeep)
    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(iterator.initializer)

    # 4 begin iteration
    if not os.path.exists(args.log_file_path):
        os.makedirs(args.log_file_path)
    log_file_path = args.log_file_path + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
    log_file = open(log_file_path, 'w')
    # 4 begin iteration
    count = 0
    total_accuracy = {}

    for i in range(args.epoch):
        sess.run(iterator.initializer)
        while True:
            try:
                images_train, labels_train = sess.run(next_element)
                feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}

                start = time.time()
                _, inference_loss_val, _, acc_val = sess.run([train_op, inference_loss, inc_op, acc], feed_dict=feed_dict)
                end = time.time()
                pre_sec = args.batch_size/(end - start)

                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, inference loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, inference_loss_val, acc_val, pre_sec))
                count += 1

                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % args.ckpt_interval == 0:
                    filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                    filename = os.path.join(ckpt_path, filename)
                    saver.save(sess, filename)

                # validate
                if count > 0 and count % args.validate_interval == 0:
                    feed_dict_test = {dropout_rate: 1.0}

            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                break

    log_file.close()
    log_file.write('\n')

