import tensorflow as tf

import argparse
import os
import time

from in_tensorflow.dataset.dataset_utils import *
from in_tensorflow.models.mobilenet_v3 import *
from in_tensorflow.test.verify_mobileNetV3_arcFace import *


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--train_datasets_dir', default=r'F:\DeepLearning_DataSet\VGGFace2_train_mtcnnpy_224_tfrecord', help='train datasets base path')
    parser.add_argument('--lfw_test_list', default=r'F:\DeepLearning_DataSet\lfw_test_pair.txt')
    parser.add_argument('--lfw_root', default=r'F:\DeepLearning_DataSet\LFW_mtcnnpy_224')

    parser.add_argument('--gpus', default=2, help='gpu nums')
    parser.add_argument('--batch_size', default=256, help='batch size to train network')
    parser.add_argument('--eval_batch_size', default=128, help='batch size to eval network')
    parser.add_argument('--image_size', default=(224, 224, 3))
    parser.add_argument('--buffer_size', default=25600, help='tf dataset api buffer size')
    parser.add_argument('--num_classes', default=8631, help='classes')
    parser.add_argument('--embedding', default=512, help='classes')

    parser.add_argument('--epoch', default=30, help='epoch to train the network')
    parser.add_argument('--lr_boundaries', default=[10000, 20000, 40000, 80000], help='learning rate to train network')
    parser.add_argument('--lr_values', default=[0.01, 0.008, 0.005, 0.001, 0.0001], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    # parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')

    parser.add_argument('--log_file_path', default=r'E:\TrainingCache\mobileNetV3_arcFace_VGGFace_tensorflow\GPU', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=1000, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=500, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=500, help='intervals to save eval model')
    parser.add_argument('--show_info_interval', default=25, help='intervals to save ckpt file')
    args = parser.parse_args()
    return args


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    print(tf.test.gpu_device_name())
    print(tf.test.is_gpu_available())

    args = get_parser()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    # 设置路径--保存训练产生的数据
    date = time.strftime("%Y-%m-%d", time.localtime())
    save_path = os.path.join(args.log_file_path, date)  # 保存的文件夹路径
    ckpt_path = os.path.join(save_path, 'ckpt')  # 保存ckpt的路径
    summary_path = os.path.join(save_path, 'summary')  # 保存summary的路径
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(summary_path, exist_ok=True)

    # 1. define global parameters
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images_placeholder = tf.placeholder(name='placeholder_inputs', shape=[None, 224, 224, 3], dtype=tf.float32)
    labels_placeholder = tf.placeholder(name='placeholder_labels', shape=[None, ], dtype=tf.int64)
    isTrain_placeholder = tf.placeholder(name='placeholder_isTrain', dtype=tf.bool)

    # splits input to different gpu
    images_s = tf.split(images_placeholder, num_or_size_splits=args.gpus, axis=0)
    labels_s = tf.split(labels_placeholder, num_or_size_splits=args.gpus, axis=0)

    # 验证集
    identity_list = get_lfw_list(args.lfw_test_list)
    lfw_img_paths = [os.path.join(args.lfw_root, each) for each in identity_list]  # 所有图片的路径

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
    dataset = dataset.map(parse_function_VGGFace2)
    # dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.shuffle(buffer_size=args.buffer_size)
    dataset = dataset.batch(batch_size=args.batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # 验证集
    identity_list = get_lfw_list(args.lfw_test_list)  # 所有人名
    lfw_img_paths = [os.path.join(args.lfw_root, each) for each in identity_list]  # 所有图片的路径
    lfw_images_list = read_all(lfw_img_paths, args.image_size)  # 所有图像(numpy数组)

    lr = tf.train.piecewise_constant(global_step, boundaries=args.lr_boundaries, values=args.lr_values, name='lr_schedule')
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)

    # Calculate the gradients for each model tower.
    tower_grads = []
    loss_dict = {}
    loss_keys = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(args.gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('mobileNetV3_tower_%d' % i) as scope:
                    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
                    model_out, end_points = mobilenet_v3_large(images_s[i], args.embedding, multiplier=1.0,
                                                               is_training=isTrain_placeholder, reuse=None)
                    arcface_logit = arcface_loss(embedding=model_out, labels=labels_s[i], w_init=w_init_method,
                                                 out_num=args.num_classes)
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arcface_logit,
                                                                                   labels=labels_s[i],
                                                                                   name='cross_entropy_per_example')
                    total_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

                    # tf.add_to_collection('losses', inference_loss)

                    # losses = tf.get_collection('losses', scope)
                    # total_loss = tf.add_n(losses, name='total_loss')

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(total_loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

                    loss_dict[('total_loss_%s_%d' % ('gpu', i))] = total_loss
                    loss_keys.append(('total_loss_%s_%d' % ('gpu', i)))

                    if i == 0:
                        model_out_verify, end_points_verify = mobilenet_v3_large(inputs=images_placeholder,
                                                                                 classes_num=args.embedding,
                                                                                 multiplier=1.0,
                                                                                 is_training=isTrain_placeholder,
                                                                                 reuse=True)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Apply the gradients to adjust the shared variables.
    train_op = opt.apply_gradients(grads, global_step=global_step)

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
    # 3.11.4 add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)
    # 3.12 saver

    # 3.13 init all variables
    sess.run(tf.global_variables_initializer(), feed_dict={isTrain_placeholder: False})
    sess.run(tf.local_variables_initializer(), feed_dict={isTrain_placeholder: False})
    sess.run(iterator.initializer)

    # 4 begin iteration
    count = 0
    total_accuracy = {}

    saver = tf.train.Saver()

    for i in range(args.epoch):
        sess.run(iterator.initializer)
        while True:
            try:
                start = time.time()
                images_train, labels_train = sess.run(next_element)
                feed_dict = {images_placeholder: images_train,
                             labels_placeholder: labels_train,
                             isTrain_placeholder: True}

                _, _, total_loss_gpu_1, total_loss_gpu_2, _lr = sess.run(
                    [train_op, inc_op, loss_dict[loss_keys[0]], loss_dict[loss_keys[1]], lr], feed_dict=feed_dict)

                end = time.time()
                pre_sec = args.batch_size/(end - start)

                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, lr: %.5f, total loss: [%.2f, %.2f], time %.3f samples/sec' %
                          (i, count, _lr, total_loss_gpu_1, total_loss_gpu_2, pre_sec))
                count += 1

                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    print('count = %d, save summary' % count)
                    feed_dict = {images_placeholder: images_train, labels_placeholder: labels_train, isTrain_placeholder:True}
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % args.ckpt_interval == 0:
                    filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                    filename = os.path.join(ckpt_path, filename)
                    print('save ckpt file: %s' % filename)
                    saver.save(sess, filename)

                # validate
                if count > 0 and count % args.validate_interval == 0:
                    test_on_lfw_when_training(sess, lfw_images_list, identity_list, args.lfw_test_list, args.batch_size,
                                              model_out_verify, images_placeholder, isTrain_placeholder)

            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                break

