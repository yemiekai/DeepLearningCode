import tensorflow as tf

import argparse
import os
import time

from dataset.dataset_utils import *
from models.mobilenet_v3 import *
from test.verify_mobileNetV3_arcFace import *


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--train_datasets_dir', default=r'F:\DeepLearning_DataSet\VGGFace2_train_mtcnnpy_224_tfrecord', help='train datasets base path')
    parser.add_argument('--lfw_test_list', default=r'F:\DeepLearning_DataSet\lfw_test_pair.txt')
    parser.add_argument('--lfw_root', default=r'F:\DeepLearning_DataSet\LFW_mtcnnpy_224')

    parser.add_argument('--gpus', default=2, help='gpu nums')
    parser.add_argument('--batch_size', default=128, help='batch size to train network')
    parser.add_argument('--eval_batch_size', default=64, help='batch size to eval network')
    parser.add_argument('--image_size', default=(224, 224, 3))
    parser.add_argument('--buffer_size', default=12800, help='tf dataset api buffer size')
    parser.add_argument('--num_classes', default=8631, help='classes')
    parser.add_argument('--embedding', default=512, help='classes')

    parser.add_argument('--epoch', default=30, help='epoch to train the network')
    parser.add_argument('--lr_boundaries', default=[10000, 20000, 40000, 80000], help='learning rate to train network')
    parser.add_argument('--lr_values', default=[0.01, 0.008, 0.005, 0.003, 0.001], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    # parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')

    parser.add_argument('--log_file_path', default=r'E:\TrainingCache\mobileNetV3_arcFace_VGGFace_tensorflow', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=10, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=2000, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=1000, help='intervals to save ckpt file')
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

    with tf.Graph().as_default():
        # 查看GPU设备
        # print(tf.test.gpu_device_name())
        # print(tf.test.is_gpu_available())
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        args = get_parser()

        # 设置路径: 保存训练产生的数据
        date = time.strftime("%Y-%m-%d", time.localtime())
        save_path = os.path.join(args.log_file_path, date)  # 保存的文件夹路径
        ckpt_path = os.path.join(save_path, 'ckpt')  # 保存ckpt的路径
        summary_path = os.path.join(save_path, 'summary')  # 保存summary的路径
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(summary_path, exist_ok=True)

        # 设置参数(从ckpt恢复计算图和参数时, 要用到这个name)
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
        images_placeholder = tf.placeholder(name='placeholder_inputs', shape=[None, 224, 224, 3], dtype=tf.float32)
        labels_placeholder = tf.placeholder(name='placeholder_labels', shape=[None, ], dtype=tf.int64)
        isTrain_placeholder = tf.placeholder(name='placeholder_isTrain', dtype=tf.bool)

        # 验证集
        identity_list = get_lfw_list(args.lfw_test_list)  # 所有人名
        lfw_img_paths = [os.path.join(args.lfw_root, each) for each in identity_list]  # 所有图片的路径
        lfw_images_list = read_all(lfw_img_paths, args.image_size)  # 所有图像(numpy数组)

        # 训练集(先要把原图转成tfrecord, 见dataset/conver_VGGFace2)
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
        iterator = dataset.make_initializable_iterator()

        next_element = iterator.get_next()

        # 学习率
        lr = tf.train.piecewise_constant(global_step, boundaries=args.lr_boundaries, values=args.lr_values, name='lr_schedule')
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)

        w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
        model_out, end_points = mobilenet_v3_small(inputs=images_placeholder,
                                                   classes_num=args.embedding,
                                                   multiplier=1.0,
                                                   is_training=isTrain_placeholder,
                                                   reuse=None)
        # model_out_verify, end_points_verify = mobilenet_v3_small(inputs=images_placeholder,
        #                                                          classes_num=args.embedding,
        #                                                          multiplier=1.0,
        #                                                          is_training=isTrain_placeholder,
        #                                                          reuse=True)
        model_out = tf.identity(model_out, 'embeddings')

        arcface_logit = arcface_loss(embedding=model_out,
                                     labels=labels_placeholder,
                                     w_init=w_init_method,
                                     out_num=args.num_classes)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arcface_logit,
                                                                       labels=labels_placeholder,
                                                                       name='cross_entropy_per_example')
        inference_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', inference_loss)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        grads = opt.compute_gradients(total_loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads, global_step=global_step)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # summary writer
        summary = tf.summary.FileWriter(summary_path, sess.graph)
        summaries = []

        # add grad histogram op
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        # add trainabel variable gradients
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        # add learning rate
        summaries.append(tf.summary.scalar('leraning_rate', lr))
        summary_op = tf.summary.merge(summaries)

        # 初始化变量
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        saver = tf.train.Saver()

        # 4 begin iteration
        count = 0

        for i in range(args.epoch):
            sess.run(iterator.initializer)
            while True:
                try:
                    images_train, labels_train = sess.run(next_element)
                    feed_dict = {images_placeholder: images_train,
                                 labels_placeholder: labels_train,
                                 isTrain_placeholder: True}

                    start = time.time()
                    _, _, _total_loss, _lr = sess.run([train_op, inc_op, total_loss, lr], feed_dict=feed_dict)
                    count += 1
                    end = time.time()

                    pre_sec = args.batch_size/(end - start)

                    # 打印训练情况
                    if count > 0 and count % args.show_info_interval == 0:
                        print('epoch %d, total_step %d, lr: %.5f, total loss: %.2f, time %.3f samples/sec' %
                              (i, count, _lr, _total_loss, pre_sec))

                    # save summary
                    if count > 0 and count % args.summary_interval == 0:
                        feed_dict = {images_placeholder: images_train, labels_placeholder: labels_train, isTrain_placeholder:True}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)

                    # 保存模型(ckpt)
                    if count > 0 and count % args.ckpt_interval == 0:
                        filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(ckpt_path, filename)
                        print('save ckpt file: %s' % filename)
                        saver.save(sess, filename)

                    # 验证
                    if count > 0 and count % args.validate_interval == 0:
                        test_on_lfw_when_training(sess, lfw_images_list, identity_list, args.lfw_test_list, args.batch_size,
                                                  model_out, images_placeholder, isTrain_placeholder)

                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break

