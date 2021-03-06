import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

import argparse
import os
import time

from dataset.dataset_utils import *
from models.mobilenet_v3 import *
from test.verify_mobileNetV3_arcFace import *


def write_log_file(filename, content):
    with open(log_filename, 'a') as fout:
        fout.write(content + '\n')


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--train_datasets_dir', default=r'E:\DeepLearning_DataSet\VGGFace2_funneled_align_double_112_tfrecord', help='train datasets base path')
    parser.add_argument('--lfw_test_list', default=r'E:\DeepLearning_DataSet\lfw_test_pair_jpg.txt')
    parser.add_argument('--lfw_root', default=r'E:\DeepLearning_DataSet\LFW_funneled_align_112')
    parser.add_argument('--model', default='small')

    parser.add_argument('--batch_size', default=512, help='batch size to train network')
    parser.add_argument('--eval_batch_size', default=64, help='batch size to eval network')
    parser.add_argument('--image_size', default=(112, 112, 3))
    parser.add_argument('--buffer_size', default=12800, help='tf dataset api buffer size')
    parser.add_argument('--num_classes', default=9131, help='VGGFace2:9131, MS1N:85164')
    parser.add_argument('--embedding', default=512, help='classes')

    parser.add_argument('--epoch', default=40, help='epoch to train the network')
    parser.add_argument('--lr_boundaries', default=[25000, 80000, 120000, 160000, 200000, 400000], help='learning rate to train network')
    parser.add_argument('--lr_values', default=[0.01, 0.008, 0.005, 0.003, 0.002, 0.001, 0.0005], help='learning rate to train network')
    # parser.add_argument('--lr_boundaries', default=[50000, 160000, 240000, 320000, 400000, 800000],
    #                     help='learning rate to train network')
    # parser.add_argument('--lr_values', default=[0.01, 0.008, 0.005, 0.003, 0.002, 0.001, 0.0005],
    #                     help='learning rate to train network')

    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--l2_weight_decay', default=1e-5, help=' ')
    parser.add_argument('--drop_out', default=0.8, help=' ')
    parser.add_argument('--bn_average_decay', default=0.99, help='batch-normalization layers with average decay')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    # parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')

    parser.add_argument('--log_file_path', default=r'F:\TrainingCache\mobileNetV3_samll_arcFace_VGGFace2_tensorflow', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=2000, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save eval model')
    parser.add_argument('--show_info_interval', default=50, help='intervals to save ckpt file')
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
    # 查看GPU设备
                
    # print(tf.test.gpu_device_name())
    # print(tf.test.is_gpu_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使0号显卡可见

    with tf.Graph().as_default():

        args = get_parser()

        # 设置路径: 保存训练产生的数据
        date = time.strftime("%Y-%m-%d", time.localtime())  # 当前日期(字符串)
        # date += "-1"
        save_path = os.path.join(args.log_file_path, date)  # 保存的文件夹路径
        log_filename = os.path.join(save_path, 'Console_Log.txt')  # 日志路径
        tflite_filename = os.path.join(save_path, 'mobileNetV3_small_insightFace.tflite')  # tflite路径
        ckpt_path = os.path.join(save_path, 'ckpt')  # 保存ckpt的路径
        summary_path = os.path.join(save_path, 'summary')  # 保存summary的路径

        # 创建文件夹
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(summary_path, exist_ok=True)

        # 设置参数(从ckpt恢复计算图和参数时, 要用到这个name)
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
        images_placeholder = tf.placeholder(name='placeholder_inputs', shape=[None, 112, 112, 3], dtype=tf.float32)
        labels_placeholder = tf.placeholder(name='placeholder_labels', shape=[None, ], dtype=tf.int64)
        isTrain_placeholder = tf.placeholder(name='placeholder_isTrain', dtype=tf.bool)

        # 验证集
        identity_list = get_lfw_list(args.lfw_test_list)  # 所有人名
        lfw_img_paths = [os.path.join(args.lfw_root, each) for each in identity_list]  # 所有图片的路径
        lfw_images_list = read_all(lfw_img_paths, args.image_size)  # 所有图像(numpy数组)

        # 准备训练集(先要把原图转成tfrecord, 见dataset/conver_VGGFace2)
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

        next_element = iterator.get_next()  # 取数据

        # 学习率
        lr = tf.train.piecewise_constant(global_step, boundaries=args.lr_boundaries, values=args.lr_values, name='lr_schedule')

        # 优化器
        optimiser = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)
        # optimiser = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=args.momentum)

        # 超参数初始化方法
        w_init_method = tf.contrib.layers.xavier_initializer(uniform=True)

        # 得到模型输出的embedding
        model_out, end_points = mobilenet_v3_small(inputs=images_placeholder,
                                                   classes_num=args.embedding,
                                                   multiplier=1.0,
                                                   is_training=isTrain_placeholder,
                                                   reuse=None)
        # model_out_verify, end_points_verify = mobilenet_v3_small(inputs=images_placeholder,
        #                                                          classes_num=args.embedding,
        #                                                          multiplier=1.0,
        #                                                          is_training=False,
        #                                                          reuse=True)
        # 重新命名，以便查找
        model_out = tf.identity(model_out, 'embeddings')

        # ArcFace损失函数, 详情见ArcFace论文
        arcface_logit = arcface_loss(embedding=model_out,
                                     labels=labels_placeholder,
                                     w_init=w_init_method,
                                     out_num=args.num_classes)

        # 交叉熵
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arcface_logit,
                                                                       labels=labels_placeholder,
                                                                       name='cross_entropy_per_example')
        inference_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', inference_loss)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')  # 最终的损失

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        grads = optimiser.compute_gradients(total_loss)  # 计算反向传播的梯度

        # 关于“滑动指数平均”可以看《实战Google深度学习框架》
        variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimiser.apply_gradients(grads, global_step=global_step)  # 优化器用梯度对网络参数进行更新(训练)

        # GPU配置
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # summary writer
        # summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

        # add grad histogram op
        # for grad, var in grads:
        #     if grad is not None:
        #         summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # add trainabel variable gradients
        # for var in tf.trainable_variables():
        #     summaries.append(tf.summary.histogram(var.op.name, var))

        # add learning rate
        # summaries.append(tf.summary.scalar('leraning_rate', lr))
        # summary_op = tf.summary.merge(summaries)

        # 初始化变量
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        saver = tf.train.Saver(max_to_keep=args.saver_maxkeep)

        # 4 begin iteration
        count = 0

        for i in range(args.epoch):
            sess.run(iterator.initializer)  # 准备数据集
            while True:
                try:
                    images_train, labels_train = sess.run(next_element)  # 取一次数据
                    feed_dict = {images_placeholder: images_train,
                                 labels_placeholder: labels_train,
                                 isTrain_placeholder: True
                                 }

                    start = time.time()
                    _, _, _total_loss, _lr = sess.run([train_op, inc_op, total_loss, lr], feed_dict=feed_dict)
                    count += 1
                    end = time.time()

                    pre_sec = args.batch_size/(end - start)  # 训练速度：*个样本每秒

                    # 打印训练情况
                    if count > 0 and count % args.show_info_interval == 0:
                        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        log = '[%s] epoch:%d  total_step:%d  lr:%.5f  total loss:%.2f  time:%.3f samples/sec' % \
                              (time_str, i, count, _lr, _total_loss, pre_sec)
                        write_log_file(log_filename, log)  # 写到文件里
                        print(log)  # 输出到控制台

                    # save summary
                    # if count > 0 and count % args.summary_interval == 0:
                    #     feed_dict = {images_placeholder: images_train,
                    #                  labels_placeholder: labels_train,
                    #                  isTrain_placeholder: False}
                    #     summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    #     summary_writer.add_summary(summary_op_val, count)

                    # 保存模型(ckpt)
                    if count > 0 and count % args.ckpt_interval == 0:
                        filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(ckpt_path, filename)
                        saver.save(sess, filename)
                        log = 'save ckpt file: %s' % filename
                        write_log_file(log_filename, log)
                        print(log)

                        # 保存计算图
                        filename_graph = os.path.join(ckpt_path, 'model_graph_{:d}.pb'.format(count))
                        # 存成pbfile, 注意必须为List才能存
                        outPut_nodeName = []
                        outPut_nodeName.append('embeddings')  # 网络输出的节点
                        output_graph_def = tf.graph_util.convert_variables_to_constants(
                            sess,  # The session is used to retrieve the weights
                            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
                            output_node_names=outPut_nodeName
                            # The output node names are used to select the usefull nodes
                        )

                        with tf.gfile.GFile(filename_graph, "wb") as f:
                            f.write(output_graph_def.SerializeToString())

                    # 验证
                    if count > 0 and count % args.validate_interval == 0:
                        accuracy, threshold = test_on_lfw_when_training(sess, lfw_images_list, identity_list,
                                                                        args.lfw_test_list, args.batch_size,
                                                                        model_out, images_placeholder,
                                                                        isTrain_placeholder
                                                                        )

                        log = '\r\nlfw face verification accuracy: %f   threshold: %f' % (accuracy, threshold)
                        write_log_file(log_filename, log)
                        print(log)
                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break

