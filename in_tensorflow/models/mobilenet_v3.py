# -*- coding: utf-8 -*-

"""Implementation of Mobilenet V3.
Architecture: https://arxiv.org/pdf/1905.02244.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math


def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output


def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)


def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid


def _make_divisible(v, divisor=8, min_value=None):
    # if min_value is None:
    #     min_value = divisor
    # new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # # Make sure that round down does not go down by more than 10%.
    # if new_v < 0.9 * v:
    #     new_v += divisor
    # return new_v

    return v

def _batch_normalization_layer(inputs, momentum=0.997, epsilon=1e-3, is_training=True, name='bn', reuse=None):
    return tf.layers.batch_normalization(inputs=inputs,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         center=True,
                                         training=is_training,
                                         name=name,
                                         reuse=reuse)


def _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=False, strides=1, reuse=None, padding="SAME"):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
        padding=padding, #('SAME' if strides == 1 else 'VALID'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name,
        reuse=reuse)
    return conv


def _conv_1x1_bn(inputs, filters_num, name, use_bias=True, is_training=True, reuse=None):
    kernel_size = 1
    strides = 1
    x = _conv2d_layer(inputs, filters_num, kernel_size, name=name + "/conv", use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, is_training=is_training, name=name + '/bn',
                                   reuse=reuse)
    return x


def _conv_bn_relu(inputs, filters_num, kernel_size, name, use_bias=True, strides=1, is_training=True, activation=relu6, reuse=None):
    x = _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, is_training=is_training, name=name + '/bn',
                                   reuse=reuse)
    x = activation(x)
    return x


def _dwise_conv(inputs, k_h=3, k_w=3, depth_multiplier=1, strides=(1, 1),
                padding='SAME', name='dwise_conv', use_bias=False,
                reuse=None):
    kernel_size = (k_w, k_h)
    in_channel = inputs.get_shape().as_list()[-1]
    filters = int(in_channel*depth_multiplier)
    return tf.layers.separable_conv2d(inputs, filters, kernel_size,
                                      strides=strides, padding=padding,
                                      data_format='channels_last', dilation_rate=(1, 1),
                                      depth_multiplier=depth_multiplier, activation=None,
                                      use_bias=use_bias, name=name, reuse=reuse
                                      )


def _fully_connected_layer(inputs, units, name="fc", activation=None, use_bias=True, reuse=None):
    return tf.layers.dense(inputs, units, activation=activation, use_bias=use_bias,
                           name=name, reuse=reuse)


def _global_avg(inputs, pool_size, strides, padding='valid', name='global_avg'):
    return tf.layers.average_pooling2d(inputs, pool_size, strides,
                                       padding=padding, data_format='channels_last', name=name)


def _squeeze_excitation_layer(input, out_dim, ratio, layer_name, is_training=True, reuse=None):
    with tf.variable_scope(layer_name, reuse=reuse):
        squeeze = _global_avg(input, pool_size=input.get_shape()[1:-1], strides=1)

        excitation = _fully_connected_layer(squeeze, units=out_dim, name=layer_name + '_excitation1',
                                            reuse=reuse)
        excitation = relu6(excitation)
        excitation = _fully_connected_layer(excitation, units=out_dim, name=layer_name + '_excitation2', reuse=reuse)
        excitation = hard_sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, int(out_dim)])
        scale = input * excitation
        return scale

def shuffle_unit(x, groups):
    with tf.variable_scope('shuffle_unit'):
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
    return x


def mobilenet_v3_block(input, k_s, expansion_ratio, output_dim, in_channels, stride, name, is_training=True,
                       use_bias=True, shortcut=True, activatation="RE", ratio=16, se=False, reuse=None):
    """
    要SE: 162ms; 不要SE:143ms
    """
    bottleneck_dim = expansion_ratio  

    with tf.variable_scope(name, reuse=reuse):

        # 没有下采样时，采用ShuffleNetV2的block
        if stride == 1:
            top, bottom = tf.split(input, num_or_size_splits=2, axis=3)  # 把通道切2份

            half_channel = output_dim // 2

            # 点卷积, 扩展到高维
            top = _conv_1x1_bn(top, bottleneck_dim, name="pw_top", use_bias=use_bias)

            # dw  深度卷积
            top = _dwise_conv(top, k_w=k_s, k_h=k_s, strides=[stride, stride], name='dw_top', use_bias=use_bias, reuse=reuse)
            top = _batch_normalization_layer(top, momentum=0.997, epsilon=1e-3, is_training=is_training, name='dw_bn_top', reuse=reuse)
            top = hard_swish(top)

            # 点卷积, 回到低维
            top = _conv_1x1_bn(top, half_channel, name="pw_linear_top", use_bias=use_bias)
            top = hard_swish(top)  # 参考mobileNetV2，非线性层放在最后用

            # SE模块, (在通道混洗之前用, 以免影响残差结构过来的之前的通道)
            top = _squeeze_excitation_layer(top, out_dim=half_channel, ratio=ratio, layer_name='se_block')

            # 连接
            net = tf.concat([top, bottom], axis=3)

            # 通道混洗
            net = shuffle_unit(net, 2)

        else:
            # 点卷积, 扩展到高维
            net = _conv_1x1_bn(input, bottleneck_dim, name="pw", use_bias=use_bias)

            # dw  深度卷积
            net = _dwise_conv(net, k_w=k_s, k_h=k_s, strides=[stride, stride], name='dw_bottom', use_bias=use_bias, reuse=reuse)
            net = _batch_normalization_layer(net, momentum=0.997, epsilon=1e-3, is_training=is_training, name='dw_bn_bottom', reuse=reuse)
            net = hard_swish(net)

            # 点卷积, 回到低维
            net = _conv_1x1_bn(net, in_channels, name="pw_linear", use_bias=use_bias)
            net = hard_swish(net)  # 参考mobileNetV2，非线性层放在最后用

            # SE模块
            net = _squeeze_excitation_layer(net, out_dim=in_channels, ratio=ratio, layer_name='se_block')

            # 残差连接, 降维后拼接过来, 使维度翻倍
            # top = _global_avg(input, (3, 3), 2, padding='same')
            top = _dwise_conv(input, k_w=3, k_h=3, strides=[2, 2], name='dw_top', use_bias=use_bias, reuse=reuse)
            top = _batch_normalization_layer(top, momentum=0.997, epsilon=1e-3, is_training=is_training, name='dw_bn_top', reuse=reuse)
            top = hard_swish(top)

            net = tf.concat([top, net], axis=3)
            net = shuffle_unit(net, 2)
            net = tf.identity(net, name='block_output')

    return net


def mobilenet_v3_small(inputs, classes_num, multiplier=1.0, is_training=True, reuse=None):
    end_points = {}
    layers = [
        # [16, 16, 3, 2, "RE", True, 16],
        [12, 24, 3, 2, "RE", True, 72],
        [24, 24, 3, 1, "RE", True, 90],
        [24, 48, 3, 2, "RE", True, 102],
        [48, 48, 3, 1, "HS", False, 120],
        [48, 48, 3, 1, "HS", False, 160],
        [48, 48, 3, 1, "HS", False, 200],
        [48, 48, 3, 1, "HS", False, 240],
        [48, 96, 3, 2, "HS", False, 288],
        [96, 96, 3, 1, "HS", False, 384],
        [96, 96, 3, 1, "HS", False, 576],
    ]

    # 给输入张量命名(加载ckpt时要用到)
    inputs = tf.identity(inputs, name='inputs')

    input_size = inputs.get_shape().as_list()[1:-1]
    # assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

    reduction_ratio = 4
    with tf.variable_scope('init', reuse=reuse):
        init_conv_out = _make_divisible(12 * multiplier)
        x = _conv_bn_relu(inputs, filters_num=init_conv_out, kernel_size=5, name='init',
                          use_bias=False, strides=2, is_training=is_training, activation=hard_swish)

    with tf.variable_scope("MobilenetV3_small", reuse=reuse):
        for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(layers):
            in_channels = _make_divisible(in_channels * multiplier)
            out_channels = _make_divisible(out_channels * multiplier)
            exp_size = _make_divisible(exp_size * multiplier)
            x = mobilenet_v3_block(x, kernel_size, exp_size, out_channels, in_channels, stride,
                                   "bneck{}".format(idx), is_training=is_training, use_bias=True,
                                   shortcut=(in_channels==out_channels), activatation=activatation,
                                   ratio=reduction_ratio, se=se)
            end_points["bneck{}".format(idx)] = x

        conv1_in = _make_divisible(96 * multiplier)
        conv1_out = _make_divisible(576 * multiplier)
        x = _conv_bn_relu(x, filters_num=conv1_out, kernel_size=1, name="conv1_out",
                          use_bias=True, strides=1, is_training=is_training, activation=hard_swish)

        # x = _squeeze_excitation_layer(x, out_dim=conv1_out, ratio=reduction_ratio, layer_name="conv1_out",
        #                              is_training=is_training, reuse=None)
        end_points["conv1_out_1x1"] = x

        x = _dwise_conv(x, k_w=7, k_h=7, strides=[1, 1], padding='VALID', name='GDWConv', use_bias=True, reuse=reuse)
        # x = _global_avg(x, pool_size=x.get_shape()[1:-1], strides=1)
        #x = hard_swish(x)
        end_points["global_depth_wise_conv"] = x

    with tf.variable_scope('Logits_out', reuse=reuse):
        conv2_in = _make_divisible(576 * multiplier)
        conv2_out = _make_divisible(1280 * multiplier)
        x = _conv2d_layer(x, filters_num=conv2_out, kernel_size=1, name="conv2", use_bias=True, strides=1)
        x = hard_swish(x)
        end_points["conv2_out_1x1"] = x

        x = _conv2d_layer(x, filters_num=classes_num, kernel_size=1, name="conv3", use_bias=True, strides=1)
        logits = tf.layers.flatten(x)
        logits = tf.layers.dropout(logits, rate=0.8, training=is_training)
        logits = tf.identity(logits, name='output')
        end_points["Logits_out"] = logits

    return logits, end_points


# def mobilenet_v3_large(inputs, classes_num, multiplier=1.0, is_training=True, reuse=None):
#     end_points = {}
#     layers = [
#         [16, 16, 3, 1, "RE", False, 16],
#         [16, 24, 3, 2, "RE", False, 64],
#         [24, 24, 3, 1, "RE", False, 72],
#         [24, 40, 5, 2, "RE", True, 72],
#         [40, 40, 5, 1, "RE", True, 120],
#
#         [40, 40, 5, 1, "RE", True, 120],
#         [40, 80, 3, 2, "HS", False, 240],
#         [80, 80, 3, 1, "HS", False, 200],
#         [80, 80, 3, 1, "HS", False, 184],
#         [80, 80, 3, 1, "HS", False, 184],
#
#         [80, 112, 3, 1, "HS", True, 480],
#         [112, 112, 3, 1, "HS", True, 672],
#         [112, 160, 5, 1, "HS", True, 672],
#         [160, 160, 5, 2, "HS", True, 672],
#         [160, 160, 5, 1, "HS", True, 960],
#     ]
#
#     input_size = inputs.get_shape().as_list()[1:-1]
#     assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))
#
#     reduction_ratio = 4
#     with tf.variable_scope('init', reuse=reuse):
#         init_conv_out = _make_divisible(16 * multiplier)
#         x = _conv_bn_relu(inputs, filters_num=init_conv_out, kernel_size=3, name='init',
#                           use_bias=False, strides=2, is_training=is_training, activation=hard_swish)
#
#     with tf.variable_scope("MobilenetV3_large", reuse=reuse):
#         for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(layers):
#             in_channels = _make_divisible(in_channels * multiplier)
#             out_channels = _make_divisible(out_channels * multiplier)
#             exp_size = _make_divisible(exp_size * multiplier)
#             x = mobilenet_v3_block(x, kernel_size, exp_size, out_channels, stride,
#                                    "bneck{}".format(idx), is_training=is_training, use_bias=True,
#                                    shortcut=(in_channels==out_channels), activatation=activatation,
#                                    ratio=reduction_ratio, se=se)
#             end_points["bneck{}".format(idx)] = x
#
#         conv1_in = _make_divisible(160 * multiplier)
#         conv1_out = _make_divisible(960 * multiplier)
#         x = _conv_bn_relu(x, filters_num=conv1_out, kernel_size=1, name="conv1_out",
#                           use_bias=True, strides=1, is_training=is_training, activation=hard_swish)
#         end_points["conv1_out_1x1"] = x
#
#         x = _global_avg(x, pool_size=x.get_shape()[1:-1], strides=1)
#         #x = hard_swish(x)
#         end_points["global_pool"] = x
#
#     with tf.variable_scope('Logits_out', reuse=reuse):
#         conv2_in = _make_divisible(960 * multiplier)
#         conv2_out = _make_divisible(1280 * multiplier)
#         x = _conv2d_layer(x, filters_num=conv2_out, kernel_size=1, name="conv2", use_bias=True, strides=1)
#         x = hard_swish(x)
#         end_points["conv2_out_1x1"] = x
#
#         x = _conv2d_layer(x, filters_num=classes_num, kernel_size=1, name="conv3", use_bias=True, strides=1)
#         logits = tf.layers.flatten(x)
#         logits = tf.identity(logits, name='output')
#         end_points["Logits_out"] = logits
#
#     return logits, end_points


if __name__ == "__main__":
    print("begin ...")
    input_test = tf.zeros([6, 224, 224, 3])
    num_classes = 512
    # model, end_points = mobilenet_v3_large(input_test, num_classes, multiplier=1.0, is_training=True, reuse=None)
    model, end_points = mobilenet_v3_small(input_test, num_classes, multiplier=1.0, is_training=True, reuse=None)
    print("[")
    for key, item in end_points.items():
        print("{}:{}".format(key, item))
    print("]")
    print(model)
    print("done !")
