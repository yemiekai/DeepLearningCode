from tensorflow.python.tools.freeze_graph import freeze_graph
import os
import tensorflow as tf

import argparse
"""
制作tflite

1. 先把计算图和参数结合, 固化到.pb文件. 使用下面的函数freeze_graph_and_parameter()
  (1)计算图在训练的时候已经通过tf.io.write_graph保存成.pb文件了
  (2)参数在.ckpt文件里, 训练的时候通过saver.save得到
  
2. 把固化的.pb转成.tflite. 使用下面的函数convert_tflite()
   需要知道输入输出结点, 可以查看网络代码, 或者加载计算图打印所有节点
"""

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--filename_graph', default=r'E:\TrainingCache\mobileNetV3_arcFace_VGGFace_tensorflow\2019-08-25\model_graph.pb')
    parser.add_argument('--filename_checkpoint', default=r'E:\TrainingCache\mobileNetV3_arcFace_VGGFace_tensorflow\2019-08-25\ckpt\InsightFace_iter_100000.ckpt')
    parser.add_argument('--filename_frozenModel', default=r'E:\TrainingCache\mobileNetV3_arcFace_VGGFace_tensorflow\2019-08-25\MobileNetV3_InsightFace_frozen.pb')
    parser.add_argument('--filename_tflite', default=r'E:\TrainingCache\mobileNetV3_arcFace_VGGFace_tensorflow\2019-08-25\MobileNetV3_InsightFace.tflite')
    args = parser.parse_args()
    return args


def freeze_graph_and_parameter(args):
    """
    将[计算图]和[参数]固化到.pb文件里
    """
    # 计算图的路径, 训练模型的时候通过tf.io.write_graph得到
    filename_graph = args.filename_graph

    # ckpt的路径
    # saver.save会保存3个文件, 分别是
    # ***.ckpt.data-00000-of-00001;
    # ***.ckpt.index;
    # ***.ckpt.meta;
    # 这里用***.ckpt即可
    filename_checkpoint = args.filename_checkpoint

    # 输出结果的路径
    filename_output = args.filename_frozenModel

    # 把图和参数结合一起, 这里output_node_names是输出节点的名字, 可以通过sess.graph.get_operations()找到你的网络所有节点名字
    freeze_graph(input_graph=filename_graph,
                 input_saver=None,
                 input_binary=False,
                 input_checkpoint=filename_checkpoint,
                 output_node_names="Logits_out/output",
                 restore_op_name=None,
                 filename_tensor_name=None,
                 output_graph=filename_output,
                 clear_devices=True,
                 initializer_nodes='')


def convert_tflite(args):
    """
    将固化的模型转成tflite
    :return:
    """
    filename_model = args.filename_frozenModel
    filename_tflite = args.filename_tflite
    input_arrays = ["inputs"]
    output_arrays = ["Logits_out/output"]

    model_exp = os.path.expanduser(filename_model)

    # 加载模型, 查看节点名字, 知道输入输出节点才能转换tflite
    # with tf.Session() as sess:
    #     with tf.gfile.FastGFile(model_exp, 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #         tf.import_graph_def(graph_def)
    #     for op in sess.graph.get_operations():
    #         print(op.name)

    converter = tf.lite.TFLiteConverter.from_frozen_graph(filename_model, input_arrays, output_arrays)
    # converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open(filename_tflite, "wb").write(tflite_model)


if __name__ == '__main__':

    args = get_parser()

    freeze_graph_and_parameter(args)  # 先得到固化的模型.pb
    convert_tflite(args)  # 再从.pb得到.tflite

