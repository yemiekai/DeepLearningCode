"""Imports a model metagraph and checkpoint file, converts the variables to constants
and exports the model as a graphdef protobuf
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
from six.moves import xrange
import tensorflow as tf
import argparse
import sys
import os
import re

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = get_model_filenames(os.path.expanduser(args.model_dir))
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            model_dir_exp = os.path.expanduser(args.model_dir)
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file), clear_devices=True)
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

            graph = tf.get_default_graph()
            input_graph_def = graph.as_graph_def()

            output_list = []
            for op in graph.get_operations():  # 看看都有哪些变量, 找到变量才能跑
                if 'Logits_out/output' in op.name:  # 这个名字要根据网络的定义
                    output_list.append(op.name)

            # 暂时没有设置黑白名单
            output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_list)

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(args.output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), args.output_file))

  
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters',
                        default=r"E:\TrainingCache\mobileNetV3_arcFace_VGGFace_tensorflow\2019-08-13\output\ckpt")

    parser.add_argument('--output_file', type=str,
                        help='Filename for the exported graphdef protobuf (.pb)',
                        default=r"1234566.pb")
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
