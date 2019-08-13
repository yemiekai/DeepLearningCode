from __future__ import print_function
import os
import cv2
from models.mobilenet_v3 import *
import numpy as np
import time
import math
import sys
import re



def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
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
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


# 正则化数据
def normalizing(feature):
    """
    # 测试用例:
    # input1 = np.random.randint(0, 255, (2, 50))
    # norm = normalizing(input1)
    # plt.scatter(norm[0, :], norm[1, :])
    # plt.show()
    """

    mean = np.mean(feature)
    standard = np.std(feature)  # 标准差

    # 结果是均值为0, 标准差为1
    return np.divide(feature-mean, standard)  # 除以standard就标准差为1, 除以variance就方差为1


def cosin_metric(x1, x2):
    """
    求两向量的cosin夹角
    :param x1:
    :param x2:
    :return:
    """
    cosin = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    return math.acos(max(min(cosin, 1), -1))


def cal_accuracy(y_score, y_true):
    # y_score feature之间的角度
    # y_true 真实标签 (1:同一人, 0:不同人)
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_threshold = 0
    for i in range(len(y_score)):
        threshold = y_score[i]  # 以某个夹角(y_score[i])作为阈值
        y_test = (y_score <= threshold)  # 小于阈值的为同一个人, 标签为1
        acc = np.mean((y_test == y_true).astype(int))  # 看看以y_score[i]作为阈值的准确率如何
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_acc, best_threshold


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]  # 根据人名获取对应feature
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    accuracy, threshold = cal_accuracy(sims, labels)
    return accuracy, threshold


def get_feature_dict(test_list, features):
    """
    用一个字典, 把每个人名和feature映射起来
    :param test_list:
    :param features:
    :return:
    """
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def load_image(img_path, input_shape=(224, 224, 3)):
    if input_shape[2] is 3:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    if input_shape[2] is 1:
        image = image[np.newaxis, :, :]  # 灰度图只有(w,h), 加一维表示通道数, 变成(1, w, h) -- channal = 1
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2读到的是BGR, 要转成RGB,
        # image = image.transpose((2, 0, 1))  # 由(w, h, c)转成(c, w, h)
    image = image[np.newaxis, :, :, :]  # 增加一维, 用于表示batchsize
    image = image.astype(np.float32, copy=False)

    # 正则化
    # image = normalizing(image/255.0)
    image -= 127.5
    image *= 0.0078125
    return image


# 一次把所有图片都读到数组里(需要占用很多内存)
def read_all(img_paths, input_shape):
    """
    用numpy.concatenate太慢了, 所以这里用list的append
    """
    print('read LFW, total: %d' % len(img_paths))
    images = []
    for i, img_path in enumerate(img_paths):
        sys.stdout.write('\r>> reading LFW: No.[%5d]: %s ' % (i, img_path.split('/')[-1]))
        sys.stdout.flush()
        image = load_image(img_path, input_shape)
        if image is None:
            print('read {} error'.format(img_path))

        images.append(image[0])

    print("read LFW done.")
    return images


def get_featurs(args, test_list, batch_size, input_shape, ckpt_path):
    images = None
    features = None
    cnt = 0
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:

            load_model(ckpt_path)
            for op in sess.graph.get_operations():  # 看看都有哪些变量, 找到变量才能跑
                print(op.name)
            _in = sess.graph.get_tensor_by_name("import/input:0")
            _out = g.get_tensor_by_name("import/embeddings:0")
            _train = g.get_tensor_by_name("import/placeholder_isTrain:0")

            for i, img_path in enumerate(test_list):
                image = load_image(img_path, input_shape)
                if image is None:
                    print('read {} error'.format(img_path))

                if images is None:
                    images = image
                else:
                    images = np.concatenate((images, image), axis=0)  # 拼接数据, 凑够一个批

                if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
                    cnt += 1

                    output = sess.run(_out, feed_dict={_in: images, _train: False})

                    if features is None:
                        features = output
                    else:
                        features = np.concatenate((features, output), axis=0)

                    images = None

    return features, cnt


def lfw_test(args, img_paths, identity_list, ckpt_path):

    s = time.time()
    features, cnt = get_featurs(args, img_paths, args.eval_batch_size, args.image_size, ckpt_path)
    print("features.shape:{}  LFW list length:{}  embedding size:{}".format(features.shape, len(img_paths), args.embedding))
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    accuracy, threshold = test_performance(fe_dict, args.lfw_test_list)
    print('lfw face verification accuracy: ', accuracy, 'threshold: ', threshold)
    return accuracy, threshold


def test_on_lfw_when_traing(sess, datas, identity_list, pair_list, batch_size, model_out_verify, images_placeholder,
                            isTrain_placeholder):
    data_nums = len(datas)
    index = 0
    features = []

    # 分批将data输入模型, 得到embeddings
    while index < data_nums:
        start = index
        end = min(index + batch_size, data_nums)
        sys.stdout.write('\r>>verify in LFW, getting embeddings: [%d]' % end)
        sys.stdout.flush()
        batch_datas = datas[start:end]

        # 输入网络, 得到embeddings
        model_out = sess.run(model_out_verify, feed_dict={images_placeholder: batch_datas,
                                                          isTrain_placeholder: False})

        # 用numpy.concatenate太慢了, 所以这里用list的append
        for feature in model_out:
            features.append(feature)

        index += batch_size

    fe_dict = get_feature_dict(identity_list, features)
    accuracy, threshold = test_performance(fe_dict, pair_list)
    print('\r\nlfw face verification accuracy: ', accuracy, 'threshold: ', threshold)


if __name__ == '__main__':
    class Argument:
        def __init__(self):
            self.eval_batch_size = 32
            self.image_size = (224, 224, 3)
            self.embedding = 512
            self.lfw_test_list = r'E:\DataSets\LFW\lfw_test_pair.txt'
            self.lfw_root = r'E:\DataSets\LFW\LFW_mtcnnpy_224'
            self.ckpt_path = r'C:\Users\Administrator\Desktop\ckpt\1234566.pb'

    args = Argument()

    # 验证集
    identity_list = get_lfw_list(args.lfw_test_list)
    lfw_img_paths = [os.path.join(args.lfw_root, each) for each in identity_list]  # 所有图片的路径

    # 从ckpt恢复模型并验证
    accuracy, threshold = lfw_test(args, lfw_img_paths, identity_list, args.ckpt_path)

