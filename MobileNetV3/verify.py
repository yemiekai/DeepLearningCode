from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
import math
from torch.nn import DataParallel
import matplotlib.pyplot as plt


# 正则化数据
def normalizing(feature):
    """
    # 测试用例:
    # input1 = np.random.randint(0, 255, (2, 50))
    # norm = normalizing(input1)
    # plt.scatter(norm[0, :], norm[1, :])
    # plt.show()
    """
    # 1. 均值0化(减去均值)
    feature = feature - np.mean(feature)

    # 2. 得到标准差, 方差
    standard = np.std(feature)  # 标准差
    variance = np.square(standard)  # 方差

    # 结果是均值为0, 标准差为1
    return np.divide(feature, variance)  # 除以standard就标准差为1, 除以variance就方差为1


def cosin_metric(x1, x2):
    """
    求两向量的cosin夹角
    :param x1:
    :param x2:
    :return:
    """
    cosin = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    return math.acos(min(max(cosin, 1), -1))


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


def load_image(img_path, input_shape=(1, 224, 224)):
    if input_shape[0] is 3:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    if input_shape[0] is 1:
        image = image[np.newaxis, :, :]  # 灰度图只有(w,h), 加一维表示通道数, 变成(1, w, h) -- channal = 1
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2读到的是BGR, 要转成RGB,
        image = image.transpose((2, 0, 1))  # 由(w, h, c)转成(c, w, h)
    image = image[np.newaxis, :, :, :]  # 增加一维, 用于表示batchsize
    image = image.astype(np.float32, copy=False)

    # 正则化
    image = normalizing(image/255.0)
    # image -= 127.5
    # image /= 127.5
    return image


def get_featurs(model, test_list, batch_size=32, input_shape=(1, 224, 224)):
    images = None
    features = None
    cnt = 0
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

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)  # 一次出来一个批, output shape = (batchsize, embedding)
            output = output.data.cpu().numpy()
            # if input_shape[0] is 1:
            #     fe_1 = output[::2]
            #     fe_2 = output[1::2]
            # else:
            #     fe_1 = output[::4]
            #     fe_2 = output[3::4]
            #
            # feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = output
            else:
                features = np.concatenate((features, output), axis=0)

            images = None

    return features, cnt


def lfw_test(model, img_paths, identity_list, opt):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, opt.test_batch_size, opt.input_shape)
    print("features.shape:{}  LFW list length:{}  embedding size:{}".format(features.shape, len(img_paths), opt.embedding))
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    accuracy, threshold = test_performance(fe_dict, opt.lfw_test_list)
    print('lfw face verification accuracy: ', accuracy, 'threshold: ', threshold)
    return accuracy, threshold



