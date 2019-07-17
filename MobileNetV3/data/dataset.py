import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
import gc


def normalizing(feature):
    """
    正则化数据

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


def caculate_mean_std(imgs):
    """
    计算数据集(图片)的总体均值和方差

    warning: 这里需要很大内存

    :param imgs: 所有图片的路径
    :return:
    """
    r = []
    g = []
    b = []

    # 抽部分图片来计算
    leng = min(1000, len(imgs))
    imgs_sample = np.random.permutation(imgs)[0:leng]

    for img in imgs_sample:
        img_data = np.array(Image.open(img))  # Image打开返回的是RGB (H , W , C)

        img_data = img_data/255.0  # 归一化
        r.append(img_data[:, :, 0].flatten())
        g.append(img_data[:, :, 1].flatten())
        b.append(img_data[:, :, 2].flatten())

    r = np.array(r).flatten()
    g = np.array(g).flatten()
    b = np.array(b).flatten()

    # 各通道均值
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    # 各通道方差
    std_r = np.std(r - mean_r)
    std_g = np.std(g - mean_g)
    std_b = np.std(b - mean_b)

    # 回收内存
    del r, g, b
    gc.collect()

    print("mean: R({}), G({}), B({})".format(mean_r, mean_g, mean_b))
    print("standard: R({}), G({}), B({})".format(std_r, std_g, std_b))
    return mean_r, mean_g, mean_b, std_r, std_g, std_b


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


def get_image_paths(facedir):
    """
    获取facedir目录下所有文件, 返回其路径
    :param facedir:
    :return:
    """
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


class Dataset(data.Dataset):

    def __init__(self, root, path_spilt, phase='train', input_shape=(1, 128, 128)):
        print("--- init Dataset ---")
        self.phase = phase
        self.input_shape = input_shape
        self.imgs = []
        self.path_spilt = path_spilt

        # 有多少个目录就有多少个class
        self.classes = [path for path in os.listdir(root) if os.path.isdir(root)]
        print("There are [{}] classes(folders) in [{}]".format(len(self.classes), root))
        print("Network input shape is {}".format(input_shape))

        # 获取所有图片路径
        for i in range(len(self.classes)):
            class_name = self.classes[i]
            facedir = os.path.join(root, class_name)
            image_paths = get_image_paths(facedir)  # 获取该目录下所有照片的路径
            self.imgs += image_paths
            sys.stdout.write('\r>> reading: No.[%d]: %s ' % (i, class_name))
            sys.stdout.flush()
        print("read dataset done.")

        self.imgs = np.random.permutation(self.imgs)  # 打乱

        if self.input_shape[0] is 1:
            normalize = T.Normalize(mean=[0.5], std=[0.5])
        else:
            print("calculating the mean and standard of dataset...")
            mean_r, mean_g, mean_b, std_r, std_g, std_b = caculate_mean_std(self.imgs)
            normalize = T.Normalize(mean=[mean_r, mean_g, mean_b], std=[std_r, std_g, std_b])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize(input_shape[2]),
                T.RandomHorizontalFlip(),  # 给定几率水平翻转图像
                T.ToTensor(),  # ToTensor()会把数据处理到[0,1]区间
                normalize  # (x-0.5)/0.5就是[-1.0, 1.0])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(input_shape[2]),
                T.ToTensor(),
                normalize
            ])
        print('Dataset inited .')

    def __getitem__(self, index):
        sample_path = self.imgs[index]  # 当前图片的路径
        sample_dir = os.path.dirname(sample_path)  # 当前图片的上级目录的路径

        # 标签必须是从0~classNums的连续整数
        class_name = sample_dir.split(self.path_spilt)[-1]  # 获取类名(当前图片的文件夹名字)
        label = self.classes.index(class_name)  # 根据类名从列表里找到它的索引, 把索引号当作标签

        data = Image.open(sample_path)  # Image打开返回的是RGB (H , W , C)

        if self.input_shape[0] is 1:  # 输入是1维, 则转成灰度图
            data = data.convert('L')  # 转为灰度图像(1维), 公式L = R*0.299 + G*0.587+ B*0.114

        data = self.transforms(data)

        return data.float(), label

    def __len__(self):
        return len(self.imgs)


# 测试一下
if __name__ == '__main__':
    dataset = Dataset(root=r'/media/yemiekai/SSD860QVO1TB/DataSets/VGGFace2/temp',
                      phase='test',
                      input_shape=(3, 224, 224))

    trainloader = data.DataLoader(dataset, batch_size=10)
    print('{} train iters per epoch:'.format(len(trainloader)))
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = data[0].numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        # img += np.array([1, 1, 1])
        img *= 127.5
        img += 127.5

        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        print(label)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
