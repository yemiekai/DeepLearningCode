
import torch
import os
from data import dataset
from config import config
from models import mobileNetV3_MixNet
from torch.nn import DataParallel
from verify import *

MODEL = "/home/yemiekai/extendDisk/SSD860QVO/TrainingCache/mobileNetV3_vggface2/2019-07-11/mobileNetV3_20_24523.pth"


def test_inference_once():
    device = torch.device("cuda")

    # 参数
    opt = config.MobileNetV3Config()

    # 加载模型
    model = mobileNetV3_MixNet.MobileNetV3_MixNet(n_class=opt.embedding, input_size=opt.input_shape[2], dropout=opt.dropout_rate)
    model.to(device)
    model = DataParallel(model)

    x_image = torch.randn(1, 3, 224, 224).to(device)
    y = model(x_image)
    print(y)


def test_pretrain_model():
    device = torch.device("cuda")

    # 参数
    opt = config.MobileNetV3Config()

    # 验证集
    identity_list = dataset.get_lfw_list(opt.lfw_test_list)
    lfw_img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]  # 所有图片的路径

    # 加载模型
    model = mobileNetV3_MixNet.MobileNetV3_MixNet(n_class=opt.embedding, input_size=opt.input_shape[2], dropout=opt.dropout_rate)
    model.to(device)
    model = DataParallel(model)

    # 加载预训练的模型
    state = torch.load(MODEL)
    model.load_state_dict(state['state_dict'])

    # 用LFW数据集测试
    accuracy, threshold = lfw_test(model, lfw_img_paths, identity_list, opt)


if __name__ == '__main__':
    test_inference_once()
