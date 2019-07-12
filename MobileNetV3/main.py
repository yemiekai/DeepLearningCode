from models import mobileNetV3
from models import focal_loss
from models import metrics
from data import dataset
from config import config
from torch.utils import data
from torch.nn import DataParallel
from test import *
import numpy as np
import torch
import os
import time
import math



# todo 暂时用这个学习率, 先把程序跑起来（这个学习率是condenseNet里的, 要看mobileNetV3论文里怎么设的）
def adjust_learning_rate(optimizer, epoch_now, args, batch=None, epoch_iters=None):

    T_total = args.max_epoch * epoch_iters
    T_cur = (epoch_now % args.max_epoch) * epoch_iters + batch
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_model(model, optimizer, save_path, name, pretrain_info_name, epoch, iter, lr, iters):

    save_name = os.path.join(save_path, name + '_' + str(epoch) + '_' + str(iter) + '.pth')

    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, save_name)

    txt_info = "{} {} {} {}\n".format(epoch, iter, lr, iters)
    lr_filename = os.path.join(save_path, pretrain_info_name)
    with open(lr_filename, 'a') as fout:
        fout.write(txt_info)

    return save_name


if __name__ == '__main__':

    print("You have ", torch.cuda.device_count(), " GPUs")
    device = torch.device("cuda")

    # 参数
    opt = config.MobileNetV3Config()

    # 设置路径--保存训练产生的数据
    date = time.strftime("%Y-%m-%d", time.localtime())
    save_path = os.path.join(opt.checkpoints_path, date)  # 保存的文件夹路径
    os.makedirs(save_path, exist_ok=True)
    log_filename = os.path.join(save_path, 'Console_Log.txt')  # 日志路径

    # 读取数据集
    train_dataset = dataset.Dataset(opt.train_root, opt.path_split, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    opt.num_classes = len(train_dataset.classes)  # 分类数量
    epoch_iters = len(trainloader)  # 每个epoch里iter总个数

    # 验证集
    identity_list = dataset.get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    criterion = focal_loss.FocalLoss(gamma=2)
    metric_fc = metrics.ArcMarginProduct(opt.embedding, opt.num_classes, s=64, m=0.5, easy_margin=opt.easy_margin)

    # 加载模型
    model = mobileNetV3.MobileNetV3(n_class=opt.embedding, input_size=opt.input_shape[2], dropout=opt.dropout_rate)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)

    print("epoch:{}".format(opt.max_epoch))
    print("iters/epoch:{}".format(epoch_iters))
    print("classes:{}".format(len(train_dataset.classes)))
    print("batch_size:{}".format(opt.train_batch_size))
    start = time.time()
    acc = 0
    for epoch in range(opt.max_epoch):

        model.train()

        for iter, data in enumerate(trainloader):
            # 训练总进程
            progress = float(epoch * epoch_iters + iter) / (opt.max_epoch * epoch_iters)

            # Adjust learning rate
            lr = adjust_learning_rate(optimizer, epoch, opt, batch=iter, epoch_iters=epoch_iters)

            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()

            feature = model(data_input)  # output (batchSize, 512) embedding
            output = metric_fc(feature, label)  # output  (batchSize, classNums)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = epoch * epoch_iters + iter

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                # acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                log_info = '{}  train epoch:{}  iter:{}  {:.5} iters/s  loss:{:.5f}  lr:{}   ' \
                           'progress:{:.2f}%  accuracy:{:.2f}%'.format(time_str, epoch, iter, speed, loss.item(),
                                                                       lr, progress * 100, acc * 100)
                print(log_info)
                with open(log_filename, 'a') as fout:
                    fout.write(log_info + '\n')

                start = time.time()

            # 测一下准确度
            if iter > 0 and iter % 5000 == 0:
                model.eval()
                acc = lfw_test(model, img_paths, identity_list, opt)
                model.train()

            # 保存模型
            if iter % opt.save_freq == 0:
                # 当前epoch, 当前epoch中iter数, 当前lr, 当前总iters(就是epoch*epoch_iters + iter)
                txt_info = "{} {} {} {}\n".format(epoch, iter, lr, iters)
                save_model(model, optimizer, save_path, opt.model_name, opt.pretrain_info_name, epoch, iter, lr, iters)

        # 完成一个epoch, 测一下准确度
        model.eval()
        acc = lfw_test(model, img_paths, identity_list, opt)

        # if opt.display:
        #     visualizer.display_current_results(iters, acc, name='test_acc')

    save_model(model, optimizer, save_path, opt.model_name, opt.pretrain_info_name, opt.max_epoch, epoch_iters, 0,
               opt.max_epoch * epoch_iters)

