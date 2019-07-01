class MobileNetV3Config(object):

    have_pretrain = False
    pretrain_info_name = "pretrain_info.txt"
    pretrain_path = r'/home/yemiekai/extendDisk/SSD860QVO/TrainingCache/condense_arcface_vggface2/2019-05-27'
    checkpoints_path = r'/home/yemiekai/extendDisk/SSD860QVO/TrainingCache/condense_arcface_vggface2'  #

    train_root = r'/home/yemiekai/extendDisk/SSD860QVO/DataSets/VGGFace2/VGGFace2_train_mtcnnpy_224'
    train_list = r'/home/yemiekai/extendDisk/SSD860QVO/DataSets/VGGFace2/train_list.txt'
    val_list = r'/home/yemiekai/extendDisk/SSD860QVO/DataSets/VGGFace2/test_list.txt'

    test_root = r'/home/yemiekai/extendDisk/SSD860QVO/DataSets/VGGFace2/VGGFace2_test_mtcnnpy_224'
    test_list = r'/home/yemiekai/extendDisk/SSD860QVO/DataSets/VGGFace2/test.txt'

    lfw_root = r'/home/yemiekai/extendDisk/SSD860QVO/DataSets/LFW/LFW_mtcnnpy_224'
    lfw_test_list = r'/home/yemiekai/extendDisk/SSD860QVO/DataSets/LFW/lfw_test_pair.txt'

    # backbone = 'condense'
    #
    # load_model_path = 'models/resnet18.pth'
    # test_model_path = 'checkpoints/resnet18_110.pth'
    # save_interval = 1
    #
    # train_batch_size = 64  # batch size
    # test_batch_size = 64
    #
    # input_shape = (1, 224, 224)
    #
    # optimizer = 'sgd'
    #
    # use_gpu = True  # use GPU or not
    # gpu_id = '0, 1'
    # num_workers = 16  # how many workers for loading data
    # print_freq = 100  # print info every N batch
    # save_freq = 5000  # print info every N batch
    #
    # debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    # result_file = 'result.csv'
    #
    # max_epoch = 10
    # # lr = 1e-1  # initial learning rate
    # lr = 0.001  # initial learning rate
    # lr_step = 10
    # lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    # lr_type = 'cosine'
    # weight_decay = 0.0001
    # dropout_rate = 0.1
    # momentum = 0.9

    # group_1x1 = 4
    # group_3x3 = 4
    # condense_factor = 4
    # bottleneck = 4
    # stages = '4-6-8-10-8'  # 5个DenseBlock, 每个block分别有4,6,8,10,8个DenseLayer
    # growth = '8-16-32-64-128'
    # data = 'vggface2'
    #
    # env = 'default'
    # classify = 'softmax'
    #
    # metric = 'arc_margin'
    # easy_margin = False
    # use_se = False
    # loss = 'focal_loss'
    #
    # display = False
    # finetune = False
    #
    # embedding = 512
    # num_classes = 8631
    # num_classes = 20
