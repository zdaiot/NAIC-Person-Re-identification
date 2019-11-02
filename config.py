import json
import argparse
from argparse import Namespace


def get_config():
    use_paras = False
    if use_paras:
        with open('./checkpoints/resnet50/' + "params.json", 'r', encoding='utf-8') as json_file:
            config = json.load(json_file)
        # dict to namespace
        config = Namespace(**config)
    else:
        parser = argparse.ArgumentParser()
        '''
        unet_resnet34时各个电脑可以设置的最大batch size
        zdaiot:12 z840:16 mxq:48
        unet_se_renext50
        hwp: 8
        unet_resnet50:
        MXQ: 24
        '''
        # model hyper-parameters
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--epoch', type=int, default=30, help='epoch')
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--selected_fold', type=list, default=[1], help='what folds for training?')

        # dataset set
        parser.add_argument('--augmentation_flag', type=bool, default=True, help='if true, use augmentation method in train set')
        parser.add_argument('--n_splits', type=int, default=3, help='n_splits_fold')
        parser.add_argument('--shuffle_train', type=bool, default=True, help='shuffle train dataset')
        # TODO not use
        parser.add_argument('--crop', type=bool, default=False, help='if true, crop image to [height, width].')
        parser.add_argument('--height', type=int, default=None, help='the height of cropped image')
        parser.add_argument('--width', type=int, default=None, help='the width of cropped image')

        # model set 
        parser.add_argument('--model_name', type=str, default='resnet50',
                            help='resnet50/se_resnext50_32x4d/efficientnet_b4/resnet50/efficientnet_b4')
        parser.add_argument('--last_stride', type=int, default=1, help='last stride in the model')

        # loss set
        parser.add_argument('--selected_loss', type=str, default='softmax_triplet',
                            help='Select the loss function, softmax_triplet/softmax/triplet')
        parser.add_argument('--margin', type=float, default=0.3, help='margin coefficient in triplet loss')
        parser.add_argument('--label_smooth', type=bool, default=False, help='use label smooth in cross entropy')

        # 优化器设置
        parser.add_argument('--optimizer_name', type=str, default='Adam', help='which optimizer to use')
        parser.add_argument('--momentum_SGD', type=float, default=0.9, help='momentum in SGD')
        parser.add_argument('--base_lr', type=float, default=5e-5, help='init lr')
        parser.add_argument('--bias_lr_factor', type=float, default=1, help='')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay in optimizer')
        parser.add_argument('--weight_decay_bias', type=float, default=0.0, help='')

        # 设置WarmupMultiStepLR
        parser.add_argument('--steps', type=list, default=[20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195,
                                                           210, 225, 240, 255], help='')
        parser.add_argument('--gamma', type=float, default=0.6, help='')
        parser.add_argument('--warmup_factor', type=float, default=0.01, help='')
        parser.add_argument('--warmup_iters', type=int, default=10, help='')
        parser.add_argument('--warmup_method', type=str, default='linear', help='constant/linear')

        # path set
        parser.add_argument('--save_path', type=str, default='./checkpoints')
        parser.add_argument('--dataset_root', type=str, default='./dataset/NAIC_data/初赛训练集')

        # 其他设置
        parser.add_argument('--cython', type=bool, default=True, help='use cython or python to eval')
        parser.add_argument('--rerank', type=bool, default=True, help='use rerank or not')

        config = parser.parse_args()
        # config = {k: v for k, v in args._get_kwargs()}

    return config


if __name__ == '__main__':
    config = get_config()
