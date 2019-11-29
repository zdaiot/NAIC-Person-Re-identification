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

        # model hyper-parameters
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--num_instances', type=int, default=4,
                            help='num_instances for each class, only use in train_dataloader')
        parser.add_argument('--epoch', type=int, default=180, help='epoch')
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--selected_fold', type=list, default=[2], help='what folds for training?')

        # dataset set
        parser.add_argument('--augmentation_flag', type=bool, default=True,
                            help='if true, use DataAugmentation class in train set')
        parser.add_argument('--erase_prob', type=float, default=0.0,
                            help='when augmentation_flag=True, erase probability in DataAugmentation class')
        parser.add_argument('--gray_prob', type=float, default=0.3,
                            help='when augmentation_flag=True, gray probability in DataAugmentation class')
        parser.add_argument('--n_splits', type=int, default=5, help='n_splits_fold')
        parser.add_argument('--use_amplify', type=bool, default=False, help='Data extension of training data set')

        # model set 
        parser.add_argument('--model_name', type=str, default='resnet50',
                            help='resnet50/resnet34/resnet101/resnet152/se_resnet50/MGN')
        parser.add_argument('--last_stride', type=int, default=1, help='last stride in the resnet model')

        # loss set
        parser.add_argument('--selected_loss', type=str, default='1.0*CrossEntropy+1.0*Triplet',
                            help='Select the loss function, CrossEntropy/SmoothCrossEntropy/Triplet')
        parser.add_argument('--margin', type=float, default=0.3, help='margin coefficient in triplet loss')

        # 优化器设置
        parser.add_argument('--optimizer_name', type=str, default='Adam',
                            help='which optimizer to use, SGD/SDG_bias/Adam/Adam_bias')
        # 学习率衰减策略
        parser.add_argument('--scheduler_name', type=str, default='Cosine',
                            help='which scheduler to use, StepLR/Cosine/WarmupMultiStepLR')
        parser.add_argument('--config_lr_optim_path', type=str, default='./config.json')

        # path set
        parser.add_argument('--save_path', type=str, default='./checkpoints')
        parser.add_argument('--dataset_root', type=str, default='./dataset/NAIC_data')

        # 其他设置
        parser.add_argument('--cython', type=bool, default=True, help='use cython or python to eval')
        parser.add_argument('--dist', type=str, default='cos_dist',
                            help='How to measure similarity, cos_dist/re_rank/euclidean_dist')

        config = parser.parse_args()

        with open(config.config_lr_optim_path, 'r', encoding='utf-8') as json_file:
            config_lr_optim = json.load(json_file)
            # dict to namespace
        config_lr_optim = Namespace(**config_lr_optim)

        config = Namespace(**vars(config), **vars(config_lr_optim))

    return config


if __name__ == '__main__':
    config = get_config()
