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
        parser.add_argument('--epoch', type=int, default=60, help='epoch')
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--selected_fold', type=list, default=[0], help='what folds for training?')

        # dataset set
        parser.add_argument('--augmentation_flag', type=bool, default=True,
                            help='if true, use DataAugmentation class in train set')
        parser.add_argument('--erase_prob', type=float, default=0.3,
                            help='when augmentation_flag=True, erase probability in DataAugmentation class')
        parser.add_argument('--gray_prob', type=float, default=0.3,
                            help='when augmentation_flag=True, gray probability in DataAugmentation class')
        parser.add_argument('--n_splits', type=int, default=4, help='n_splits_fold')
        parser.add_argument('--use_amplify', type=bool, default=False, help='Data extension of training data set')

        # model set 
        parser.add_argument('--model_name', type=str, default='MGN',
                            help='resnet50/resnet34/resnet101/resnet152/se_resnet50/MGN')
        parser.add_argument('--last_stride', type=int, default=1, help='last stride in the resnet model')

        # loss set
        parser.add_argument('--selected_loss', type=str, default='1*CrossEntropy+1*Triplet',
                            help='Select the loss function, softmax_triplet/softmax/triplet')
        parser.add_argument('--margin', type=float, default=0.3, help='margin coefficient in triplet loss')
        parser.add_argument('--label_smooth', type=bool, default=False, help='use label smooth in cross entropy')

        # 优化器设置
        parser.add_argument('--optimizer_name', type=str, default='author',
                            help='which optimizer to use, Adam/SGD/author')
        parser.add_argument('--momentum_SGD', type=float, default=0.9, help='momentum in SGD')
        parser.add_argument('--base_lr', type=float, default=1e-4, help='init lr')
        parser.add_argument('--bias_lr_factor', type=float, default=1,
                            help='only use when optimizer_name=author, bias_lr=base_lr*bias_lr_factor')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay in optimizer')
        parser.add_argument('--weight_decay_bias', type=float, default=0.0,
                            help='only use when optimizer_name=author, weight_decay for bias')

        # 学习率衰减策略
        parser.add_argument('--scheduler_name', type=str, default='author',
                            help='which scheduler to use, StepLR/COS/author')
        # 设置WarmupMultiStepLR, 只有当scheduler_name=author时下面参数才有作用
        parser.add_argument('--steps', type=list, default=[20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195,
                                                           210, 225, 240, 255], help='')
        parser.add_argument('--gamma', type=float, default=0.6, help='')
        parser.add_argument('--warmup_factor', type=float, default=0.01, help='')
        parser.add_argument('--warmup_iters', type=int, default=10,
                            help='The first warmup_iters epoch adopts warm up strategy')
        parser.add_argument('--warmup_method', type=str, default='linear', help='warmup method, constant/linear')

        # path set
        parser.add_argument('--save_path', type=str, default='./checkpoints')
        parser.add_argument('--dataset_root', type=str, default='./dataset/NAIC_data')

        # 其他设置
        parser.add_argument('--cython', type=bool, default=True, help='use cython or python to eval')
        parser.add_argument('--dist', type=str, default='cos_dist',
                            help='How to measure similarity, cos_dist/re_rank/euclidean_dist')

        config = parser.parse_args()
        # config = {k: v for k, v in args._get_kwargs()}

    return config


if __name__ == '__main__':
    config = get_config()
