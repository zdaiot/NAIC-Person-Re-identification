import tqdm
import datetime
import os
import codecs
import json
import time
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from solver import Solver
from models.model import build_model, get_model
from losses.get_loss import Loss
from dataset.NAIC_dataset import get_loaders
from utils.set_seed import seed_torch
from models.sync_bn.batchnorm import convert_model
from utils.custom_optim import get_optimizer, get_scheduler
from evaluate import euclidean_dist, eval_func, re_rank, cos_dist


class TrainVal(object):
    def __init__(self, config, num_query, num_classes, num_valid_classes, fold, train_triplet=False):
        """

        :param config: 配置参数
        :param num_query: 该fold查询集的数量；类型为int
        :param num_classes: 该fold训练集的类别数；类型为int
        :param num_valid_classes: 该fold验证集的类别数；类型为int
        :param fold: 训练的哪一折；类型为int
        :param train_triplet: 是否只训练triplet损失；类型为bool
        """
        self.num_query = num_query
        self.num_classes = num_classes
        self.fold = fold

        self.model_name = config.model_name
        self.last_stride = config.last_stride
        self.dist = config.dist
        self.cython = config.cython
        self.num_gpus = torch.cuda.device_count()
        print('Using {} GPUS'.format(self.num_gpus))
        print('TRAIN_VALID_RATIO: {}'.format(self.num_classes/num_valid_classes))
        print('NUM_CLASS: {}'.format(self.num_classes))
        if self.cython:
            print('USE CYTHON TO EVAL!')
        print('USE LOSS: {}'.format(config.selected_loss))

        # 加载模型，只要有GPU，则使用DataParallel函数，当GPU有多个GPU时，调用sync_bn函数
        self.model = get_model(self.model_name, self.num_classes, self.last_stride)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            if self.num_gpus > 1:
                self.model = convert_model(self.model)
            self.model = self.model.cuda()

        # 加载超参数
        self.epoch = config.epoch

        # 实例化实现各种子函数的 solver 类
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.solver = Solver(self.model, self.device)

        # 如果只训练Triplet损失
        if train_triplet:
            assert 'CrossEntropy' not in config.selected_loss
            self.solver.load_checkpoint(os.path.join(self.model_path, '{}_fold{}_best.pth'.format(self.model_name,
                                                                                                  self.fold)))

        # 加载损失函数
        self.criterion = Loss(self.model_name, config.selected_loss, config.margin, self.num_classes)

        # 加载优化函数
        self.optim = get_optimizer(config, self.model)

        # 加载学习率衰减策略
        self.scheduler = get_scheduler(config, self.optim)

        # 创建保存权重的路径
        self.model_path = os.path.join(config.save_path, config.model_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # 保存json文件和初始化tensorboard
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_path, TIMESTAMP))
        with codecs.open(self.model_path + '/' + TIMESTAMP + '.json', 'w', "utf-8") as json_file:
            json.dump({k: v for k, v in config._get_kwargs()}, json_file, ensure_ascii=False)

        # 设置随机种子，注意交叉验证部分划分训练集和验证集的时候，要保持种子固定
        self.seed = int(time.time())
        seed_torch(self.seed)
        with open(self.model_path + '/' + TIMESTAMP + '.pkl', 'wb') as f:
            pickle.dump({'seed': self.seed}, f, -1)

        # 设置其他参数
        self.max_score = 0

    def train(self, train_loader, valid_loader):
        """ 完成模型的训练，保存模型与日志

        :param train_loader: 训练集的Dataloader
        :param valid_loader: 验证集的Dataloader
        :return: None
        """

        global_step = 0

        for epoch in range(self.epoch):
            epoch += 1
            self.model.train()
            images_number, epoch_corrects, index = 0, 0, 0

            tbar = tqdm.tqdm(train_loader)
            for index, (images, labels) in enumerate(tbar):
                # 网络的前向传播与反向传播
                outputs = self.solver.forward((images, labels))
                loss = self.solver.cal_loss(outputs, labels, self.criterion)
                self.solver.backword(self.optim, loss)

                images_number += images.size(0)
                epoch_corrects += self.model.module.get_classify_result(outputs, labels, self.device).sum()
                train_acc_iteration = self.model.module.get_classify_result(outputs, labels, self.device).mean() * 100

                # 保存到tensorboard，每一步存储一个
                global_step += 1
                descript = self.criterion.record_loss_iteration(self.writer.add_scalar, global_step)
                self.writer.add_scalar('TrainAccIteration', train_acc_iteration, global_step)

                descript = '[Train][epoch: {}/{}][Lr :{:.7f}][Acc: {:.2f}]'.format(epoch, self.epoch,
                                                                               self.scheduler.get_lr()[1],
                                                                               train_acc_iteration) + descript
                tbar.set_description(desc=descript)

            # 每一个epoch完毕之后，执行学习率衰减
            self.scheduler.step()

            # 写到tensorboard中
            epoch_acc = epoch_corrects / images_number * 100
            self.writer.add_scalar('TrainAccEpoch', epoch_acc, epoch)
            self.writer.add_scalar('Lr', self.scheduler.get_lr()[1], epoch)
            descript = self.criterion.record_loss_epoch(index, self.writer.add_scalar, epoch)

            # Print the log info
            print('[Finish epoch: {}/{}][Average Acc: {:.2}]'.format(epoch, self.epoch, epoch_acc) + descript)

            # 验证模型
            rank1, mAP, average_score = self.validation(valid_loader)

            if average_score > self.max_score:
                is_best = True
                self.max_score = average_score
            else:
                is_best = False

            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'max_score': self.max_score
            }

            self.solver.save_checkpoint(
                os.path.join(self.model_path, '{}_fold{}.pth'.format(self.model_name, self.fold)), state, is_best)
            self.writer.add_scalar('Rank1', rank1, epoch)
            self.writer.add_scalar('MAP', mAP, epoch)
            self.writer.add_scalar('AverageScore', average_score, epoch)

    def validation(self, valid_loader):
        """ 完成模型的验证过程

        :param valid_loader: 验证集的Dataloader
        :return rank1: rank1得分；类型为float
        :return mAP: 平均检索精度；类型为float
        :return average_score: 平均得分；类型为float
        """
        self.model.eval()
        tbar = tqdm.tqdm(valid_loader)
        features_all, labels_all = [], []
        with torch.no_grad():
            for i, (images, labels, paths) in enumerate(tbar):
                # 完成网络的前向传播
                # features = self.solver.forward((images, labels))[-1]
                features = self.solver.tta((images, labels))
                features_all.append(features.detach().cpu())
                labels_all.append(labels)

        features_all = torch.cat(features_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        query_features = features_all[:self.num_query]
        query_labels = labels_all[:self.num_query]

        gallery_features = features_all[self.num_query:]
        gallery_labels = labels_all[self.num_query:]

        if self.dist == 're_rank':
            distmat = re_rank(query_features, gallery_features)
        elif self.dist == 'cos_dist':
            distmat = cos_dist(query_features, gallery_features)
        elif self.dist == 'euclidean_dist':
            distmat = euclidean_dist(query_features, gallery_features)
        else:
            assert "Not implemented :{}".format(self.dist)

        all_rank_precison, mAP, _ = eval_func(distmat, query_labels.numpy(), gallery_labels.numpy(),
                                              use_cython=self.cython)

        rank1 = all_rank_precison[0]
        average_score = 0.5 * rank1 + 0.5 * mAP
        print('Rank1: {:.2%}, mAP {:.2%}, average score {:.2%}'.format(rank1, mAP, average_score))
        return rank1, mAP, average_score


if __name__ == "__main__":
    config = get_config()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if config.use_amplify:
        train_dataset_root = os.path.join(config.dataset_root, 'train_amplify')
    else:
        train_dataset_root = os.path.join(config.dataset_root, '初赛训练集')
    train_dataloader_folds, valid_dataloader_folds, num_query_folds, num_classes_folds = \
        get_loaders(
            train_dataset_root,
            config.n_splits,
            config.batch_size,
            config.num_instances,
            config.num_workers,
            config.augmentation_flag,
            config.erase_prob,
            config.gray_prob,
            mean, std
        )

    for fold_index, [train_loader, valid_loader, num_query, num_classes] in \
        enumerate(
            zip(
            train_dataloader_folds,
            valid_dataloader_folds,
            num_query_folds,
            num_classes_folds,
            )
    ):

        if fold_index not in config.selected_fold:
            continue
        # 注意fold之间的因为类别数不同所以模型也不同，所以均要实例化TrainVal
        train_val = TrainVal(config, num_query, num_classes[0], num_classes[1], fold_index)
        train_val.train(train_loader, valid_loader)
