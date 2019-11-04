import tqdm
import datetime
import os
import codecs
import json
import time
import pickle
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from solver import Solver
from models.model import build_model
from losses.get_loss import get_loss
from dataset.NAIC_dataset import get_loaders
from utils.set_seed import seed_torch
from models.sync_bn.batchnorm import convert_model
from utils.custom_optim import make_optimizer, WarmupMultiStepLR
from evaluate import euclidean_dist, eval_func


class TrainVal():
    def __init__(self, config, num_query, num_classes, train_valid_ratio, fold):
        """

        :param config: 配置参数
        :param num_query: 该fold查询集的数量；类型为int
        :param num_classes: 该fold训练集的类别数；类型为int
        :param train_valid_ratio: 该fold训练集与验证集之间的比例；类型为float
        :param fold: 训练的哪一折；类型为int
        """
        self.num_query = num_query
        self.num_classes = num_classes
        self.fold = fold

        self.model_name = config.model_name
        self.last_stride = config.last_stride
        self.cython = config.cython
        self.num_gpus = torch.cuda.device_count()
        print('Using {} GPUS'.format(self.num_gpus))
        print('TRAIN_VALID_RATIO', train_valid_ratio)
        if self.cython:
            print('USE CYTHON TO EVAL!')

        # 加载模型，只要有GPU，则使用DataParallel函数，当GPU有多个GPU时，调用sync_bn函数
        self.model = build_model(self.model_name, self.num_classes, self.last_stride)
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

        # 加载损失函数
        self.criterion = get_loss(config.selected_loss, config.margin, config.label_smooth, self.num_classes)

        # 加载优化函数以及学习率衰减策略
        self.optim = make_optimizer(config.optimizer_name, config.base_lr, config.momentum_SGD, config.bias_lr_factor,
                                    config.weight_decay, config.weight_decay_bias, self.model, self.num_gpus)
        self.scheduler = WarmupMultiStepLR(self.optim, config.steps, config.gamma, config.warmup_factor,
                                           config.warmup_iters, config.warmup_method)

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
        self.max_average_score = 0

    def train(self, train_loader, valid_loader):
        """ 完成模型的训练，保存模型与日志

        :param train_loader: 训练集的Dataloader
        :param valid_loader: 验证集的Dataloader
        :return: None
        """

        global_step = 0

        for epoch in range(self.epoch):
            epoch += 1
            epoch_loss = 0
            self.model.train()

            tbar = tqdm.tqdm(train_loader)
            for i, (images, labels) in enumerate(tbar):
                # 网络的前向传播与反向传播
                labels_predict, global_features, _ = self.solver.forward(images)
                loss = self.solver.cal_loss(labels_predict, global_features, labels, self.criterion)
                epoch_loss += loss.item()
                self.solver.backword(self.optim, loss)

                train_acc = (labels_predict.max(1)[1] == labels.to(self.device)).float().mean()

                # 保存到tensorboard，每一步存储一个
                self.writer.add_scalar('train_loss', loss.item(), global_step + i)
                self.writer.add_scalar('train_acc', train_acc, global_step + i)

                descript = "Fold: %d, Train Loss: %.7f, Train Acc :%.2f, Lr :%.7f" % (self.fold, loss.item(),
                                                                                      train_acc.item() * 100,
                                                                                      self.scheduler.get_lr()[0])
                tbar.set_description(desc=descript)

            # 每一个epoch完毕之后，执行学习率衰减
            self.scheduler.step()
            global_step += len(train_loader)

            # Print the log info
            print('Finish Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch, epoch_loss / len(tbar)))

            # 验证模型
            rank1, mAP, average_score, loss_mean_valid = self.validation(valid_loader)

            if average_score > self.max_average_score:
                is_best = True
                self.max_average_score = average_score
            else:
                is_best = False

            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'max_average_score': self.max_average_score,
            }

            self.solver.save_checkpoint(
                os.path.join(self.model_path, '{}_fold{}.pth'.format(self.model_name, self.fold)), state, is_best)
            self.writer.add_scalar('lr', self.scheduler.get_lr()[0], epoch)
            self.writer.add_scalar('valid_loss', loss_mean_valid, epoch)
            self.writer.add_scalar('rank1', rank1, epoch)
            self.writer.add_scalar('mAP', mAP, epoch)
            self.writer.add_scalar('average_score', average_score, epoch)

    def validation(self, valid_loader):
        """ 完成模型的验证过程

        :param valid_loader: 验证集的Dataloader
        :return: None
        """
        self.model.eval()
        tbar = tqdm.tqdm(valid_loader)
        loss_sum = 0
        features_all, labels_all = [], []
        with torch.no_grad():
            for i, (images, labels, paths) in enumerate(tbar):
                # 完成网络的前向传播
                labels_predict, global_features, features = self.solver.forward(images)
                loss = self.solver.cal_loss(labels_predict, global_features, labels, self.criterion)
                loss_sum += loss.item()

                features_all.append(features.detach().cpu())
                labels_all.append(labels)

                descript = "Val Loss: {:.7f}".format(loss.item())
                tbar.set_description(desc=descript)
        loss_mean = loss_sum / len(tbar)

        features_all = torch.cat(features_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        query_features = features_all[:self.num_query]
        query_labels = labels_all[:self.num_query]

        gallery_features = features_all[self.num_query:]
        gallery_labels = labels_all[self.num_query:]

        distmat = euclidean_dist(query_features, gallery_features)
        all_rank_precison, mAP, _ = eval_func(distmat.numpy(), query_labels.numpy(), gallery_labels.numpy(),
                                              use_cython=self.cython)

        rank1 = all_rank_precison[0]
        average_score = 0.5 * rank1 + 0.5 * mAP
        print('Rank1: {:.2%}, mAP {:.2%}, average score {:.2%}'.format(rank1, mAP, average_score))
        return rank1, mAP, average_score, loss_mean


if __name__ == "__main__":
    config = get_config()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset_root = os.path.join(config.dataset_root, '初赛训练集')
    train_dataloader_folds, valid_dataloader_folds, num_query_folds, num_classes_folds, train_valid_ratio_folds = get_loaders(
        train_dataset_root, config.n_splits, config.batch_size, config.num_workers, config.shuffle_train,
        config.use_erase, mean, std)

    for fold_index, [train_loader, valid_loader, num_query, num_classes, train_valid_ratio] in enumerate(zip(
            train_dataloader_folds, valid_dataloader_folds, num_query_folds, num_classes_folds,
            train_valid_ratio_folds)):

        if fold_index not in config.selected_fold:
            continue
        # 注意fold之间的因为类别数不同所以模型也不同，所以均要实例化TrainVal
        train_val = TrainVal(config, num_query, num_classes, train_valid_ratio, fold_index)
        train_val.train(train_loader, valid_loader)
