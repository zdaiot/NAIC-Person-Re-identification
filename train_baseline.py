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
from dataset.NAIC_dataset import get_baseline_loader
from utils.set_seed import seed_torch
from models.sync_bn.batchnorm import convert_model
from utils.custom_optim import get_optimizer, get_scheduler


class TrainBaseline(object):
    def __init__(self, config, num_classes):
        """

        :param config: 配置参数
        :param num_classes: 训练集的类别数；类型为int
        """
        self.num_classes = num_classes

        self.model_name = config.model_name
        self.last_stride = config.last_stride
        self.num_gpus = torch.cuda.device_count()
        print('Using {} GPUS'.format(self.num_gpus))
        print('NUM_CLASS: {}'.format(self.num_classes))
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

    def train(self, train_loader):
        """ 完成模型的训练，保存模型与日志

        :param train_loader: 训练集的Dataloader
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

            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
            }

            self.solver.save_checkpoint(
                os.path.join(self.model_path, '{}.pth'.format(self.model_name)), state, False)


if __name__ == "__main__":
    config = get_config()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset_root = os.path.join(config.dataset_root, '初赛训练集')
    train_loader, num_classes = get_baseline_loader(
        train_dataset_root,
        config.batch_size,
        config.num_instances,
        config.num_workers,
        config.augmentation_flag,
        config.erase_prob,
        config.gray_prob,
        mean, std)
    train_baseline = TrainBaseline(config, num_classes)
    train_baseline.train(train_loader)
