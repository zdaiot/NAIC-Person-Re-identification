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
from models.model import build_model, get_model
from dataset.NAIC_dataset import get_baseline_loader
from utils.set_seed import seed_torch


class TrainBaseline(object):
    def __init__(self, config, num_classes):
        """

        :param config: 配置参数
        :param num_query: 该fold查询集的数量；类型为int
        :param num_classes: 该fold训练集的类别数；类型为int
        :param train_valid_ratio: 该fold训练集与验证集之间的比例；类型为float
        :param fold: 训练的哪一折；类型为int
        """
        self.num_classes = num_classes

        self.model_name = config.model_name
        self.last_stride = config.last_stride
        self.num_gpus = torch.cuda.device_count()
        print('Using {} GPUS'.format(self.num_gpus))
        print('NUM_CLASS: {}'.format(self.num_classes))

        # 加载模型，只要有GPU，则使用DataParallel函数，当GPU有多个GPU时，调用sync_bn函数
        self.model = get_model(self.model_name, self.num_classes, self.last_stride)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

        # 加载超参数
        self.epoch = config.epoch

        # 实例化实现各种子函数的 solver 类
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.solver = Solver(self.model, self.device)

        # 加载损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

        # 加载优化函数以及学习率衰减策略
        self.optim = optim.Adam(self.model.module.parameters(), config.base_lr, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, self.epoch+10)

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
                loss = self.criterion(labels_predict, labels.to(self.device))
                epoch_loss += loss.item()
                self.solver.backword(self.optim, loss)

                train_acc = (labels_predict.max(1)[1] == labels.to(self.device)).float().mean()

                # 保存到tensorboard，每一步存储一个
                self.writer.add_scalar('train_loss', loss.item(), global_step + i)
                self.writer.add_scalar('train_acc', train_acc, global_step + i)

                descript = "Train Loss: %.7f, Train Acc :%.2f, Lr :%.7f" % (loss.item(),
                                                                            train_acc.item() * 100,
                                                                            self.scheduler.get_lr()[0])
                tbar.set_description(desc=descript)

            # 每一个epoch完毕之后，执行学习率衰减
            self.scheduler.step()
            global_step += len(train_loader)

            # Print the log info
            print('Finish Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch, epoch_loss / len(tbar)))
            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
            }

            self.solver.save_checkpoint(
                os.path.join(self.model_path, '{}.pth'.format(self.model_name)), state, False)
            self.writer.add_scalar('lr', self.scheduler.get_lr()[0], epoch)


if __name__ == "__main__":
    config = get_config()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset_root = os.path.join(config.dataset_root, '初赛训练集')
    train_loader, num_classes = get_baseline_loader(train_dataset_root, config.batch_size, config.num_workers, config.shuffle_train, mean, std)
    train_baseline = TrainBaseline(config, num_classes)
    train_baseline.train(train_loader)
