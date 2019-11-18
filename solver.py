"""
该文件的功能：实现模型的前向传播，反向传播，损失函数计算，保存模型，加载模型功能
"""

import torch
import shutil
import os


class Solver:
    def __init__(self, model, device):
        ''' 完成solver类的初始化
        Args:
            model: 网络模型
        '''
        self.model = model
        self.device = device

    def forward(self, inputs):
        """ 实现网络的前向传播功能

        Args:
            inputs: 网络的输入；类型为tuple；具体维度和模型有关

        Return:
            outputs: 网络的输出，具体维度和含义与self.model有关
        """
        inputs = list(inputs)
        for index, x in enumerate(inputs):
            inputs[index] = x.to(self.device)
        inputs = tuple(inputs)
        outputs = self.model(*inputs)
        return outputs

    def tta(self, inputs):
        """测试时数据增强

        Args:
            inputs: 网络的输入；类型为tuple；具体维度和模型有关
        Return:
            preds: 经过tta后的输出
        """
        inputs = list(inputs)
        for index, x in enumerate(inputs):
            inputs[index] = x.to(self.device)
        inputs = tuple(inputs)

        # 原图，outputs的最后一个数据保留可以用于行人重识别的特征
        outputs = self.model(*inputs)
        pred_origin = outputs[-1]

        # 水平翻转，outputs的最后一个数据保留可以用于行人重识别的特征
        images_hflp = torch.flip(inputs[0], dims=[3])
        outputs = self.model(*(images_hflp, inputs[1]))
        pred_hflip = outputs[-1]

        preds = pred_origin + pred_hflip

        # 求平均
        # preds = preds / 2.0
        return preds

    def cal_loss(self, predicts, targets, criterion):
        """ 根据真实类标和预测出的类标计算损失

        Args:
            predicts: 网络的预测输出，类型为tuple，具体维度和self.model有关
            targets: 真实类标，具体维度和self.model有关
            criterion: 使用的损失函数

        Return:
            损失函数的输出
        """
        targets = targets.to(self.device)
        return criterion(predicts, targets)

    def backword(self, optimizer, loss):
        ''' 实现网络的反向传播
        
        Args:
            optimizer: 模型使用的优化器
            loss: 模型计算出的loss值
        Return:
            None
        '''
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def save_checkpoint(self, save_path, state, is_best):
        ''' 保存模型参数

        Args:
            save_path: 要保存的权重路径
            state: 存有模型参数、最大dice等信息的字典
            is_best: 是否为最优模型
        Return:
            None
        '''
        torch.save(state, save_path)
        if is_best:
            print('Saving Best Model.')
            save_best_path = save_path.replace('.pth', '_best.pth')
            shutil.copyfile(save_path, save_best_path)
    
    def load_checkpoint(self, load_path):
        ''' 保存模型参数

        Args:
            load_path: 要加载的权重路径
        
        Return:
            加载过权重的模型
        '''
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path)
            self.model.module.load_state_dict(checkpoint['state_dict'])
            print('Successfully Loaded from %s' % (load_path))
            return self.model
        else:
            raise FileNotFoundError("Can not find weight file in {}".format(load_path))
