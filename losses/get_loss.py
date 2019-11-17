import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.triplet_loss import TripletLoss, CrossEntropyLabelSmooth, TripletLossOrigin


def get_loss(selected_loss, margin, label_smooth, num_classes):
    """

    :param selected_loss: loss的种类；类型为str
    :param margin: triplet loss中的margin参数；类型为float
    :param label_smooth: 交叉熵函数中是否使用label smooth；类型为bool
    :param num_classes: 训练集的类别数；类型为int
    :return: 损失函数；类型为可调用的函数
    """
    triplet = TripletLoss(margin)

    if label_smooth:
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, num_classes:", num_classes)
    else:
        xent = F.cross_entropy

    if selected_loss == 'softmax':
        def loss_func(score, feat, target):
            return xent(score, target)
    elif selected_loss == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif selected_loss == 'softmax_triplet':
        def loss_func(score, feat, target):
            return xent(score, target) + triplet(feat, target)[0]
    else:
        print('expected selected_loss should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(selected_loss))
    return loss_func


class Loss(nn.modules.loss._Loss):
    def __init__(self, model_name, loss_name, margin, num_classes):
        super(Loss, self).__init__()
        self.model_name = model_name
        self.loss_name = loss_name
        self.loss_struct = []

        for loss in self.loss_name.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy':
                loss_function = nn.CrossEntropyLoss()
            elif loss_type == 'SmoothCrossEntropy':
                loss_function = CrossEntropyLabelSmooth(num_classes=num_classes)
            elif loss_type == 'Triplet':
                loss_function = TripletLoss(margin)
            else:
                assert "loss: {} not support yet".format(self.loss_name)

            self.loss_struct.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

        # 如果有多个损失函数，在加上一个求和操作
        if len(self.loss_struct) > 1:
            self.loss_struct.append({'type': 'Total', 'weight': 0, 'function': None})

        self.loss_module = nn.ModuleList([l['function'] for l in self.loss_struct if l['function'] is not None])

        # self.log的维度为[1, len(self.loss)]，前面几个分别存放某次迭代各个损失函数的损失值，最后一个存放某次迭代损失值之和
        self.log, self.log_sum = torch.zeros(len(self.loss_struct)), torch.zeros(len(self.loss_struct))

        if torch.cuda.is_available():
            self.loss_module = torch.nn.DataParallel(self.loss_module)
            self.loss_module.cuda()

    def forward(self, outputs, labels):
        """

        :param outputs: 网络的输出，具体维度和网络有关
        :param labels: 数据的真实类标，具体维度和网络有关
        :return loss_sum: 损失函数之和，未经过item()函数，可用于反向传播
        :return self.log: 维度为[1, len(self.loss)]，前面几个分别存放某次迭代各个损失函数的损失值（经过了item()），最后一个存放某次迭代损失值之和
        """
        losses = []
        # 计算每一个损失函数的损失值
        for i, l in enumerate(self.loss_struct):
            # 处理MGN网络的损失计算
            if self.model_name == 'MGN' and l['type'] == 'Triplet':
                loss = [l['function'](output, labels)[0] for output in outputs[8:11]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[i] = effective_loss.item()
                self.log_sum[i] += self.log[i]
            elif self.model_name == 'MGN' and l['type'] in ['CrossEntropy', 'SmoothCrossEntropy']:
                loss = [l['function'](output, labels) for output in outputs[:8]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[i] = effective_loss.item()
                self.log_sum[i] += self.log[i]

            # 处理其它网络的损失计算
            elif self.model_name != 'MGN' and l['type'] == 'Triplet':
                loss = l['function'](outputs[1], labels)[0]
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[i] = effective_loss.item()
                self.log_sum[i] += self.log[i]
            elif self.model_name == 'MGN' and l['type'] in ['CrossEntropy', 'SmoothCrossEntropy']:
                loss = l['function'](outputs[0], labels)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[i] = effective_loss.item()
                self.log_sum[i] += self.log[i]

            # 保留接口
            else:
                pass

        loss_sum = sum(losses)
        if len(self.loss_struct) > 1:
            self.log[-1] = loss_sum.item()
            self.log_sum[-1] = loss_sum.item()

        return loss_sum

    def record_loss_iteration(self, writer_function=None, global_step=None):
        descript = []
        for l, each_loss in zip(self.loss_struct, self.log):
            if writer_function:
                writer_function(l['type'] + 'iteration', each_loss, global_step)
            descript.append('[{}: {:.4f}]'.format(l['type'], each_loss))
        return ''.join(descript)

    def record_loss_epoch(self, num_iterations, writer_function=None, global_step=None):
        descript = []
        for l, each_loss in zip(self.loss_struct, self.log_sum):
            if writer_function:
                writer_function(l['type'] + 'epoch', each_loss/num_iterations, global_step)
            descript.append('[Average {}: {:.4f}]'.format(l['type'], each_loss/num_iterations))

        # 注意要把 self.log_sum清零
        self.log_sum = torch.zeros(len(self.loss_struct))
        return ''.join(descript)

