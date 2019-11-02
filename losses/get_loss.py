import torch.nn.functional as F

from losses.triplet_loss import TripletLoss, CrossEntropyLabelSmooth


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


