from models.baseline import Baseline
from models.custom_resnet import CustomResnet


def build_model(model_name, num_classes, last_stride, model_pretrain_path='dataset/NAIC_data/resnet50-19c8e357.pth'):
    """ 作者实现的resnet

    :param model_name: 模型的名称；类型为str
    :param num_classes: 训练集的类别数；类型为int
    :param last_stride: 模型最后一层的步长；类型为int
    :param model_pretrain_path: ImageNet预训练权重的位置，类型为str
    :return: 模型的实例
    """
    if model_name == 'resnet50':
        return Baseline(num_classes, last_stride, model_pretrain_path)


def get_model(model_name, num_classes, last_stride):
    """ 我自己实现的resnet

    :param model_name: 模型的名称；类型为str
    :param num_classes: 训练集的类别数；类型为int
    :param last_stride: 模型最后一层的步长；类型为int
    :return: 模型的实例
    """
    if 'resnet' in model_name:
        return CustomResnet('resnet50', last_stride=last_stride, num_classes=num_classes)
