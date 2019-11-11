from torchvision import models
import torch
import torch.nn as nn
from models.weights_init import weights_init_kaiming_another, weights_init_classifier_another


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming_another)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier_another)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


class CustomResnet(nn.Module):
    def __init__(self, resnet_name, last_stride, num_classes):
        """

        :param resnet_name: resnet模型的名称；类型为str
        :param last_stride: resnet最后一个下采样层的步长；类型为int
        :param num_classes: 类别数目；类型为int
        """
        super(CustomResnet, self).__init__()
        self.resnet_name = resnet_name
        self.last_stride = last_stride
        self.num_classes = num_classes

        model = self.get_official_resnet()
        self.resnet_layer, self.avgpool, self.bottleneck, self.fc = None, None, None, None
        self.get_custom_resnet(model)

    def get_official_resnet(self):
        """ 得到官方的resnet模型
        """
        if self.resnet_name == 'resnet50':
            return models.resnet50(pretrained=True)
        elif self.resnet_name == 'resnet34':
            return models.resnet34(pretrained=True)
        elif self.resnet_name == 'resnet101':
            return models.resnet101(pretrained=True)
        elif self.resnet_name == 'resnet152':
            return models.resnet152(pretrained=True)

    def get_custom_resnet(self, model):
        """ 得到自己的resnet模型

        :param model: 官方的resnet模型
        :return: None
        """
        model.layer4[0].conv2.stride = (self.last_stride, self.last_stride)
        model.layer4[0].downsample[0].stride = (self.last_stride, self.last_stride)

        self.resnet_layer = torch.nn.Sequential(*list(model.children())[:-1])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.bottleneck = torch.nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.fc = torch.nn.Linear(in_features=2048, out_features=self.num_classes, bias=False)
        self.classifier = ClassBlock(2048, self.num_classes, droprate=0.5, return_f=True)

    def forward(self, x):
        """

        :param x: 网络的输入；类型为tensor；维度为[batch_size, channel, height, width]
        :return score: 网络预测的类别；类型为tensor；维度为[batch_size, num_classes]
        :return global_features: 网络预测的全局特征，用于triplet loss；类型为tensor；维度为[btch_size, num_features]
        :return features: 网络预测的最终特征，用于分类损失；类型为tensor；维度为[batch_size, num_features]
        """
        x = self.resnet_layer(x)
        global_features = self.avgpool(x)
        global_features = global_features.view(global_features.shape[0], -1)
        # features = self.bottleneck(global_features)
        # scores = self.fc(features)
        scores, features = self.classifier(global_features)
        return scores, global_features, features


if __name__ == '__main__':
    inputs = torch.rand((64, 3, 256, 128))
    custom_resnet = CustomResnet('resnet50', last_stride=1, num_classes=2000)
    scores, global_features, features = custom_resnet(inputs)
    print(scores.size(), global_features.size(), features.size())
