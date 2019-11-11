from torchvision import models
import torch
import torch.nn as nn


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
        model.layer4[0].conv2 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3),
                                                          stride=(self.last_stride, self.last_stride), padding=(1, 1),
                                                          bias=False)
        model.layer4[0].downsample[0] = torch.nn.Conv2d(1024, 2048, kernel_size=(1, 1),
                                                                  stride=(self.last_stride, self.last_stride),
                                                                  bias=False)

        self.resnet_layer = torch.nn.Sequential(*list(model.children())[:-1])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.bottleneck = torch.nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc = torch.nn.Linear(in_features=2048, out_features=self.num_classes, bias=False)

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
        features = self.bottleneck(global_features)
        scores = self.fc(features)
        return scores, global_features, features


if __name__ == '__main__':
    inputs = torch.rand((64, 3, 256, 128))
    custom_resnet = CustomResnet('resnet50', last_stride=1, num_classes=2000)
    scores, global_features, features = custom_resnet(inputs)
    print(scores.size(), global_features.size(), features.size())
