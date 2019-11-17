from torchvision import models
import torch
import torch.nn as nn
import pretrainedmodels
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


class CustomModel(nn.Module):
    def __init__(self, model_name, last_stride, num_classes):
        """

        :param model_name: resnet模型的名称；类型为str
        :param last_stride: resnet最后一个下采样层的步长；类型为int
        :param num_classes: 类别数目；类型为int
        """
        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.last_stride = last_stride
        self.num_classes = num_classes

        if self.model_name.startswith('resnet'):
            model = getattr(models, self.model_name)(pretrained=True)
            if self.model_name == 'resnet18' or self.model_name == 'resnet34':
                model.layer4[0].conv1.stride = (self.last_stride, self.last_stride)
            else:
                model.layer4[0].conv2.stride = (self.last_stride, self.last_stride)
            model.layer4[0].downsample[0].stride = (self.last_stride, self.last_stride)
            in_features = model.fc.in_features
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-1])

        elif self.model_name.startswith('dpn'):
            model = getattr(pretrainedmodels, self.model_name)(pretrained='imagenet')
            in_features = model.last_linear.in_channels
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-1])
            self.feature_layer.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        elif self.model_name.startswith('densenet'):
            model = getattr(pretrainedmodels, self.model_name)(pretrained='imagenet')
            in_features = model.last_linear.in_features
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-1])
            self.feature_layer.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        else:
            model = getattr(pretrainedmodels, self.model_name)(pretrained='imagenet')
            model.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # for p in model.layer0.parameters(): p.requires_grad = False
            # for p in model.layer1.parameters(): p.requires_grad = False
            in_features = model.last_linear.in_features
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-1])

        # self.bottleneck = torch.nn.BatchNorm1d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.fc = torch.nn.Linear(in_features=in_features, out_features=self.num_classes, bias=False)
        self.classifier = ClassBlock(in_features, self.num_classes, droprate=0.5, return_f=True)

    def forward(self, x):
        """

        :param x: 网络的输入；类型为tensor；维度为[batch_size, channel, height, width]
        :return score: 网络预测的类别；类型为tensor；维度为[batch_size, num_classes]
        :return global_features: 网络预测的全局特征，用于triplet loss；类型为tensor；维度为[btch_size, num_features]
        :return features: 网络预测的最终特征，用于分类损失；类型为tensor；维度为[batch_size, num_features]
        """
        global_features = self.feature_layer(x)
        global_features = global_features.view(global_features.shape[0], -1)
        # features = self.bottleneck(global_features)
        # scores = self.fc(features)
        scores, features = self.classifier(global_features)
        return scores, global_features, features

    def get_classify_result(self, outputs, labels, device):
        return (outputs[0].max(1)[1] == labels.to(device)).float()


if __name__ == '__main__':
    inputs = torch.rand((64, 3, 256, 128))
    custom_resnet = CustomModel('se_resnet50', last_stride=1, num_classes=2000)
    scores, global_features, features = custom_resnet(inputs)
    print(scores.size(), global_features.size(), features.size())
