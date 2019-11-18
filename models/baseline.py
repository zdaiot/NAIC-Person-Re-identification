import torch
from torch import nn

from models.backbones.resnet import ResNet
from models.weights_init import weights_init_kaiming, weights_init_classifier


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path):
        """

        :param num_classes: 类别数目；类型为int
        :param last_stride: resnet最后一个下采样层的步长；类型为int
        :param model_path: resnet预训练模型的权重；类型为str
        """
        super(Baseline, self).__init__()
        self.base = ResNet(last_stride)
        self.base.load_param(model_path)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def load_param(self, model_path):
        param = torch.load(model_path)
        for i in param:
            if 'fc' in i: continue
            if i not in self.state_dict().keys(): continue
            if param[i].shape != self.state_dict()[i].shape: continue
            self.state_dict()[i].copy_(param[i])

    def forward(self, x):
        global_features = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_features = global_features.view(global_features.shape[0], -1)  # flatten to (bs, 2048)
        features = self.bottleneck(global_features)  # normalize for angular softmax
        cls_score = self.classifier(features)
        return cls_score, global_features, features  # global feature for triplet loss

    def get_classify_result(self, outputs, labels, device):
        return (outputs[0].max(1)[1] == labels.to(device)).float()


if __name__ == '__main__':
    inputs = torch.rand((64, 3, 256, 128))
    model = Baseline(num_classes=2000, last_stride=1, model_path='dataset/NAIC_data/resnet50-19c8e357.pth')
    scores, global_features, features = model(inputs)
    print(scores.size(), global_features.size(), features.size())

