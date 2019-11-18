from bisect import bisect_right
import torch
from torch import optim


def get_optimizer(config, model):
    """

    :param config: 配置参数
    :param model: 网络模型
    :return: 优化器
    """
    # 加载优化函数
    if config.optimizer_name == 'Adam':
        assert config.model_name != 'MGN', "This optimizer not support MGN"
        optimizer = optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, model.module.feature_layer.parameters()),
              'lr': config.base_lr * 0.1},
             {'params': model.module.classifier.parameters(), 'lr': config.base_lr}],
            weight_decay=config.weight_decay)
    elif config.optimizer_name == 'SGD':
        assert config.model_name != 'MGN', "This optimizer not support MGN"
        optimizer = optim.SGD(
            [{'params': model.module.feature_layer.parameters(), 'lr': config.base_lr * 0.1},
             {'params': model.module.classifier.parameters(), 'lr': config.base_lr}],
            weight_decay=config.weight_decay, momentum=config.SGD['momentum'], nesterov=True)
    elif config.optimizer_name == 'SGD_bias':
        optimizer = make_optimizer('SGD', config.base_lr, config.SGD_bias['momentum'],
                                   config.SGD_bias["bias_lr_factor"],
                                   config.weight_decay, config.SGD_bias['weight_decay_bias'], model)
    elif config.optimizer_name == 'Adam_bias':
        optimizer = make_optimizer('Adam', config.base_lr, config.Adam_bias['momentum'],
                                   config.Adam_bias["bias_lr_factor"],
                                   config.weight_decay, config.Adam_bias['weight_decay_bias'], model)
    return optimizer


def get_scheduler(config, optimizer):
    """

    :param config: 配置参数
    :param optimizer: 优化器
    :return: 学习率衰减策略
    """
    # 加载学习率衰减策略
    if config.scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.StepLR['decay_step'],
                                              gamma=config.StepLR["gamma"])
    elif config.scheduler_name == 'Cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.Cosine['restart_step'],
                                                         eta_min=config.Cosine['eta_min'])
    elif config.scheduler_name == 'author':
        scheduler = WarmupMultiStepLR(optimizer,
                                      config.WarmupMultiStepLR["steps"],
                                      config.WarmupMultiStepLR["gamma"],
                                      config.WarmupMultiStepLR["warmup_factor"],
                                      config.WarmupMultiStepLR["warmup_iters"],
                                      config.WarmupMultiStepLR["warmup_method"]
                                      )
    return scheduler


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear",
                 last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of" " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted got {}".format(warmup_method))

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def make_optimizer(optimizer_name, base_lr, momentum_SGD, bias_lr_factor, weight_decay, weight_decay_bias, model,
                   num_gpus=1):
    """ 加载优化器

    :param optimizer_name: 优化器的名称；类型为str
    :param base_lr: 基础学习率；类型为float
    :param momentum_SGD: SGD中的动量参数；类型为float
    :param bias_lr_factor: bias_lr = base_lr * bias_lr_factor；类型为float
    :param weight_decay: 权重衰减；类型为float
    :param weight_decay_bias: bias的权重衰减；类型为float
    :param model: 模型
    :param num_gpus: GPU的数量，暂时没有用到（原作者代码中lr = base_lr * num_gpus）；类型为int
    :return: 优化器
    """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        # linear scaling rule
        if "bias" in key:
            lr = base_lr * bias_lr_factor
            weight_decay = weight_decay_bias
        else:
            weight_decay = weight_decay
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, optimizer_name)(params, momentum=momentum_SGD, nesterov=True)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(params)
    return optimizer
