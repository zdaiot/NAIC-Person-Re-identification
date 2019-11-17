from bisect import bisect_right
import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method="linear",
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
