import torch
import torch.nn.functional as F


def focal_loss(input, target, OHEM_percent=None):
    gamma = 2
    assert target.size() == input.size()

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    invprobs = F.logsigmoid(-input * (target * 2 - 1))
    loss = (invprobs * gamma).exp() * loss

    if OHEM_percent is None:
        return loss.mean()
    else:
        OHEM, _ = loss.topk(k=int(10008 * OHEM_percent), dim=1, largest=True, sorted=True)
        return OHEM.mean()


def bce_loss(input, target, OHEM_percent=None):
    if OHEM_percent is None:
        loss = F.binary_cross_entropy_with_logits(input, target, reduce=True)
        return loss
    else:
        loss = F.binary_cross_entropy_with_logits(input, target, reduce=False)
        value, index = loss.topk(int(10008 * OHEM_percent), dim=1, largest=True, sorted=True)
        return value.mean()


def focal_OHEM(results, labels, labels_onehot, OHEM_percent=0.1):
    batch_size, class_num = results.shape
    labels = labels.view(-1)
    loss0 = bce_loss(results, labels_onehot, OHEM_percent)
    loss1 = focal_loss(results, labels_onehot, OHEM_percent)
    # nonzero() 返回非零位置索引 [[第一个非零元素在第几行, 第二个非零元素在第几行, ...], [第一个非零元素在第几列, 第二个非零元素在第几列, ...]]
    indexs_ = (labels != class_num).nonzero().view(-1)
    if len(indexs_) == 0:
        return loss0 + loss1
    results_ = results[torch.arange(0, len(results))[indexs_], labels[indexs_]].contiguous()
    loss2 = focal_loss(results_, torch.ones_like(results_).float().cuda())
    return loss0 + loss1 + loss2
