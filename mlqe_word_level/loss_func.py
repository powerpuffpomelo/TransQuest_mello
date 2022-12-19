from torch import nn
import torch
from torch.nn import functional as F

# 0 ok 1 bad

class CELoss(nn.Module):
    def __init__(self, num_class=2, eps=1e-8, reduction='mean'):
        super(CELoss,self).__init__()
        self.num_class = num_class
        self.eps = eps
        self.reduction = reduction
        self.weight = torch.Tensor([0.4, 0.6])

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        labels = labels.view(-1)
        
        device = labels.device
        weight = self.weight.to(device)

        ce_func = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
        ce_loss = ce_func(preds, labels)

        if self.reduction == 'mean':
            ce_loss = ce_loss.mean()
        else:
            ce_loss = ce_loss.sum()
        return ce_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_class = 2, reduction='mean'):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha: 阿尔法α,类别权重.      
                    当α是列表时,为各类别权重；
                    当α为常数时,类别权重为[α, 1-α, 1-α, ....],
                    常用于目标检测算法中抑制背景类, 
                    retainnet中设置为0.25
        :param gamma: 伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes: 类别数量
        :param size_average: 损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.num_class = num_class
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds: 预测类别. size:[B,N,C] or [B,C]    分
                别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        labels = labels.view(-1)

        mask = labels.ge(0)
        preds = torch.masked_select(preds, mask.unsqueeze(-1)).view(-1, self.num_class)
        labels = torch.masked_select(labels, mask)
        alpha = [self.alpha if labels[i] == 0 else (1 - self.alpha) for i in range(labels.size(-1))]
        device = labels.device
        alpha = torch.Tensor(alpha).to(device)

        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然也可以使用log_softmax,然后进行exp操作)
        preds_softmax = F.softmax(preds, dim=-1)
        # 对所求概率进行 clamp 操作，不然当某一概率过小时，进行 log 操作，会使得 loss 变为 nan!!!
        preds_softmax = preds_softmax.clamp(min=0.0001, max=1.0)
        log_softmax = torch.log(preds_softmax)

        logpt = torch.gather(log_softmax, dim=1, index=labels.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了  同 torch.nn.CrossEntropyLoss(reduction='none')

        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = (1 - pt) ** self.gamma * alpha * 10 * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        else:
            focal_loss = focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, num_class=2, eps=1e-8):
        super().__init__()
        self.num_class = num_class
        self.eps = eps

    def forward(self, logits, labels):
        # assert logits.dim()==2 and labels.dim()==1
        logits = logits.view(-1, logits.size(-1))   # logits [seq_len, num_cls]
        labels = labels.view(-1)   # labels [seq_len]

        mask = labels.ge(0)
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.num_class)
        labels = torch.masked_select(labels, mask)

        probs = F.softmax(logits, dim=-1)

        # v1
        prob_1 = probs[:, 1]
        prob_0 = probs[:, 0]
        label_1 = (labels == 1).int()
        label_0 = (labels == 0).int()

        intersection_1 = 2 * torch.sum(prob_1 * label_1) + self.eps
        union_1 = torch.sum(prob_1) + torch.sum(label_1) + self.eps
        dice_1 = intersection_1 / union_1

        intersection_0 = 2 * torch.sum(prob_0 * label_0) + self.eps
        union_0 = torch.sum(prob_0) + torch.sum(label_0) + self.eps
        dice_0 = intersection_0 / union_0

        dice = dice_1 * dice_0
        loss = 1 - dice

        # v2 很糟没法用
        # probs = torch.gather(probs, dim=-1, index=labels.view(-1, 1)).squeeze()
        # intersection = 2 * torch.sum(probs * labels) + self.eps
        # union = torch.sum(probs) + torch.sum(labels) + self.eps
        # dice = intersection / union
        # loss = 1 - dice

        return loss


class LogitAdjustmentLoss(nn.Module):
    def __init__(self, num_class=2, tau=0.3, eps=1e-8, reduction='mean'):
        super(LogitAdjustmentLoss,self).__init__()
        self.tau = tau
        self.num_class = num_class
        self.eps = eps
        self.reduction = reduction
        self.prior = torch.Tensor([0.7, 0.3])

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        labels = labels.view(-1)
        
        mask = labels.ge(0)
        preds = torch.masked_select(preds, mask.unsqueeze(-1)).view(-1, self.num_class)
        labels = torch.masked_select(labels, mask)
        
        device = labels.device
        prior = self.prior.to(device)
        log_prior = torch.log(prior + self.eps).unsqueeze(0)
        # print(preds)
        preds = preds + self.tau * log_prior
        # print(log_prior)
        # print(preds)

        ce_func = torch.nn.CrossEntropyLoss(reduction='none')
        logit_adjustment_loss = ce_func(preds, labels)

        if self.reduction == 'mean':
            logit_adjustment_loss = logit_adjustment_loss.mean()
        else:
            logit_adjustment_loss = logit_adjustment_loss.sum()
        return logit_adjustment_loss


dc = LogitAdjustmentLoss()
logit = torch.rand((5, 2))
label = torch.tensor([1, 0, 0, -100, -100])
loss = dc(logit, label)
# print(loss)

# python3 /opt/tiger/fake_arnold/TransQuest_mello/mlqe_word_level/loss_func.py