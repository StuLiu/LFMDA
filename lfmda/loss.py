
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torch.autograd import Variable
import numpy as np


MSELoss = nn.MSELoss


class KLDLoss(nn.Module):

    def __init__(self, temperature=2.0, ignore_label=255):

        super(KLDLoss, self).__init__()

        self.temperature = temperature
        self.ignore_label = ignore_label

    def forward(self, stu_out, tea_out, labels):
        """
        Args:
            stu_out: contain shape (n, k) vector. or (b, k, h, w)
            tea_out: contain shape (n, k) vector. or (b, k, h, w)
        """
        assert stu_out.shape == tea_out.shape
        if stu_out.dim() == 4:
            feat_channel = stu_out.size(1)
            stu_out = stu_out.permute(0, 2, 3, 1).reshape(-1, feat_channel)
            tea_out = tea_out.permute(0, 2, 3, 1).reshape(-1, feat_channel)
        labels = labels.view(-1, 1)

        masks = (labels != self.ignore_label).int()

        stu_out_ = tnf.log_softmax(stu_out / self.temperature, dim=1)
        tea_out_ = tnf.softmax(tea_out.detach() / self.temperature, dim=1)

        loss_kd = tnf.kl_div(stu_out_, tea_out_, reduction='none')

        loss_kd = torch.sum(loss_kd * masks) / (torch.sum(masks) + 1e-7)

        return loss_kd


class KDMSELoss(nn.Module):

    def __init__(self, temperature=8.0, ignore_label=-1):
        super().__init__()
        self.temperature = temperature
        self.ignore_label = ignore_label
        self.mse_criterion = nn.MSELoss()

    def forward(self, feat_stu, feat_tea, labels):
        return self.mse_criterion(feat_stu, feat_tea)


class PrototypeContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.3, ignore_label=-1):
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.ignore_label = ignore_label
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, Proto, feat, labels):
        """
        Args:
            Proto: shape: (C, A) the mean representation of each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )

        Returns:

        """
        assert not Proto.requires_grad and not labels.requires_grad and feat.requires_grad
        if feat.dim() != 2:
            k = feat.size(1)
            feat = feat.permute(0, 2, 3, 1).reshape(-1, k)
        if labels.dim() != 1:
            labels = labels.reshape(-1, )
        assert feat.dim() == 2 and labels.dim() == 1
        # remove IGNORE_LABEL pixels
        mask = (labels != self.ignore_label)
        labels = labels[mask]
        feat = feat[mask]

        feat = tnf.normalize(feat, p=2, dim=1)
        Proto = tnf.normalize(Proto, p=2, dim=1)

        logits = feat.mm(Proto.permute(1, 0).contiguous())
        logits = logits / self.temperature

        loss = self.ce_criterion(logits, labels)
        return loss


def channel_1toN(img, num_channel):
    T = torch.LongTensor(num_channel, img.shape[1], img.shape[2]).zero_()
    mask = torch.LongTensor(img.shape[1], img.shape[2]).zero_()
    for i in range(num_channel):
        T[i] = T[i] + i
        layer = T[i] - img
        T[i] = torch.from_numpy(np.logical_not(np.logical_xor(layer.numpy(), mask.numpy())).astype(int))
    return T.float()


class WeightedBCEWithLogitsLoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average

    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)


class CrossEntropy2d(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        N, C, H, W = predict.size()
        sm = nn.Softmax2d()

        P = sm(predict)
        P = torch.clamp(P, min = 1e-9, max = 1-(1e-9))

        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask].view(1, -1)
        predict = P[target_mask.view(N, 1, H, W).repeat(1, C, 1, 1)].view(C, -1)
        probs = torch.gather(predict, dim = 0, index = target)
        log_p = probs.log()
        batch_loss = -(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:

            loss = batch_loss.sum()
        return loss


def get_kd_loss(loss_name='MSELoss', **kwargs):
    if loss_name == 'MSELoss':
        return KDMSELoss()
    elif loss_name == 'KLDLoss':
        return KLDLoss(
            ignore_label=kwargs.get('ignore_label', 255)
        )
    elif loss_name == 'PrototypeContrastiveLoss':
        return PrototypeContrastiveLoss(
            temperature=kwargs.get('temperature', 0.3),
            ignore_label=kwargs.get('ignore_label', 255)
        )
    elif loss_name == 'None':
        return None
    else:
        raise NotImplemented(f'Errors: kd loss {loss_name} is not implemented!')
