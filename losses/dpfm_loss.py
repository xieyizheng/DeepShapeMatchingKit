# Loss functions from original DPFM repo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.registry import LOSS_REGISTRY

class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum((a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.binary_loss = nn.BCELoss(reduction="none")

    def forward(self, prediction, gt):
        class_loss = self.binary_loss(prediction, gt)

        # Handle degenerate cases where gt is all 0s or all 1s
        n_positive = gt.sum()
        total = gt.size(0)
        if n_positive == 0 or n_positive == total:
            # Note: When gt contains only one class, reweighting would zero out the loss.
            # Fallback to unweighted BCE loss in these cases.
            return torch.mean(class_loss)

        weights = torch.ones_like(gt)
        w_negative = n_positive / total
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative

        return torch.mean(weights * class_loss)

class NCESoftmaxLoss(nn.Module):
    def __init__(self, nce_t, nce_num_pairs=None):
        super().__init__()
        self.nce_t = nce_t
        self.nce_num_pairs = nce_num_pairs
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat_x, feat_y, corr_x, corr_y): #  we've changed orignal map21 to corr_x and corr_y to take into accound partial to partial
        # don't consider batch for ease of implementation

        feat_x, feat_y = feat_x.squeeze(0), feat_y.squeeze(0)
        feat_x, feat_y = F.normalize(feat_x, p=2, dim=-1), F.normalize(feat_y, p=2, dim=-1)

        logits = feat_x @ feat_y.transpose(0,1) / self.nce_t   # Nx x Ny

        logits_x = logits[corr_x]
        logits_y = logits.transpose(0,1)[corr_y]

        loss_x = F.cross_entropy(logits_x, corr_y)
        loss_y = F.cross_entropy(logits_y, corr_x)

        return loss_x + loss_y

@LOSS_REGISTRY.register()
class DPFMLoss(nn.Module):
    def __init__(self, w_fmap=1, w_acc=1, w_nce=0.1, nce_t=0.07, nce_num_pairs=4096):
        super().__init__()

        self.w_fmap = w_fmap
        self.w_acc = w_acc
        self.w_nce = w_nce

        self.frob_loss = FrobeniusLoss()
        self.binary_loss = WeightedBCELoss()
        self.nce_softmax_loss = NCESoftmaxLoss(nce_t, nce_num_pairs)

    def forward(self, C12, C_gt, corr_x, corr_y, feat1, feat2, overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21): # we've changed map21 to corr_x and corr_y
        loss = 0

        # fmap loss
        fmap_loss = self.frob_loss(C12, C_gt) * self.w_fmap
        loss += fmap_loss

        # -------------

        # overlap loss
        acc_loss = self.binary_loss(overlap_score12, gt_partiality_mask12.float()) * self.w_acc
        acc_loss += self.binary_loss(overlap_score21, gt_partiality_mask21.float()) * self.w_acc

        # -------------

        loss += acc_loss

        # nce loss
        nce_loss = self.nce_softmax_loss(feat1, feat2, corr_x, corr_y) * self.w_nce
        loss += nce_loss

        return fmap_loss, acc_loss, nce_loss # we return all the losses separately so that we can log and monitor them


