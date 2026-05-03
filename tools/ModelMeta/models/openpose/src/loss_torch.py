import torch
import torch.nn as nn


class MyLoss(nn.Module):
    """
    Base class for other losses.
    """

    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__()

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.reduction = reduction

    def get_loss(self, x, weights=1.0):
        """
        Computes the weighted loss

        Args:
            x: Tensor
            weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
                inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
        """
        x = x.float() * weights

        if self.reduction == 'mean':
            x = torch.mean(x)
        elif self.reduction == 'sum':
            x = torch.sum(x)
        return x

    def forward(self, base, target):
        raise NotImplementedError


class openpose_loss(MyLoss):
    def __init__(self):
        super(openpose_loss, self).__init__()

    def mean_square_error(self, map1, map2, mask=None):
        if mask is None:
            mse = torch.mean((map1 - map2) ** 2)
            return mse

        squareMap = (map1 - map2) ** 2
        squareMap_mask = squareMap * mask
        mse = torch.mean(squareMap_mask)
        return mse

    def forward(self, logit_paf, logit_heatmap, gt_paf, gt_heatmap, ignore_mask):
        # Input
        # ignore_mask, make sure the ignore_mask is the 0-1 array instead of the bool-false array
        heatmaps_loss = []
        pafs_loss = []
        total_loss = 0

        paf_masks = ignore_mask.unsqueeze(1).repeat(1, gt_paf.size(1), 1, 1)
        heatmap_masks = ignore_mask.unsqueeze(1).repeat(1, gt_heatmap.size(1), 1, 1)

        for logit_paf_t, logit_heatmap_t in zip(logit_paf, logit_heatmap):
            pafs_loss_t = self.mean_square_error(logit_paf_t, gt_paf, paf_masks)
            heatmaps_loss_t = self.mean_square_error(logit_heatmap_t, gt_heatmap, heatmap_masks)

            total_loss = total_loss + pafs_loss_t + heatmaps_loss_t
            heatmaps_loss.append(heatmaps_loss_t)
            pafs_loss.append(pafs_loss_t)

        return total_loss, heatmaps_loss, pafs_loss


class BuildTrainNetwork(nn.Module):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def forward(self, input_data, gt_paf, gt_heatmap, mask):
        logit_pafs, logit_heatmap = self.network(input_data)
        loss, _, _ = self.criterion(logit_pafs, logit_heatmap, gt_paf, gt_heatmap, mask)
        return loss
