import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryWeightedDiceLoss(nn.Module):
    def __init__(self, kernel_size=3, boundary_weight=2.0):
        super(BoundaryWeightedDiceLoss, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.boundary_weight = boundary_weight
        self.smooth = 1e-5

    def extract_boundary(self, mask):
        dilation = F.max_pool3d(mask, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        erosion = 1 - F.max_pool3d(1 - mask, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        return dilation - erosion

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        boundary_map = self.extract_boundary(target.float())
        weight_map = torch.ones_like(target.float()) + (boundary_map * (self.boundary_weight - 1))
        intersection = torch.sum(pred * target * weight_map, dim=[2, 3, 4])
        cardinality = torch.sum((pred + target) * weight_map, dim=[2, 3, 4])
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()