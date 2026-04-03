
import torch

class PointDistance:
    """Mean Euclidean distance between predicted and target point sets.

    The training code uses this as a reporting metric when labels are defined in
    point space. Inputs are expected to contain XYZ coordinates in ``dim=2``.
    """

    def __init__(self,paired=True):
        self.paired = paired
    
    def __call__(self,preds,labels):
        if self.paired:
            # Keep one averaged distance per relative frame pair by reducing over
            # the batch axis and point index, but not the pair axis.
            return ((preds-labels)**2).sum(dim=2).sqrt().mean(dim=(0,2))
        else:
            # Reduce everything to a single global mean distance.
            return ((preds-labels)**2).sum(dim=2).sqrt().mean()
