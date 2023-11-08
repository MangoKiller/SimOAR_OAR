from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Data

EPS = 1e-6


class Explainer(ABC):
    device = 'cpu'

    @abstractmethod
    def explain_graph(self, graph: Data, **kwargs):
        """
        Main part for different graph attribution methods
        :param graph: target graph instance to be explained
        :param kwargs:
        :return: edge_mask, i.e., attributions for edges, which are derived from the attribution methods.
        """
        ...

    def to(self, device: str):
        self.device = device
        return self

    @staticmethod
    def post_process_mask(mask: torch.Tensor, hard_mask: torch.Tensor = None, apply_sigmoid: bool = False, scale: bool = False, normalize: bool = False) -> torch.Tensor:
        """Post processes any mask to not include any attributions of elements not involved during message passing."""
        mask = mask.detach()

        if apply_sigmoid:
            mask = mask.sigmoid()
        elif scale:
            mask = mask / (mask.max() + EPS)
        elif normalize:
            mask[mask < 0] = 0
            mask += EPS
            mask = mask / mask.sum()

        if hard_mask is not None and mask.size(0) == hard_mask.size(0):
            mask[~hard_mask] = 0.

        return mask

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
