from math import sqrt

import torch
from torch_geometric.data import Data
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from .base import Explainer


class GNNExplainer(Explainer):
    """
    GNNExplainer for interpreting graph prediction results from GNNs.

    Citation:
    Based on `torch_geometric.nn.models.GNNExplainer` which generates explainations in node prediction tasks.

    Original Paper: `Ying et al. GNNExplainer: Generating Explanations for Graph Neural Networks`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_ent': 1.0,
        'EPS': 1e-15
    }

    def __init__(self, epochs: int = 200, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

    def _loss(self, y_hat: torch.Tensor, y: torch.Tensor, edge_mask: torch.Tensor):
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        m = edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        return loss

    def explain_graph(self, model: torch.nn.Module, graph: Data):
        model, graph = model.to(self.device), graph.to(self.device)

        N, E= graph.x.size(0), graph.edge_index.size(1)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        edge_mask = (torch.randn(E) * std).to(self.device)

        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)

        set_masks(model, edge_mask, graph.edge_index)
        for _ in range(1, self.epochs + 1):
            logits = model(graph)
            loss = self._loss(logits, graph.y, edge_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        clear_masks(model)

        edge_mask = self.post_process_mask(edge_mask, apply_sigmoid=True)
        return edge_mask
