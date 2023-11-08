from typing import Iterable

import torch
from torch_geometric.data import Data
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import MLP

from explainers.base import Explainer


class PGExplainer(Explainer):
    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0.0,
    }

    def __init__(self, epochs: int = 30, lr: float = 0.003, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.mlp = MLP([-1, 64, 1], act='relu', norm=None)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss(self, y_hat: torch.Tensor, y: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        # Regularization loss:
        mask = edge_mask.sigmoid()
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']
        return loss + size_loss + mask_ent_loss

    def train(self, model: torch.nn.Module, graphs: Iterable[Data], verbose: bool = False):
        self.mlp, model = self.mlp.to(self.device), model.to(self.device)

        for epoch in range(1, self.epochs + 1):
            for i, graph in enumerate(graphs):
                graph = graph.to(self.device)

                z, orig_logits = model(graph, return_embeds='node')
                target = orig_logits.softmax(dim=-1)

                temperature = self._get_temperature(epoch)

                inputs = torch.cat([z[graph.edge_index[0]], z[graph.edge_index[1]]], dim=-1)
                logits = self.mlp(inputs).view(-1)
                edge_mask = self._concrete_sample(logits, temperature)

                set_masks(model, edge_mask, graph.edge_index, apply_sigmoid=True)
                logits = model(graph)
                loss = self._loss(logits, target, edge_mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                clear_masks(model)

                if verbose:
                    print(f'\rEpoch {epoch}/{self.epochs} | Graph {i + 1}/{len(graphs)}: loss = {loss.item():.4f}', end='')
        if verbose:
            print()
        return self

    @torch.no_grad()
    def explain_graph(self, model: torch.nn.Module, graph: Data) -> torch.Tensor:
        self.mlp, model, graph = self.mlp.to(self.device), model.to(self.device), graph.to(self.device)

        z, _ = model(graph, return_embeds='node')

        inputs = torch.cat([z[graph.edge_index[0]], z[graph.edge_index[1]]], dim=-1)
        logits = self.mlp(inputs).view(-1)

        edge_mask = self.post_process_mask(logits, apply_sigmoid=True)
        return edge_mask
