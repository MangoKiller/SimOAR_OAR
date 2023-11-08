import copy
import math
import random
from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import MLP, ARMAConv, BatchNorm, Linear
from torch_geometric.utils import remove_isolated_nodes

from explainers.base import Explainer


class EdgeMaskNet(torch.nn.Module):
    def __init__(self, hidden_channels: int = 72, n_layers: int = 3):
        super().__init__()
        self.node_lin = Linear(-1, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(n_layers):
            conv = ARMAConv(in_channels=hidden_channels, out_channels=hidden_channels)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

        self.edge_lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.edge_lin2 = Linear(-1, hidden_channels)

        self.mlp = MLP([2 * hidden_channels, hidden_channels, 1], act='tanh', norm=None)
        self._initialize_weights()

    def forward(self, data: Data) -> torch.Tensor:
        x = self.node_lin(data.x).relu()
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, data.edge_index).relu()
            x = batch_norm(x)

        e = torch.cat([x[data.edge_index[0]], x[data.edge_index[1]]], dim=1)

        if data.edge_attr is not None and data.edge_attr.size(-1) > 1:
            e1 = self.edge_lin1(e)
            e2 = self.edge_lin2(data.edge_attr)
            e = torch.cat([e1, e2], dim=1)  # connection

        return self.mlp(e).view(-1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)


class ReFine(Explainer):
    coeffs = {
        'edge_size': 1e-4,
        'edge_ent': 1e-2,
        'gamma': 5.0,
        'EPS': 1e-6
    }

    def __init__(self, n_label: int, hidden_channels: int = 50, n_layers: int = 2, epochs: int = 50, lr: float = 1e-3, batch_size: int = 256, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.coeffs.update(kwargs)

        self.edge_mask_net = torch.nn.ModuleList([EdgeMaskNet(hidden_channels, n_layers) for _ in range(n_label)])
        self.optimizer = torch.optim.Adam(self.edge_mask_net.parameters(), lr=self.lr)

    def _reparameterize(self, log_alpha, temperature: float = 1.0):
        eps = torch.rand_like(log_alpha)
        return (eps.log() - (1 - eps).log() + log_alpha) / temperature

    def _fidelity_loss(self, y_hat: torch.Tensor, y: torch.Tensor, edge_mask: torch.Tensor):
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        mask = edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * mask.mean()
        ent = -mask * torch.log(mask + self.coeffs['EPS']) - (1 - mask) * torch.log(1 - mask + self.coeffs['EPS'])
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        return loss

    def _contrastive_loss(self, embeds, classes):
        embeds = embeds / embeds.norm(dim=1, keepdim=True)
        sim_mat = torch.mm(embeds, embeds.T).relu()
        hit_mat = classes[:, None] == classes[None, :]
        loss = ((-1)**hit_mat * torch.nn.functional.softplus(sim_mat)).mean()
        return loss

    def pretrain(self, model: torch.nn.Module, graphs: Iterable[Data], verbose: bool = False):
        self.edge_mask_net, model = self.edge_mask_net.to(self.device), model.to(self.device)

        graphs = list(graphs)
        for epoch in range(1, self.epochs + 1):
            random.shuffle(graphs)
            for i in range(0, len(graphs), self.batch_size):
                batch_graphs = graphs[i:i + self.batch_size]
                batch: Data = Batch.from_data_list(batch_graphs).to(self.device)

                orig_logits = model(batch)
                pred_classes = orig_logits.argmax(-1)

                edge_mask = torch.cat([self.edge_mask_net[c](graph.to(self.device)) for c, graph in zip(pred_classes, batch_graphs)])
                edge_mask = self._reparameterize(edge_mask)

                set_masks(model, edge_mask, batch.edge_index, apply_sigmoid=True)
                embeds, logits = model(batch, return_embeds='graph')
                clear_masks(model)

                fid_loss = self._fidelity_loss(logits, orig_logits, edge_mask)
                cts_loss = self._contrastive_loss(embeds, pred_classes)

                loss =  fid_loss + self.coeffs['gamma'] * cts_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if verbose:
                    print(f'\rEpoch: {epoch}, Batch: {i // self.batch_size + 1}, Loss: {loss.item():.4f}, Fidelity: {fid_loss.item():.4f}, Contrastive: {cts_loss.item():.4f}', end='')
        if verbose:
            print()
        return self

    def explain_graph(self, model: torch.nn.Module, graph: Data, fine_tune: bool = True, epoch: int = 50, lr: float = 1e-4, ratio: float = 1.0) -> torch.Tensor:
        self.edge_mask_net, model, graph = self.edge_mask_net.to(self.device), model.to(self.device), graph.to(self.device)

        with torch.no_grad():
            orig_logit = model(graph)
        pred_class = orig_logit.argmax(-1).item()
        mask_net = self.edge_mask_net[pred_class]

        if fine_tune:
            mask_net = copy.deepcopy(mask_net)
            optimizer = torch.optim.Adam(mask_net.parameters(), lr=lr)
            for _ in range(epoch):
                edge_mask = mask_net(graph)
                edge_mask = self._reparameterize(edge_mask)

                topk = max(math.ceil(ratio * graph.num_edges), 1)
                pos_indices = edge_mask.argsort(descending=True)[:topk]
                sub_edge_mask = edge_mask[pos_indices]

                subgraph = graph.clone()
                edge_index, edge_attr, node_mask = remove_isolated_nodes(graph.edge_index[:, pos_indices], graph.edge_attr[pos_indices], graph.num_nodes)
                subgraph.edge_index = edge_index
                subgraph.edge_attr = edge_attr
                subgraph.x = graph.x[node_mask]
                if graph.pos is not None:
                    subgraph.pos = graph.pos[node_mask]

                set_masks(model, sub_edge_mask, subgraph.edge_index, apply_sigmoid=True)
                logits = model(subgraph)
                clear_masks(model)

                fid_loss = self._fidelity_loss(logits, orig_logit, sub_edge_mask)

                optimizer.zero_grad()
                fid_loss.backward()
                optimizer.step()

        edge_mask = mask_net(graph)
        edge_mask = self.post_process_mask(edge_mask, apply_sigmoid=True)
        return edge_mask
