import random
from typing import Iterable

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, MLP
from torch_geometric.utils import softmax

from explainers.base import Explainer


class CXModel(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int):
        super().__init__()
        in_channels = -1
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels * 2, hidden_channels, 1], norm=None, dropout=0.5)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        edge_embeds = torch.cat([x[data.edge_index[0]], x[data.edge_index[1]]], dim=-1)
        edge_scores = self.mlp(edge_embeds).view(-1)
        if data.batch is not None:
            edge_scores = softmax(edge_scores, index=data.batch[data.edge_index[0]])
        else:
            edge_scores = edge_scores.softmax(dim=0)
        return edge_scores


class CXPlain(Explainer):
    def __init__(self, cx_model: torch.nn.Module = CXModel(32, 2), epochs: int = 50, lr: float = 0.002, batch_size: int = 128, patience: int = 12):
        super().__init__()
        self.cx_model = cx_model
        self.optimizer = torch.optim.Adam(self.cx_model.parameters(), lr=lr)

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

    @torch.no_grad()
    def _get_granger_scores(self, model: torch.nn.Module, graph: Data):
        logits = model(graph)
        orig_loss = torch.nn.functional.cross_entropy(logits, graph.y)

        scores = []
        for e_id in range(graph.num_edges):
            edge_mask = graph.x.new_ones(graph.num_edges, dtype=torch.bool)
            edge_mask[e_id] = False
            masked_graph = graph.clone()
            masked_graph.edge_index = graph.edge_index[:, edge_mask]
            if graph.edge_attr is not None:
                masked_graph.edge_attr = graph.edge_attr[edge_mask]

            masked_logits = model(masked_graph)
            masked_loss = torch.nn.functional.cross_entropy(masked_logits, graph.y)
            scores.append((masked_loss - orig_loss).item())

        scores = torch.tensor(scores, device=graph.x.device)
        scores = self.post_process_mask(scores, normalize=True)

        return scores

    def train(self, model: torch.nn.Module, train_graphs: Iterable[Data], val_graphs: Iterable[Data] = None, verbose: bool = False) -> 'CXPlain':
        self.cx_model, model = self.cx_model.to(self.device), model.to(self.device)

        train_graphs, val_graphs = list(train_graphs), list(val_graphs) if val_graphs else None

        best_val_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            # train
            self.cx_model.train()
            random.shuffle(train_graphs)
            for i in range(0, len(train_graphs), self.batch_size):
                batch_graphs = train_graphs[i:i + self.batch_size]

                granger_scores = torch.cat([self._get_granger_scores(model, graph.to(self.device)) for graph in batch_graphs])
                estimated_scores = self.cx_model(Batch.from_data_list(batch_graphs).to(self.device))
                loss = torch.nn.functional.kl_div(estimated_scores, granger_scores, reduction='sum') / len(batch_graphs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if verbose:
                    print(f'\rEpoch: {epoch:03d}, Batch: {i // self.batch_size + 1:03d}, Loss: {loss.item():.4f}', end='')

            if val_graphs is None:
                continue

            # validate
            self.cx_model.eval()
            val_loss = 0
            for i in range(0, len(val_graphs), self.batch_size):
                batch_graphs = val_graphs[i:i + self.batch_size]

                granger_scores = torch.cat([self._get_granger_scores(model, graph.to(self.device)) for graph in batch_graphs])
                with torch.no_grad():
                    estimated_scores = self.cx_model(Batch.from_data_list(batch_graphs).to(self.device))
                val_loss += torch.nn.functional.kl_div(estimated_scores, granger_scores, reduction='sum').item()
            val_loss /= len(val_graphs)

            if verbose:
                print(f', Val Loss: {val_loss:.4f}', end='')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = self.patience
            else:
                patience -= 1
                if patience == 0:  # early stopping
                    break
        if verbose:
            print()
        return self

    @torch.no_grad()
    def explain_graph(self, model: torch.nn.Module, graph: Data):
        self.cx_model, graph = self.cx_model.to(self.device), graph.to(self.device)

        self.cx_model.eval()
        edge_mask = self.cx_model(graph)
        return edge_mask
