from typing import Union

import torch
from torch_geometric.data import Batch

from .graph_utils import reduce_graph, repeat_interleave_as, reduce_interleave_as


@torch.no_grad()
def score_cross_entropy(
    subgraphs: Batch, graphs: Batch, gnn: torch.nn.Module
) -> torch.Tensor:
    """Assume `subgraph` is a non-reduced subgraph of `graph`."""
    full_preds = gnn(graphs.to(device))
    sub_preds = gnn(reduce_graph(subgraphs).to(device))
    full_probs = torch.softmax(full_preds, dim=-1)
    full_probs = repeat_interleave_as(full_probs, sub_preds)
    scores = -torch.nn.functional.cross_entropy(sub_preds, full_probs, reduction="none")
    scores = reduce_interleave_as(scores, graphs.y)
    return scores


@torch.no_grad()
def score_confidence(
    subgraphs: Batch, graphs: Batch, gnn: torch.nn.Module
) -> torch.Tensor:
    """Assume `subgraph` is a non-reduced subgraph of `graph`."""
    sub_preds = gnn(reduce_graph(subgraphs).to(device))
    sub_probs = torch.softmax(sub_preds, dim=-1)
    y = repeat_interleave_as(graphs.y.to(device), sub_probs)
    scores = sub_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
    scores = reduce_interleave_as(scores, graphs.y)
    return scores


@torch.no_grad()
def score_random_deletion(
    subgraphs: Batch,
    graphs: Batch,
    gnn: torch.nn.Module,
    n_del: Union[int, float] = 1,
    n_samples: int = 1,
    on_edges: bool = True,
) -> torch.Tensor:
    aug_subgraphs = []
    for subgraph, graph in zip(subgraphs.to_data_list(), graphs.to_data_list()):
        aug_subgraphs.extend(
            random_deletion(graph, subgraph, n_del, n_samples, on_edges)
        )
    aug_subgraphs = Batch.from_data_list(aug_subgraphs)
    scores = score_confidence(aug_subgraphs, graphs, gnn)
    return scores
