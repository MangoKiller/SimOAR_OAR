from typing import List, Union

import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    index_to_mask,
    is_undirected,
    subgraph,
    to_undirected,
)

from utils import as_undirected


def sample_subgraphs(
    graph: Data,
    n_total: Union[int, float] = 1.0,
    n_pos: Union[int, float] = 0.5,
    n_samples: int = 10,
    on_edges: bool = True,
    force_directed: bool = False,
    mode: str = "random",
) -> List[Data]:
    """Randomly sample `n_samples` subgraphs with a total of `n_total` edges(or nodes) and `n_pos` positive edges(or nodes).

    Args:
        graph (Data): The original graph.
        n_total (Union[int, float]): The total number of edges(or nodes) in the sampled subgraphs. If float, it is the ratio to the total
            number of positive edges(or nodes) in the original graph.
        n_pos (Union[int, float]): The number of positive edges(or nodes) in the sampled subgraphs.
            If float, it is the ratio to the total number of positive edges(or nodes) in the original graph.
        n_samples (int): The number of sampled subgraphs.
        on_edges (bool): Whether to sample on edges or nodes, default to True, i.e. sample on edges.
        force_directed (bool): Whether to force the input graph to be directed. Default to False, i.e. determined by `graph.is_directed()`.
        mode (str): The sampling mode, can be 'random' or 'connect'. In 'connect' mode, the result subgraph will always be connected. Default to 'random'.
    """
    edge_index, edge_attr = graph.edge_index, graph.edge_attr
    ground_truth_mask = (
        graph.ground_truth_mask
        if hasattr(graph, "ground_truth_mask")
        else edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
    )

    device = edge_index.device

    directed = not is_undirected(edge_index, edge_attr) or force_directed
    if not directed:
        edge_index, edge_attr, mask = as_undirected(
            edge_index, edge_attr, return_mask=True
        )
        ground_truth_mask = ground_truth_mask[mask]

    pos_mask = ground_truth_mask

    if on_edges:
        total = edge_index.size(1)
    else:
        pos_mask = index_to_mask(edge_index[:, pos_mask].unique(), size=graph.num_nodes)
        total = graph.num_nodes

    total_pos = pos_mask.sum().item()
    total_neg = total - total_pos

    if isinstance(n_total, float):
        n_total = round(n_total * total_pos)
    if isinstance(n_pos, float):
        n_pos = round(n_pos * total_pos)
    n_neg = n_total - n_pos
    if n_pos > total_pos or n_neg > total_neg or n_total < n_pos:
        return None

    if mode == "random":
        sample_mask = torch.zeros(n_samples, total, dtype=torch.bool, device=device)
        pos_sample_mask = torch.zeros(
            n_samples, total_pos, dtype=torch.bool, device=device
        )
        rand_perm = torch.randn(n_samples, total_pos, device=device).argsort(dim=1)
        pos_sample_mask.scatter_(1, rand_perm[:, :n_pos], True)
        sample_mask[:, pos_mask] = pos_sample_mask

        neg_sample_mask = torch.zeros(
            n_samples, total_neg, dtype=torch.bool, device=device
        )
        rand_perm = torch.randn(n_samples, total_neg, device=device).argsort(dim=1)
        neg_sample_mask.scatter_(1, rand_perm[:, :n_neg], True)
        sample_mask[:, ~pos_mask] = neg_sample_mask

    elif mode == "connect":
        sample_mask_list = []
        while len(sample_mask_list) < n_samples:
            sample_mask = torch.zeros(total, dtype=torch.bool, device=device)
            connected_mask = torch.zeros(
                total, dtype=torch.bool, device=device
            )  # indicate whether an edge(or a node) is connected to selected edges(or nodes)
            while (n_sampled := sample_mask.sum()) < n_total:
                available_mask = (
                    pos_mask.clone() if n_sampled < n_pos else (~pos_mask).clone()
                )
                available_mask &= ~sample_mask
                if n_sampled > 0:
                    available_mask &= connected_mask
                if not available_mask.any():
                    break
                available_indices = available_mask.nonzero().view(-1)
                new_index = available_indices[
                    torch.randint(0, len(available_indices), ())
                ]
                sample_mask[new_index] = True
                if on_edges:
                    new_edge = edge_index[:, new_index]
                    connected_mask |= (
                        (edge_index[0] == new_edge[0])
                        | (edge_index[1] == new_edge[0])
                        | (edge_index[0] == new_edge[1])
                        | (edge_index[1] == new_edge[1])
                    )
                else:
                    connected_mask[
                        edge_index[
                            :,
                            (edge_index[0] == new_index) | (edge_index[1] == new_index),
                        ].unique()
                    ] = True

            if n_sampled < n_total:
                continue

            sample_mask_list.append(sample_mask)
        sample_mask = torch.stack(sample_mask_list, dim=0)

    if not on_edges:
        sample_mask = sample_mask[:, edge_index[0]] & sample_mask[:, edge_index[1]]

    subgraphs = []
    for i in range(n_samples):
        subgraph = graph.clone()
        delattr(subgraph, "ground_truth_mask")
        subgraph.edge_index = edge_index[:, sample_mask[i]]
        subgraph.edge_attr = edge_attr[sample_mask[i]]
        if not directed:
            subgraph.edge_index, subgraph.edge_attr = to_undirected(
                subgraph.edge_index, subgraph.edge_attr, num_nodes=subgraph.num_nodes
            )
        subgraphs.append(subgraph)
    return subgraphs


def random_deletion(
    graph: Data,
    subgraph: Data,
    n_del: Union[int, float],
    n_samples: int = 1,
    on_edges: bool = True,
    force_directed: bool = False,
) -> List[Data]:
    """Random delete edges or nodes from graph `graph` while fixing subgraph `subgraph`."""
    edge_index, edge_attr = graph.edge_index, graph.edge_attr

    device = edge_index.device

    directed = is_undirected(edge_index, edge_attr) or force_directed
    if not directed:
        edge_index, edge_attr = as_undirected(edge_index, edge_attr)

    total = edge_index.size(1) if on_edges else graph.num_nodes
    if on_edges:
        neg_mask = (
            (edge_index[:, :, None] != subgraph.edge_index[:, None, :]).any(0).all(-1)
        )
    else:
        neg_mask = ~index_to_mask(subgraph.edge_index.unique(), size=graph.num_nodes)
    total_neg = neg_mask.sum().item()
    if isinstance(n_del, float):
        n_del = round(n_del * total)

    sample_mask = torch.ones(n_samples, total, dtype=torch.bool, device=device)
    neg_sample_mask = torch.ones(n_samples, total_neg, dtype=torch.bool, device=device)
    rand_perm = torch.randn(n_samples, total_neg, device=device).argsort(dim=1)
    neg_sample_mask.scatter_(1, rand_perm[:, :n_del], False)
    sample_mask[:, neg_mask] = neg_sample_mask

    if not on_edges:
        sample_mask = sample_mask[:, edge_index[0]] & sample_mask[:, edge_index[1]]

    subgraphs = []
    for i in range(n_samples):
        subgraph = graph.clone()
        delattr(subgraph, "ground_truth_mask")
        subgraph.edge_index = edge_index[:, sample_mask[i]]
        subgraph.edge_attr = edge_attr[sample_mask[i]]
        if not directed:
            subgraph.edge_index, subgraph.edge_attr = to_undirected(
                subgraph.edge_index, subgraph.edge_attr, num_nodes=subgraph.num_nodes
            )
        subgraphs.append(subgraph)
    return subgraphs
