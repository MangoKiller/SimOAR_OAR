from typing import List, Callable, Optional
from functools import wraps

import torch
from torch_geometric.data import Data, Batch


def allow_batch(func: Callable) -> Callable:
    """Decorator to allow `func` to accept batched input."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], Batch):
            return Batch.from_data_list(
                [func(arg, *args[1:], **kwargs) for arg in args[0].to_data_list()]
            )
        else:
            return func(*args, **kwargs)

    return wrapper


@allow_batch
def get_oracle_rationale(graph: Data) -> Data:
    """Given original graph `graph`, return its non-reduced oracle rationale."""
    rationale = graph.clone()
    rationale.edge_index = graph.edge_index[:, graph.ground_truth_mask]
    rationale.edge_attr = graph.edge_attr[graph.ground_truth_mask]
    return rationale


@allow_batch
def reduce_graph(graph: Data, inplace: bool = False) -> Data:
    """return a new graph with isolated nodes removed."""
    if not inplace:
        graph = graph.clone()
    edge_index, edge_attr, node_mask = remove_isolated_nodes(
        graph.edge_index, graph.edge_attr, graph.num_nodes
    )
    graph.edge_index = edge_index
    graph.edge_attr = edge_attr
    graph.x = graph.x[node_mask]
    if graph.pos is not None:
        graph.pos = graph.pos[node_mask]
    return graph


def repeat_interleave_as(
    obj: torch.Tensor, ref: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """Repeat each element of `obj` as many times as the corresponding element in `ref` along dimension `dim`."""
    if obj.size(dim) < ref.size(dim):
        assert ref.size(dim) % obj.size(dim) == 0
        obj = obj.repeat_interleave(ref.size(dim) // obj.size(dim), dim=dim)
    return obj


def reduce_interleave_as(
    obj: torch.Tensor, ref: torch.Tensor, dim: int = 0, reduction="mean"
) -> torch.Tensor:
    if obj.size(dim) > ref.size(dim):
        assert obj.size(dim) % ref.size(dim) == 0
        obj_shape = list(obj.shape)
        obj_shape[dim] = ref.size(dim)
        obj_shape.insert(dim + 1, -1)
        obj = getattr(obj.view(*obj_shape), reduction)(dim=dim + 1)
    return obj


def as_undirected(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    return_mask: bool = False,
):
    """remove duplicate undirected edges in `edge_index`, i.e., edge (i, j) will be removed if (j, i) is present."""
    mask = edge_index[0] <= edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    if return_mask:
        return edge_index, edge_attr, mask
    return edge_index, edge_attr
