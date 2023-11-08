import torch
from torch_geometric.data import Data

from explainers.base import Explainer


class SAExplainer(Explainer):
    def __init__(self):
        super().__init__()

    def explain_graph(self, model: torch.nn.Module, graph: Data):
        model, graph = model.to(self.device), graph.to(self.device)

        graph = graph.clone()
        graph.x.requires_grad_()
        logits = model(graph)

        node_score = (torch.autograd.grad(logits[0, graph.y.item()], graph.x)[0]**2).sum(dim=1)
        node_mask = self.post_process_mask(node_score, scale=True)

        edge_mask = node_mask[graph.edge_index[0]] * node_mask[graph.edge_index[1]]

        return edge_mask
