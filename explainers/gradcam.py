import torch
from torch_geometric.data import Data

from explainers.base import Explainer


class GradCAM(Explainer):
    def __init__(self):
        super().__init__()

    def explain_graph(self, model: torch.nn.Module, graph: Data):
        model, graph = model.to(self.device), graph.to(self.device)

        graph = graph.clone()
        embeds, logits = model(graph, return_embeds='node')

        alpha = torch.mean(torch.autograd.grad(logits[0, graph.y.item()], embeds)[0], dim=0)
        node_score = (embeds @ alpha).relu()
        node_mask = self.post_process_mask(node_score, scale=True)

        edge_mask = node_mask[graph.edge_index[0]] * node_mask[graph.edge_index[1]]

        return edge_mask
