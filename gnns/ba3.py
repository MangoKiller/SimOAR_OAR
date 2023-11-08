# base on https://github.com/pyg-team/pytorch_geometric/blob/89a54d9454d3832f814f9a574ed421c58f1fce10/examples/mutag_gin.py

import argparse
import copy
import os
import os.path as osp
import sys
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GINConv, global_add_pool

sys.path.append(osp.join(osp.dirname(osp.realpath(__file__)), '..'))
from datasets import BA3Motif


class Net(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, data: Data, return_embeds: Optional[Literal['node', 'graph']] = None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        if return_embeds == 'node':
            embeds = x
        x = global_add_pool(x, batch)
        if return_embeds == 'graph':
            embeds = x
        logits = self.mlp(x)
        if return_embeds:
            return embeds, logits
        return logits


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader: DataLoader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs='?', default=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'BA3'), help='data path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'models', 'gnns') , help='path for saving trained model.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default= 0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--hidden_channels', type=int, default=32, help='size of hidden channels')
    parser.add_argument('--num_layers', type=int, default=2, help='number of Convolution layers')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use if any (default: 0)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_dataset = BA3Motif(root=args.data_path, train=True)
    test_dataset = BA3Motif(root=args.data_path, train=False)

    train_dataset = train_dataset.shuffle()
    train_dataset, val_dataset = train_dataset[len(test_dataset):], train_dataset[:len(test_dataset)]

    print(f'Number of graphs: #train = {len(train_dataset)}, #val = {len(val_dataset)}, #test = {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = Net(train_dataset.num_features, args.hidden_channels, train_dataset.num_classes, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_epoch, best_val_acc, best_weights = -1, 0, None
    for epoch in range(1, args.epochs + 1):
        loss = train()
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_epoch = epoch
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    test_acc = test(test_loader)
    print(f'Best Epoch(w.r.t. Val ACC): {best_epoch}, Best Val ACC: {best_val_acc:.4f}, Test ACC: {test_acc:.4f}')

    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.cpu(), osp.join(args.model_path, 'ba3.pt'))
