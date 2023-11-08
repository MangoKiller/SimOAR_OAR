# based on https://github.com/pyg-team/pytorch_geometric/blob/89a54d9454d3832f814f9a574ed421c58f1fce10/examples/mnist_voxel_grid.py

import argparse
import copy
import os
import os.path as osp
from typing import Literal, Optional

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SplineConv, max_pool, max_pool_x, voxel_grid

transform = T.Cartesian(cat=False)


class Net(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.conv1 = SplineConv(num_features, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(4 * 64, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, data: Data, return_embeds: Optional[Literal['node', 'graph']] = None):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        cluster = voxel_grid(data.pos, batch=data.batch, size=5, start=0, end=28)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        cluster = voxel_grid(data.pos, batch=data.batch, size=7, start=0, end=28)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        if return_embeds == 'node':
            embeds = data.x

        cluster = voxel_grid(data.pos, batch=data.batch, size=14, start=0, end=27.99)
        x, _ = max_pool_x(cluster, data.x, data.batch, size=4)
        x = x.view(-1, self.fc1.weight.size(1))

        if return_embeds == 'graph':
            embeds = x

        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        logits = F.log_softmax(x, dim=1)

        if return_embeds:
            return embeds, logits
        return logits


def train(epoch: int):
    model.train()

    if epoch == 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader: DataLoader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MNIST Spuer-Pixel Model")
    parser.add_argument('--data_path', nargs='?', default=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST'), help='data path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'models', 'gnns') , help='path for saving trained model.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default= 0.01, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use if any (default: 0)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_dataset = MNISTSuperpixels(args.data_path, train=True, transform=transform)
    test_dataset = MNISTSuperpixels(args.data_path, train=False, transform=transform)

    test_size = 10000
    test_dataset = test_dataset.shuffle()[:test_size]
    train_dataset = train_dataset.shuffle()[:9*test_size]
    train_dataset, val_dataset = train_dataset[len(test_dataset):], train_dataset[:len(test_dataset)]

    print(f'Number of graphs: #train = {len(train_dataset)}, #val = {len(val_dataset)}, #test = {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = Net(train_dataset.num_features, train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_epoch, best_val_acc, best_weights = -1, 0, None
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
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
    torch.save(model.cpu(), osp.join(args.model_path, 'mnist.pt'))
