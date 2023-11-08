import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.transforms import Cartesian
from torch_geometric.utils import batched_negative_sampling
from tqdm import tqdm

from datasets import BA3Motif


# from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class OODBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.vgae = VGAE(VariationalGCNEncoder(in_channels, out_channels))

    def train(self, dataloader, optimizer, scheduler, device):
        self.vgae.train()
        total_loss = 0.
        for data in tqdm(dataloader, desc='Train'):
            data = data.to(device)

            z = self.vgae.encode(data.x, data.edge_index)
            neg_edge_index = batched_negative_sampling(data.edge_index, data.batch)
            recon_loss = self.vgae.recon_loss(z, data.edge_index, neg_edge_index)
            kl_loss = self.vgae.kl_loss()
            loss = recon_loss + (1 / data.num_nodes) * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader, device):
        self.vgae.eval()
        total_loss, total_auc, total_ap = 0., 0., 0.
        for data in dataloader:
            data = data.to(device)

            z = self.vgae.encode(data.x, data.edge_index)
            neg_edge_index = batched_negative_sampling(data.edge_index, data.batch)
            recon_loss = self.vgae.recon_loss(z, data.edge_index, neg_edge_index)
            kl_loss = self.vgae.kl_loss()
            loss = recon_loss + (1 / data.num_nodes) * kl_loss

            auc, ap = self.vgae.test(z, data.edge_index, neg_edge_index)

            total_loss += loss.item()
            total_auc += auc
            total_ap += ap
        return total_loss / len(dataloader), total_auc / len(dataloader), total_ap / len(dataloader)


def get_dataset(dataset):
    if dataset == 'mnist':
        root = 'data/MNIST'
        transform = Cartesian(cat=False, max_value=9)
        train_dataset = MNISTSuperpixels(root, train=True, transform=transform)
        test_dataset = MNISTSuperpixels(root, train=False, transform=transform)
        train_dataset, val_dataset = train_dataset[:50000], train_dataset[50000:]
    elif dataset == "ba3":
        root = 'data/BA3'
        train_dataset = BA3Motif(root, train=True)
        test_dataset = BA3Motif(root, train=False)
        train_dataset, val_dataset = train_dataset[400:], train_dataset[:400]
    return train_dataset, val_dataset, test_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="VGAE Training.")
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='param/ood/',
                        help='path to save model.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='One of [ba3, mnist, graphsst2]')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Device NO. of cuda.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--out_channels', type=int, default=128,
                        help='Out channels for CG.')
    parser.add_argument('--test', action='store_true',
                        help='Test only.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset, test_dataset = get_dataset(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ood_evaluator = OODBlock(train_dataset[0].num_node_features, args.out_channels).to(device)

    if not args.test:
        optimizer = Adam(ood_evaluator.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=1.0)
        for epoch in range(args.n_epochs):
            train_loss = ood_evaluator.train(train_loader, optimizer, scheduler, device=device)
            val_loss, _, _ = ood_evaluator.evaluate(val_loader, device=device)
            _, test_auc, test_ap = ood_evaluator.evaluate(test_loader, device=device)

            print('Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Test AUC: {:.4f} | AP: {:.4f}'.format(epoch, train_loss, val_loss, test_auc, test_ap))

        torch.save(ood_evaluator, args.model_path + f'{args.dataset}.pth')
    else:
        ood_evaluator = torch.load(args.model_path + f'{args.dataset}.pth', map_location=device)
        _, test_auc, test_ap = ood_evaluator.evaluate(test_loader, device=device)
        print('Test AUC: {:.4f} | AP: {:.4f}'.format(test_auc, test_ap))
