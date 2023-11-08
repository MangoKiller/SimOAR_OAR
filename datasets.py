import os
import os.path as osp
import random

import numpy as np
import torch
from torch_geometric.data import (Batch, Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import to_dense_adj, to_undirected


class BA3Motif(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/Wuyxin/ReFine/main/data/BA3/raw/BA-3motif.npy'

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super(BA3Motif, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['BA-3motif.npy']

    @property
    def processed_file_names(self):
        return ['train_data.pt', 'test_data.pt']

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        edge_index_list, label_list, ground_truth_list, role_id_list, pos_list = np.load(osp.join(self.raw_dir, self.raw_file_names[0]), allow_pickle=True)

        data_list = []
        for idx, (edge_index, label, ground_truth, role_id, pos) in enumerate(zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos_list)):
            z = torch.from_numpy(role_id)
            x = torch.ones(z.size(0), 1)
            y = torch.tensor(label, dtype=torch.long).unsqueeze(dim=0)

            edge_index = torch.from_numpy(edge_index).long()
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))
            edge_attr = torch.ones(edge_index.size(1), 1)

            pos = torch.stack([torch.from_numpy(pos[i]) for i in range(z.size(0))], dim=0)

            node_mask = z > 0
            ground_truth_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]

            data = Data(x=x, y=y, z=z,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=pos,
                        ground_truth_mask=ground_truth_mask,
                        name=f'BA-3motif-{idx}', idx=idx)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        random.shuffle(data_list)
        n_test = round(len(data_list) * 0.1)
        torch.save(self.collate(data_list[n_test:]), self.processed_paths[0])
        torch.save(self.collate(data_list[:n_test]), self.processed_paths[1])


class TR3Motif(InMemoryDataset):
    url = 'https://anonymous.4open.science/api/repo/DSE-24BC/file/data/TR3/raw/TR-3motif.npy'

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super(TR3Motif, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['TR-3motif.npy']

    @property
    def processed_file_names(self):
        return ['train_data.pt', 'test_data.pt']

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        edge_index_list, label_list, ground_truth_list, role_id_list, pos_list = np.load(osp.join(self.raw_dir, self.raw_file_names[0]), allow_pickle=True)

        data_list = []
        for idx, (edge_index, label, ground_truth, role_id, pos) in enumerate(zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos_list)):
            z = torch.from_numpy(role_id)
            x = torch.ones(z.size(0), 1)
            y = torch.tensor(label, dtype=torch.long).unsqueeze(dim=0)

            edge_index = torch.from_numpy(edge_index).long()
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))
            edge_attr = torch.ones(edge_index.size(1), 1)

            pos = torch.stack([torch.from_numpy(pos[i]) for i in range(z.size(0))], dim=0)

            node_mask = z > 0
            ground_truth_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]

            data = Data(x=x, y=y, z=z,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=pos,
                        ground_truth_mask=ground_truth_mask,
                        name=f'TR-3motif-{idx}', idx=idx)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        random.shuffle(data_list)
        n_test = round(len(data_list) * 0.1)
        torch.save(self.collate(data_list[n_test:]), self.processed_paths[0])
        torch.save(self.collate(data_list[:n_test]), self.processed_paths[1])


class Mutagenicity(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/flyingdoog/PGExplainer/master/dataset/Mutagenicity.zip'

    node_label_map = {
        0: 'C',
        1: 'O',
        2: 'Cl',
        3: 'H',
        4: 'N',
        5: 'F',
        6: 'Br',
        7: 'S',
        8: 'P',
        9: 'I',
        10: 'Na',
        11: 'K',
        12: 'Li',
        13: 'Ca'
    }

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super(Mutagenicity, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'Mutagenicity_A.txt',
            'Mutagenicity_edge_gt.txt',
            'Mutagenicity_edge_labels.txt',
            'Mutagenicity_graph_indicator.txt',
            'Mutagenicity_graph_labels.txt',
            'Mutagenicity_node_labels.txt',
        ]

    @property
    def processed_file_names(self):
        return ['train_data.pt', 'test_data.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        edge_index = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[0]), dtype=int, delimiter=',').T
        edge_index = torch.from_numpy(edge_index - 1)  # node idx from 0

        edge_ground_truth_mask = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[1]), dtype=bool)
        edge_ground_truth_mask = torch.from_numpy(edge_ground_truth_mask)

        edge_labels = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]), dtype=int)
        edge_labels = torch.from_numpy(edge_labels)
        edge_attr = torch.nn.functional.one_hot(edge_labels).float()

        graph_indicator = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[3]), dtype=int)

        graph_labels = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[4]), dtype=int)
        graph_labels = torch.from_numpy(graph_labels).unsqueeze(1)

        node_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[5]), dtype=int)
        node_label = torch.from_numpy(node_label)
        node_attr = torch.nn.functional.one_hot(node_label).float()

        num_graphs = graph_labels.size(0)
        total_edges = edge_index.size(1)
        begin = end = 0
        data_list = []
        for i in range(num_graphs):
            node_indices = np.where(graph_indicator == i + 1)[0]
            max_node_id = node_indices.max()
            while end < total_edges and edge_index[0, end] <= max_node_id:
                end += 1

            data = Data(
                x=node_attr[node_indices],
                y=graph_labels[i], 
                z=node_label[node_indices],
                edge_index=edge_index[:, begin:end] - node_indices.min(),
                edge_attr=edge_attr[begin:end],
                ground_truth_mask=edge_ground_truth_mask[begin:end],
                name="mutag_%d" % i,
                idx=i
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

            begin = end

        assert len(data_list) == num_graphs == 4337

        random.shuffle(data_list)
        n_test = round(len(data_list) * 0.1)
        torch.save(self.collate(data_list[n_test:]), self.processed_paths[0])
        torch.save(self.collate(data_list[:n_test]), self.processed_paths[1])


if __name__ == '__main__':
    # statistics of datasets
    DATASETS = {
        'BA3': BA3Motif,
        'TR3': TR3Motif,
        'MNIST': MNISTSuperpixels,
        'Mutagenicity': Mutagenicity,
    }
    for name, dataset in DATASETS.items():
        train_dataset = dataset(root='data/' + name, train=True)
        test_dataset = dataset(root='data/' + name, train=False)
        data = Batch.from_data_list([*train_dataset, *test_dataset])
        print(name.center(22, '='))
        print('- #(Graphs):', data.num_graphs)
        print('- #(Classes):', train_dataset.num_classes)
        print(f'- #(Avg. Nodes): {data.num_nodes / data.num_graphs:.2f}')
        num_edges = (data.edge_index[0] >= data.edge_index[1]).sum() if data.is_undirected() else data.num_edges
        print(f'- #(Avg. Edges): {num_edges / data.num_graphs:.2f}')
