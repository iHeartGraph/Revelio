import os
import torch
import numpy as np
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid, Amazon, GitHub, FacebookPagePage, LastFMAsia, \
    DeezerEurope, WikiCS, Flickr, Twitch, Coauthor
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.utils import add_self_loops, remove_self_loops

from data import MoleculeDataset, SynGraphDataset

rng = np.random.default_rng(2024)


def statistics(dataset, self_loop: bool):
    data = dataset.data
    N = data.num_nodes
    E = data.num_edges if not self_loop else data.num_edges - N
    X = data.num_node_features
    C = dataset.num_classes

    print('Dataset:', dataset.name)
    print('# graphs:', len(dataset))
    print('# nodes:', N, N / len(dataset))
    print('# edges:', E, E / len(dataset))
    print('# features:', X)
    print('# classes', C)
    print()


def get_nc_dataset(dataset_dir: str, dataset_name: str, self_loops: bool = True):
    """load dataset"""
    if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=dataset_dir, name=dataset_name)
    elif dataset_name.lower() in ['computers', 'photo']:
        dataset = Amazon(root=dataset_dir, name=dataset_name)
    elif dataset_name.lower() == 'github':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = GitHub(root=dataset_dir)
    elif dataset_name.lower() == 'facebook':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = FacebookPagePage(root=dataset_dir)
    elif dataset_name.lower() == 'lastfmasia':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = LastFMAsia(root=dataset_dir)
    elif dataset_name.lower() == 'deezereurope':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = DeezerEurope(root=dataset_dir)
    elif dataset_name.lower() == 'wikics':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = WikiCS(root=dataset_dir, is_undirected=True)
    elif dataset_name.lower() == 'flicker':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = Flickr(root=dataset_dir)
    elif dataset_name.lower() in ['de', 'en', 'es', 'fr', 'pt', 'ru']:
        dataset = Twitch(root=dataset_dir, name=dataset_name.upper())
    elif dataset_name.lower() in ['cs', 'physics']:
        dataset = Coauthor(root=dataset_dir, name=dataset_name)
    elif dataset_name.lower() == 'house':
        file_path = os.path.join(dataset_dir, dataset_name + '.pt')
        if not os.path.exists(file_path):
            dataset = ExplainerDataset(
                graph_generator=BAGraph(num_nodes=300, num_edges=5),
                motif_generator='house',
                num_motifs=80,
                transform=T.Constant(),
            )
            torch.save(dataset, file_path)
        else:
            dataset = torch.load(file_path)
    elif dataset_name.lower() in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle']:
        dataset = SynGraphDataset(root=dataset_dir, name=dataset_name)
    else:
        raise NotImplementedError

    data = dataset.data
    '''transform data'''
    data.edge_index, _ = remove_self_loops(data.edge_index)
    if self_loops:
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    '''split data'''
    num_train = int(data.num_nodes * 0.6)
    num_val = int(data.num_nodes * 0.2)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    perm_index = torch.randperm(data.num_nodes, generator=torch.random.manual_seed(0))
    data.train_mask[perm_index[:num_train]] = True
    data.val_mask[perm_index[num_train:num_train + num_val]] = True
    data.test_mask[perm_index[num_train + num_val:]] = True
    dataset.data = data

    statistics(dataset, self_loops)

    return dataset


def get_gc_dataset(dataset_dir: str, dataset_name: str, self_loops: bool = True):
    """load dataset"""
    if dataset_name.lower() == 'ba_2motifs':
        dataset = SynGraphDataset(root=dataset_dir, name=dataset_name)
    else:
        dataset = MoleculeDataset(root=dataset_dir, name=dataset_name)
    # else:
    #     raise NotImplementedError

    data = dataset.data
    '''transform data'''
    data.x = data.x.float()
    data.y = data.y.long()
    edge_index, _ = remove_self_loops(data.edge_index)
    if self_loops:
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    data.edge_index = edge_index
    dataset.data = data

    statistics(dataset, self_loops)

    return dataset
