import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from deeprobust.graph.data import Dataset, PrePtbDataset
import os.path as osp

from utils import to_mask


def load_data(dataset_str: str, root='./data/', atk_name='clean', atk_rate=0):
    dataset_str = dataset_str.lower()
    atk_name = atk_name.lower()

    assert atk_name in ['clean', 'mettack', 'minmax'], ValueError(f"{atk_name}({atk_rate}) is not supported")
    if atk_name == 'clean':
        atk_rate = 0
    elif atk_name == 'mettack':
        assert atk_rate in [0.05, 0.1, 0.15, 0.2, 0.25], ValueError(f"`atk_rate` of mettack is chosen from [0.05, 0.1, 0.15, 0.2, 0.25]")
    elif atk_name == 'minmax':
        assert atk_rate in [0.05, 0.15, 0.25], ValueError(f"`atk_rate` of minmax is chosen from [0.05, 0.15, 0.25]")

    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        data = Dataset(root=root, name=dataset_str, setting='prognn')
        data.features = data.features.toarray()
        if atk_name == 'mettack':
            pert_data = PrePtbDataset(root=root, name=dataset_str, attack_method='meta', ptb_rate=atk_rate)
            data.adj = pert_data.adj
        elif atk_name == 'minmax':
            pert_data = load_minmax_data(root, dataset_str, atk_rate)
            data.adj = pert_data['adj']
            data.idx_train = pert_data['idx_train']
            data.idx_val = pert_data['idx_val']
            data.idx_test = pert_data['idx_test']
        train_mask, val_mask, test_mask = to_mask(data.idx_train, data.idx_val, data.idx_test,
                                                  num=data.features.shape[0])
        data = Data(
            x=torch.Tensor(data.features),
            edge_index=from_scipy_sparse_matrix(data.adj)[0],
            y=torch.LongTensor(data.labels),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
    else:
        raise ValueError(f"Dataset is chosen from [cora, citeseer, pubmed]")

    data_fmt_str = f"{str.lower(dataset_str)}_{atk_name}_{int(atk_rate*100)}"
    return data, data_fmt_str


def load_minmax_data(root, dataset_str, atk_rate):
    strf_atk_rate = f"{atk_rate:.2f}".replace('.', 'p')
    pert_path = osp.join(root, f"{dataset_str}_minmax_CE_{strf_atk_rate}.npz")
    with np.load(pert_path, allow_pickle=True) as loader:
        loader = dict(loader)
        pert_x = torch.Tensor(loader['attr'])
        pert_adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])
        idx_train = loader['idx_train']
        idx_val = loader['idx_val']
        idx_test = loader['idx_test']
    return {
        'x': pert_x,
        'adj': pert_adj,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
    }


def load_pyg_dataset(dataset_str: str, root='./data/'):
    dataset_str = dataset_str.lower()
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=dataset_str)
    else:
        raise ValueError
    return dataset


def largest_connected_components(adj, n_components=1):
    """Select k largest connected components.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        input adjacency matrix
    n_components : int
        n largest connected components we want to select
    """

    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


if __name__ == '__main__':
    dataset = load_pyg_dataset('cora', root='./data/')
    data = dataset[0]
    print(data)
