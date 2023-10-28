from ogb.linkproppred import PygLinkPropPredDataset
from ogb.linkproppred import Evaluator
from torch_geometric import transforms as T
import torch


def load_dataset(dataset_name):
    dataset = PygLinkPropPredDataset(name=dataset_name)
    data = dataset[0]
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(name=dataset_name)

    if hasattr(data, 'edge_weight'):
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)

    if hasattr(data, 'x'):
        if data.x is not None:
            data.x = data.x.to(torch.float)

    if dataset_name == 'ogbl-citation2':
        data.adj_t = data.adj_t.to_symmetric()

    data = T.ToSparseTensor()(data)
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack([col, row], dim=0)

    return data, split_edge, evaluator
