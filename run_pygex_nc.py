import os
import random
import time

from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.explain import Explainer
from torch_geometric.utils import k_hop_subgraph

from models import GCN, GIN, GAT
from configs import get_arguments
from load_datasets import get_nc_dataset
from explainers import GNNExplainer, PGExplainer, GraphMaskExplainer, MsgFlow

args = get_arguments()
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
dataset_name = args.dataset.lower()
dataset = get_nc_dataset(dataset_path, dataset_name, self_loops=True)
data = dataset[0]

if args.model.lower() == 'gcn':
    gnn = GCN(in_channels=dataset.num_node_features,
              hidden_channels=args.hidden_channels,
              num_layers=args.num_layers,
              out_channels=dataset.num_classes,
              dropout=args.dropout,
              add_self_loops=False)
elif args.model.lower() == 'gin':
    gnn = GIN(in_channels=dataset.num_node_features,
              hidden_channels=args.hidden_channels,
              num_layers=args.num_layers,
              out_channels=dataset.num_classes,
              dropout=args.dropout)
elif args.model.lower() == 'gat':
    gnn = GAT(in_channels=dataset.num_node_features,
              hidden_channels=args.hidden_channels,
              num_layers=args.num_layers,
              out_channels=dataset.num_classes,
              dropout=args.dropout,
              add_self_loops=False,
              heads=8)
else:
    raise NotImplementedError('GNN not implemented!')
model_dir = './src'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_name = dataset_name + '_' + args.model.lower() + '_l' + str(args.num_layers)
gnn.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pt')))
gnn.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = gnn.to(device)
data = data.to(device)

out = gnn(data.x, data.edge_index)
prob = F.softmax(out, dim=-1)
pred = out.argmax(dim=-1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = torch.div(correct / data.test_mask.sum(), 1e-4, rounding_mode='floor') * 1e-4
print(f'Accuracy: {acc*100:.2f}')

if args.explainer == 'ours':
    explainer = Explainer(
        model=gnn,
        algorithm=MsgFlow(epochs=500, lr=1e-2, l_edge=True),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        )
    )
elif args.explainer == 'gnnexplainer':
    explainer = Explainer(
        model=gnn,
        algorithm=GNNExplainer(epochs=500, lr=1e-2),
        explanation_type='model',
        # node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )
elif args.explainer == 'pgexplainer':
    explainer = Explainer(
        model=gnn,
        algorithm=PGExplainer(epochs=500, lr=3e-3).to(device),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )
elif args.explainer == 'graphmask':
    explainer = Explainer(
        model=gnn,
        algorithm=GraphMaskExplainer(gnn.num_layers, epochs=200, lr=1e-2).to(device),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        )
    )
else:
    raise ValueError()
explainer.algorithm.fidelity_plus = args.fidelity_plus

res_dir = os.path.join('./res', model_name)
os.makedirs(res_dir, exist_ok=True)

random.seed(2024)
node_ids = list(range(data.num_nodes))
random.shuffle(node_ids)
candidates = args.candidates
if candidates is None or candidates > data.num_nodes:
    candidates = data.num_nodes

duration = 0.
if args.explainer == 'pgexplainer':
    print('PGExplaienr training')
    tic = time.perf_counter()
    for epoch in tqdm(range(500)):
        for node_index in node_ids[:candidates]:
            if data.y[node_index] == 0 or pred[node_index] != data.y[node_index]:
                continue
            loss = explainer.algorithm.train(epoch, gnn, data.x, data.edge_index,
                                             target=pred, index=node_index)
    duration += time.perf_counter() - tic
    print('duration:', duration)

duration = 0.
pbar = tqdm(total=candidates)
for node_index in node_ids:
    indices, edge_index, mapping, mask = k_hop_subgraph(node_index,
                                                        args.num_layers + 1,
                                                        data.edge_index,
                                                        relabel_nodes=True,
                                                        num_nodes=data.num_nodes,
                                                        directed=True)
    if edge_index.shape[1] == 1:
        continue
    if candidates == 0:
        break
    candidates -= 1
    pbar.update(1)

    tic = time.perf_counter()
    explanation = explainer(data.x[indices], edge_index,
                            target=pred[indices] if args.explainer == 'pgexplainer' else None, index=mapping[0].item())
    duration += time.perf_counter() - tic

    if args.explainer == 'ours':
        res = explanation.flows
        res['mask'] = explanation.edge_mask
    else:
        res = explanation.edge_mask

    if args.fidelity_plus:
        torch.save(res, os.path.join(res_dir, args.explainer + '_plus_' + str(node_index) + '.pt'))
    else:
        torch.save(res, os.path.join(res_dir, args.explainer + '_' + str(node_index) + '.pt'))
pbar.close()
print('duration:', duration)


