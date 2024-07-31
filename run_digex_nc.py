import os
import random
import time

from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.utils import k_hop_subgraph

from models import GCN, GIN, GAT
from configs import get_arguments
from load_datasets import get_nc_dataset
from explainers import GNN_LRP, FlowX, DeepLIFT, GradCAM

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
print(f'Accuracy: {acc * 100:.2f}')

if args.explainer == 'gnn-lrp':
    explainer = GNN_LRP(gnn, explain_graph=False)
elif args.explainer == 'flowx':
    explainer = FlowX(gnn, explain_graph=False)
    explainer.fidelity_plus = args.fidelity_plus
elif args.explainer == 'deeplift':
    explainer = DeepLIFT(gnn, explain_graph=False)
elif args.explainer == 'gradcam':
    explainer = GradCAM(gnn, explain_graph=False)
else:
    raise ValueError()

res_dir = os.path.join('./res', model_name)
os.makedirs(res_dir, exist_ok=True)

random.seed(2024)
node_ids = list(range(data.num_nodes))
random.shuffle(node_ids)
candidates = args.candidates
if candidates is None or candidates > data.num_nodes:
    candidates = data.num_nodes

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
    explanation = explainer(data.x[indices],
                            edge_index,
                            node_idx=mapping[0].item(),
                            target=pred[node_index].item(),
                            num_classes=dataset.num_classes)
    duration += time.perf_counter() - tic
    if args.explainer in ['gnn-lrp', 'flowx']:
        walks, masks = explanation
        walks['score'] = walks['score'].view(-1)
        walks['mask'] = masks[0]
        res = walks
    else:
        res = explanation[0]

    if args.explainer == 'flowx' and args.fidelity_plus:
        torch.save(res, os.path.join(res_dir, args.explainer + '_plus_' + str(node_index) + '.pt'))
    else:
        torch.save(res, os.path.join(res_dir, args.explainer + '_' + str(node_index) + '.pt'))
pbar.close()
print("duration:", duration)
