import os
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F

from torch_geometric.utils import k_hop_subgraph

from models import GCN, GIN, GAT
from configs import get_arguments
from load_datasets import get_nc_dataset
from explainers.evaluate import eval_top_edges_drop, eval_top_edges_keep

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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# gnn = gnn.to(device)
# data = data.to(device)

out = gnn(data.x, data.edge_index)
prob = F.softmax(out, dim=-1)
pred = out.argmax(dim=-1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = torch.div(correct / data.test_mask.sum(), 1e-4, rounding_mode='floor') * 1e-4
print(f'Accuracy: {acc * 100:.2f}')

res_dir = os.path.join('./res', model_name)
random.seed(2024)
node_ids = list(range(data.num_nodes))
random.shuffle(node_ids)
candidates = args.candidates
if candidates is None or candidates > data.num_nodes:
    candidates = data.num_nodes

new_pred_drop = []
new_pred_keep = []
AUC = []
eval_nodes = []
loop_start = data.num_edges - data.num_nodes

pbar = tqdm(total=candidates)
for node_index in node_ids:
    # if pred[node_index] == data.y[node_index] == 0 or pred[node_index] != data.y[node_index]:
    #     continue
    indices, edge_index, mapping, mask = k_hop_subgraph(node_index,
                                                        args.num_layers + 1,
                                                        data.edge_index,
                                                        relabel_nodes=True,
                                                        num_nodes=data.num_nodes,
                                                        directed=True)
    _, _, _, mask_ = k_hop_subgraph(mapping,
                                    args.num_layers,
                                    edge_index,
                                    relabel_nodes=True,
                                    # num_nodes=data.num_nodes,
                                    directed=True)
    if edge_index.shape[1] == 1:
        continue
    if candidates == 0:
        break
    candidates -= 1
    eval_nodes.append(node_index)
    pbar.update(1)

    if args.explainer in ['ours', 'gnn-lrp', 'flowx']:
        if args.fidelity_plus:
            flows = torch.load(os.path.join(res_dir, args.explainer + '_plus_' + str(node_index) + '.pt'),
                               map_location='cpu')
        else:
            flows = torch.load(os.path.join(res_dir, args.explainer + '_' + str(node_index) + '.pt'),
                               map_location='cpu')
        edge_mask = flows['mask']
    else:
        if args.fidelity_plus:
            edge_mask = torch.load(os.path.join(res_dir, args.explainer + '_plus_' + str(node_index) + '.pt'),
                                   map_location='cpu')
        else:
            edge_mask = torch.load(os.path.join(res_dir, args.explainer + '_' + str(node_index) + '.pt'),
                                   map_location='cpu')
    valid_indices = torch.nonzero(mask_.view(-1)).view(-1)

    pred_hat_drop = eval_top_edges_drop(gnn, data.x[indices], edge_index, edge_mask, valid_indices, pred[node_index],
                                        mapping[0].item(), 0.5, 0.6, 0.7, 0.8, 0.9)
    new_pred_drop.append(pred_hat_drop)
    pred_hat_keep = eval_top_edges_keep(gnn, data.x[indices], edge_index, edge_mask, valid_indices, pred[node_index],
                                        mapping[0].item(), 0.5, 0.6, 0.7, 0.8, 0.9)
    new_pred_keep.append(pred_hat_keep)

    # for synthetic dataset
    if (dataset_name.lower() in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle'] and
            0 < data.y[node_index] == pred[node_index]):
        ground_truth = dataset.gen_motif_edge_mask(data, node_index, args.num_layers)
        indices1, edge_index1, mapping1, mask1 = k_hop_subgraph(node_index,
                                                                args.num_layers,
                                                                data.edge_index,
                                                                relabel_nodes=True,
                                                                num_nodes=data.num_nodes,
                                                                directed=True)
        '''edge masks without self-loop'''
        # edge mask to undirected
        # mask_adj = to_dense_adj(edge_index, edge_attr=edge_mask).squeeze()
        # mask_adj += mask_adj.clone().t()
        # edge_mask = mask_adj[(edge_index[0, :], edge_index[1, :])]

        # extend to fit the size of ground_truth
        mask[loop_start:] = False  # remove self-loop
        edge_mask = edge_mask[:torch.sum(mask)]
        edge_mask_long = torch.zeros(data.edge_index.shape[1], dtype=torch.float)
        # mask = mask[:loop_start]
        edge_mask_long[mask] = edge_mask
        # mask1 = mask1[:loop_start]
        mask1[loop_start:] = False
        y_true, y_pred = ground_truth[mask1], edge_mask_long[mask1]  # only compare edges within L layers
        if y_true.sum() == y_true.shape[0]:
            continue
        auc = roc_auc_score(y_true, y_pred)
        AUC.append(auc)
pbar.close()

eval_nodes = torch.tensor(eval_nodes, dtype=torch.long)
old_pred = prob[eval_nodes, pred[eval_nodes]]
new_pred_drop = torch.tensor(new_pred_drop, dtype=torch.float, device=old_pred.device)
new_pred_keep = torch.tensor(new_pred_keep, dtype=torch.float, device=old_pred.device)

print('Fidelity-')
fidelity_pos = (old_pred.unsqueeze(1) - new_pred_keep).mean(dim=0)
print(fidelity_pos.tolist())
print('Fidelity+')
fidelity_neg = (old_pred.unsqueeze(1) - new_pred_drop).mean(dim=0)
print(fidelity_neg.tolist())
if dataset_name in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle']:
    print('AUC:', sum(AUC) / len(AUC), ', %d instances are considered.' % (len(AUC)))
