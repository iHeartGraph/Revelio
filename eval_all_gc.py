import os
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
from torch.utils.data import random_split

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader

from models import GCN, GIN, GAT
from configs import get_arguments
from load_datasets import get_gc_dataset
from explainers.evaluate import eval_top_edges_drop, eval_top_edges_keep

args = get_arguments()
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
dataset_name = args.dataset
dataset = get_gc_dataset(dataset_path, dataset_name, self_loops=True)
num_train = int(.6 * len(dataset))
num_test = int(.2 * len(dataset))
num_eval = len(dataset) - num_train - num_test
train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                 generator=torch.Generator().manual_seed(0))
train_batches = DataLoader(train, batch_size=args.batch_size, shuffle=True)
eval_batches = DataLoader(eval, batch_size=num_eval)
test_batches = DataLoader(test, batch_size=num_test)

if args.model.lower() == 'gcn':
    gnn = GCN(in_channels=dataset.num_node_features,
              hidden_channels=args.hidden_channels,
              num_layers=args.num_layers,
              out_channels=dataset.num_classes,
              dropout=args.dropout,
              readout=args.readout,
              add_self_loops=False)
elif args.model.lower() == 'gin':
    gnn = GIN(in_channels=dataset.num_node_features,
              hidden_channels=args.hidden_channels,
              num_layers=args.num_layers,
              out_channels=dataset.num_classes,
              dropout=args.dropout,
              readout=args.readout)
elif args.model.lower() == 'gat':
    gnn = GAT(in_channels=dataset.num_node_features,
              hidden_channels=args.hidden_channels,
              num_layers=args.num_layers,
              out_channels=dataset.num_classes,
              dropout=args.dropout,
              readout=args.readout,
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

data = next(iter(test_batches))
device = torch.device('cpu')
gnn = gnn.to(device)
data = data.to(device)

out = gnn(data.x, data.edge_index, batch=data.batch)
prob = F.softmax(out, dim=-1)
pred = out.argmax(dim=-1)
correct = (pred == data.y.view(-1)).sum()
acc = torch.div(correct / len(data), 1e-4, rounding_mode='floor') * 1e-4
print(f'Accuracy: {acc * 100:.2f}')

res_dir = os.path.join('./res', model_name)
random.seed(2024)
graph_ids = list(range(len(dataset)))
random.shuffle(graph_ids)

if args.model == 'gat':
    explainers = ['gradcam', 'deeplift', 'gnnexplainer', 'pgexplainer', 'graphmask', 'flowx', 'ours']
else:
    explainers = ['gradcam', 'deeplift', 'gnnexplainer', 'pgexplainer', 'graphmask', 'gnn-lrp', 'flowx', 'ours']

for explainer in explainers:

    candidates = args.candidates
    if candidates is None or candidates > len(dataset):
        candidates = len(dataset)
    old_pred = []
    new_pred_drop = []
    new_pred_keep = []
    AUC = []
    for index in tqdm(graph_ids[:candidates], desc=explainer.capitalize()):
        if explainer in ['ours', 'gnn-lrp', 'flowx']:
            if args.fidelity_plus and explainer != 'gnn-lrp':
                flows = torch.load(os.path.join(res_dir, explainer + '_plus_' + str(index) + '.pt'),
                                   map_location='cpu')
            else:
                flows = torch.load(os.path.join(res_dir, explainer + '_' + str(index) + '.pt'),
                                   map_location='cpu')
            edge_mask = flows['mask']
        else:
            if args.fidelity_plus and explainer not in ['gradcam', 'deeplift']:
                edge_mask = torch.load(os.path.join(res_dir, explainer + '_plus_' + str(index) + '.pt'),
                                       map_location='cpu')
            else:
                edge_mask = torch.load(os.path.join(res_dir, explainer + '_' + str(index) + '.pt'),
                                       map_location='cpu')
        data = dataset[index]
        out = gnn(data.x, data.edge_index).view(-1)
        prob = F.softmax(out, dim=-1)
        pred = out.argmax(dim=-1)
        pred_hat_drop = eval_top_edges_drop(gnn, data.x, data.edge_index, edge_mask, torch.arange(edge_mask.shape[0]),
                                            pred, None, 0.5, 0.6, 0.7, 0.8, 0.9)
        new_pred_drop.append(pred_hat_drop)
        pred_hat_keep = eval_top_edges_keep(gnn, data.x, data.edge_index, edge_mask, torch.arange(edge_mask.shape[0]),
                                            pred, None, 0.5, 0.6, 0.7, 0.8, 0.9)
        new_pred_keep.append(pred_hat_keep)
        old_pred.append(prob[pred])
        if torch.isnan(edge_mask).sum() > 0:
            continue

        # for synthetic dataset
        if dataset_name.lower() == 'ba_2motifs' and pred == data.y == 1:
            loop_start = data.num_edges - data.num_nodes
            ground_truth = dataset.gen_motif_edge_mask(data, args.num_layers)
            y_true, y_pred = ground_truth[:loop_start], edge_mask[:loop_start]
            if y_true.sum() == y_true.shape[0]:
                continue
            auc = roc_auc_score(y_true, y_pred)
            AUC.append(auc)

    old_pred = torch.tensor(old_pred, dtype=torch.float)
    new_pred_drop = torch.tensor(new_pred_drop, dtype=torch.float, device=old_pred.device)
    new_pred_keep = torch.tensor(new_pred_keep, dtype=torch.float, device=old_pred.device)

    print('Fidelity-')
    fidelity_pos = (old_pred.unsqueeze(1) - new_pred_keep).mean(dim=0)
    print(fidelity_pos.tolist())
    print('Fidelity+')
    fidelity_neg = (old_pred.unsqueeze(1) - new_pred_drop).mean(dim=0)
    print(fidelity_neg.tolist())
    if dataset_name.lower() == 'ba_2motifs':
        print('AUC:', sum(AUC) / len(AUC), ', %d instances are considered.' % (len(AUC)))
