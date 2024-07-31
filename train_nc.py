import os

import torch
import torch.nn.functional as F

from torch_geometric.utils import dropout_edge

from models import GCN, GIN, GAT
from configs import get_arguments
from load_datasets import get_nc_dataset

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

optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=5e-4)

early_stop = 100
early_stop_count = 0
best_acc = 0
best_loss = 100
EPS = 1e-15
for epoch in range(args.epochs):

    gnn.train()
    optimizer.zero_grad()
    out = gnn(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    gnn.eval()
    out = gnn(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    eval_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    eval_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
    print(epoch, eval_acc / data.val_mask.sum(), eval_loss)

    is_best = (eval_acc > best_acc) or (eval_loss < best_loss and eval_acc == best_acc)
    if is_best:
        early_stop_count = 0
        best_acc = eval_acc
        best_loss = eval_loss
        torch.save(gnn.state_dict(), os.path.join(model_dir, model_name + '.pt'))
    else:
        early_stop_count += 1
    if early_stop_count > early_stop:
        break

gnn.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pt')))
gnn.eval()
out = gnn(data.x, data.edge_index)
pred = out.argmax(dim=-1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = torch.div(correct / data.test_mask.sum(), 1e-4, rounding_mode='floor') * 1e-4
print(f'Accuracy: {acc * 100:.2f}')
