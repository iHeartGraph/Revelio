import os

import torch
import torch.nn.functional as F
from torch.utils.data import random_split

from torch_geometric.loader import DataLoader

from models import GCN, GIN, GAT
from configs import get_arguments
from load_datasets import get_gc_dataset

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

optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=5e-4)

early_stop = 100
early_stop_count = 0
best_acc = 0
best_loss = 100
EPS = 1e-15
for epoch in range(args.epochs):

    gnn.train()
    for data in train_batches:
        optimizer.zero_grad()
        out = gnn(data.x, data.edge_index, batch=data.batch)
        loss = F.cross_entropy(out, data.y.view(-1))
        loss.backward()
        optimizer.step()

    gnn.eval()
    data = next(iter(eval_batches))
    out = gnn(data.x, data.edge_index, batch=data.batch)
    pred = out.argmax(dim=-1)
    eval_acc = (pred == data.y.view(-1)).sum()
    eval_loss = F.cross_entropy(out, data.y.view(-1))
    print(epoch, eval_acc / len(data), eval_loss)

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

data = next(iter(test_batches))
gnn.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pt')))
gnn.eval()
out = gnn(data.x, data.edge_index, batch=data.batch)
pred = out.argmax(dim=-1)
correct = (pred == data.y.view(-1)).sum()
acc = torch.div(correct / len(data), 1e-4, rounding_mode='floor') * 1e-4
print(f'Accuracy: {acc * 100:.2f}')
