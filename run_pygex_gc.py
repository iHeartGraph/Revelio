import os
import random
import time

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import random_split

from torch_geometric.explain import Explainer
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader

from models import GCN, GIN, GAT
from configs import get_arguments
from load_datasets import get_gc_dataset
from explainers import GNNExplainer, PGExplainer, GraphMaskExplainer, PGMExplainer, Revelio

'''load dataset'''
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

'''load model'''
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = gnn.to(device)
data = data.to(device)

out = gnn(data.x, data.edge_index, batch=data.batch)
prob = F.softmax(out, dim=-1)
pred = out.argmax(dim=-1)
correct = (pred == data.y.view(-1)).sum()
acc = torch.div(correct / len(data), 1e-4, rounding_mode='floor') * 1e-4
print(f'Accuracy: {acc * 100:.2f}')

'''select explainer'''
if args.explainer == 'revelio':
    explainer = Explainer(
        model=gnn,
        algorithm=Revelio(epochs=500, lr=1e-2, l_edge=True),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
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
            task_level='graph',
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
            task_level='graph',
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
            task_level='graph',
            return_type='raw',
        )
    )
elif args.explainer == 'pgmexplainer':
    explainer = Explainer(
        model=gnn,
        algorithm=PGMExplainer(num_samples=1000, pred_threshold=0.01).to(device),
        explanation_type='model',
        node_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        )
    )
else:
    raise ValueError()
explainer.algorithm.fidelity_plus = args.fidelity_plus

res_dir = os.path.join('./res', model_name)
os.makedirs(res_dir, exist_ok=True)

'''select data to explain'''
random.seed(2024)
graph_ids = list(range(len(dataset)))
random.shuffle(graph_ids)
candidates = args.candidates
if candidates is None or candidates > len(dataset):
    candidates = len(dataset)

'''train PGExplainer'''
duration = 0.
if args.explainer == 'pgexplainer':
    print('PGExplaienr training')
    tic = time.perf_counter()
    for epoch in tqdm(range(500)):
        for index in range(candidates):
            data = dataset[index].to(device)
            pred = gnn(data.x, data.edge_index, batch=data.batch).argmax(dim=-1)
            loss = explainer.algorithm.train(epoch, gnn, data.x, data.edge_index,
                                             target=pred)
    duration += time.perf_counter() - tic
    print('duration:', duration)

'''explain'''
duration = 0.
for index in tqdm(graph_ids[:candidates]):
    data = dataset[index].to(device)
    pred = gnn(data.x, data.edge_index, batch=data.batch).argmax(dim=-1)
    tic = time.perf_counter()
    explanation = explainer(data.x, data.edge_index, target=pred if args.explainer == 'pgexplainer' else None)
    duration += time.perf_counter() - tic

    if args.explainer == 'revelio':
        res = explanation.flows
        res['mask'] = explanation.edge_mask
    else:
        res = explanation.edge_mask

    if args.fidelity_plus and args.explainer != 'pgmexplainer':
        torch.save(res, os.path.join(res_dir, args.explainer + '_plus_' + str(index) + '.pt'))
    else:
        torch.save(res, os.path.join(res_dir, args.explainer + '_' + str(index) + '.pt'))
print('duration:', duration)


