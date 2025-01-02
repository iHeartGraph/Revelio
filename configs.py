import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        # choices=['cora', 'citeseer', 'pubmed', 'cs', 'facebook', 'github', 'lastfmasia',
                        # 'house', 'ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle', 'ba_2motifs',
                        # 'MUTAG', 'BBBP', 'ENZYMES', 'PROTEINS', 'TOX21', 'CLINTOX']
                        )
    '''model'''
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gin', 'gat'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    # ba_shapes, ba2motif: 1e-3
    # default: 1e-2
    parser.add_argument('--lr', type=float, default=1e-2)
    # house/ba2motif: 0
    # default: 0.5
    parser.add_argument('--dropout', type=float, default=0.5)
    # synthetic: leakyrelu
    # default: relu
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--readout', type=str, default='mean')
    parser.add_argument('--batch_size', type=int, default=32)
    '''explainer'''
    parser.add_argument('--explainer', type=str, default='revelio',
                        choices=['gnnexplainer', 'pgexplainer', 'graphmask', 'pgmexplainer',
                                 'gnn-lrp', 'flowx', 'deeplift', 'gradcam', 'flowx', 'revelio'])
    parser.add_argument('--candidates', type=int, default=50)  # the number of candidates to explain
    parser.add_argument('--fidelity_plus', action='store_true', default=False)

    return parser.parse_args()
