import collections

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt
from math import sqrt, floor, ceil
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.loop import add_self_loops, remove_self_loops, maybe_num_nodes
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, one_hot, get_num_hops
from rdkit import Chem
from matplotlib.axes import Axes
from typing import List, Tuple, Optional, Union

from torch_geometric.explain import Explainer, ExplainerConfig, ModelConfig, Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel


class temp_mask(object):

    def __init__(self, model, temp_edge_mask):
        self.model = model
        self.temp_edge_mask = temp_edge_mask

    def __enter__(self):

        i = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = True
                module._edge_mask = self.temp_edge_mask[i]
                # module._loop_mask = torch.ones_like(self.temp_edge_mask[i], dtype=torch.bool)
                module._apply_sigmoid = False
                i += 1

    def __exit__(self, *args):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True


def eval_top_edges_drop(model: torch.nn.Module,
                        x: Tensor,
                        edge_index: Tensor,
                        edge_mask: Tensor,
                        valid_indices: Tensor,
                        target: Optional[Union[int, Tensor]],
                        node_idx: int = None,
                        *sparsity: float) -> List:
    if not sparsity:
        sparsity = [0.7]

    # pred = model(x, edge_index)
    # if node_idx is not None:
    #     pred = pred[node_idx]
    # pred = F.softmax(pred, dim=-1)[target]

    sub_mask = 1 - edge_mask[valid_indices]
    _, sub_indices = torch.sort(sub_mask, descending=True)
    mask_len = sub_mask.shape[0]
    new_pred = []
    for s in sparsity:
        split_point = mask_len - ceil((1 - s) * mask_len)
        important_sub_indices = sub_indices[: split_point]
        important_indices = valid_indices[important_sub_indices]
        hard_edge_mask = torch.zeros_like(edge_mask, dtype=torch.float)
        hard_edge_mask[important_indices] = 1
        # print(torch.sum(hard_edge_mask))
        # print(edge_index.t()[important_indices])
        with temp_mask(model, [hard_edge_mask for _ in range(get_num_hops(model))]):
            pred_hat = model(x, edge_index)
            if node_idx is not None:
                pred_hat = pred_hat[node_idx]
            pred_hat = F.softmax(pred_hat, dim=-1).view(-1)[target]
        new_pred.append(pred_hat.item())
    return new_pred


def eval_top_edges_keep(model: torch.nn.Module,
                        x: Tensor,
                        edge_index: Tensor,
                        edge_mask: Tensor,
                        valid_indices: Tensor,
                        target: Optional[Union[int, Tensor]],
                        node_idx: int = None,
                        *sparsity: float) -> List:
    if not sparsity:
        sparsity = [0.7]

    # pred = model(x, edge_index)
    # if node_idx is not None:
    #     pred = pred[node_idx]
    # pred = F.softmax(pred, dim=-1)[target]

    sub_mask = edge_mask[valid_indices]
    _, sub_indices = torch.sort(sub_mask, descending=True)
    mask_len = sub_mask.shape[0]
    new_pred = []
    for s in sparsity:
        split_point = ceil((1 - s) * mask_len)
        important_sub_indices = sub_indices[: split_point]
        important_indices = valid_indices[important_sub_indices]
        hard_edge_mask = torch.zeros_like(edge_mask, dtype=torch.float)
        hard_edge_mask[important_indices] = 1
        # print(torch.sum(hard_edge_mask))
        # print(edge_index.t()[important_indices])
        with temp_mask(model, [hard_edge_mask for _ in range(get_num_hops(model))]):
            pred_hat = model(x, edge_index)
            if node_idx is not None:
                pred_hat = pred_hat[node_idx]
            pred_hat = F.softmax(pred_hat, dim=-1).view(-1)[target]
        new_pred.append(pred_hat.item())
    return new_pred