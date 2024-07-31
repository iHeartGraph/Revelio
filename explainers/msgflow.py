import collections

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from math import sqrt
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.loop import add_self_loops, remove_self_loops, maybe_num_nodes
from torch_geometric.data import Data, Batch
from rdkit import Chem
from matplotlib.axes import Axes
from typing import List, Tuple, Optional, Union

from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel, ModelReturnType

# from .utils import gumbel_softmax, k_hop_flows, temp_mask

EPS = 1e-15


class MsgFlow(ExplainerAlgorithm):
    coeffs = {
        'edge_size': 0.,
        # 'edge_size': .0,
        'edge_reduction': 'mean',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }
    fidelity_plus = False

    def __init__(self,
                 epochs: int = 100,
                 lr: float = 0.05,
                 **kwargs
                 ):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.l_edge = kwargs.get('l_edge', True)

        self.node_mask = self.hard_node_mask = None
        self.l_edge_mask = self.hard_l_edge_mask = None
        self.edge_mask = self.hard_edge_mask = None
        self.flow_ids = self.flow_indices = self.l_edge_count = None
        self.flow_mask = None
        self.hard_flow_mask = None

    def _init_masks(self, model: torch.nn.Module, x: Tensor, edge_index: Tensor, index: int):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device, requires_grad=True) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device, requires_grad=True) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device, requires_grad=True) * std)
        else:
            assert False

        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
        if edge_mask_type is None:
            self.l_edge_mask = None
        elif edge_mask_type == MaskType.object:
            self.flow_ids, self.hard_flow_mask, self.l_edge_count = self.find_flows(model, edge_index, index)

            self.hard_l_edge_mask = torch.zeros_like(self.l_edge_count, dtype=torch.bool, device=x.device)
            self.hard_l_edge_mask[self.l_edge_count.nonzero(as_tuple=True)] = True
            self.hard_edge_mask = self.hard_l_edge_mask[0]

            self.flow_mask = Parameter(
                torch.randn(self.flow_ids.shape[0], requires_grad=True, device=device) * std
                # torch.ones(self.flow_ids.shape[0], requires_grad=True, device=device)
                # torch.rand(self.flow_ids.shape[0], requires_grad=True, device=device) * 2 - 1
            )
            self.w = Parameter(torch.zeros(self.flow_ids.shape[1], requires_grad=True, device=device) * 0.1)
        else:
            assert False

    def _set_masks(self, model: torch.nn.Module):
        num_hops = self._num_hops(model)
        # self.l_edge_mask = torch.full_like(self.l_edge_count, -math.inf)
        # self.l_edge_mask[self.hard_l_edge_mask] = 0
        # self.l_edge_mask = torch.zeros_like(self.l_edge_count, device=self.flow_mask.device)

        flow_mask = self.flow_mask.tanh() #* 8 #/ (sqrt(self.flow_mask.shape[0]) / num_hops)
        # flow_mask = F.softplus(self.flow_mask)
        # flow_mask = F.dropout(flow_mask, 0.5, training=self.training)
        # flow_mask = F.softsign(self.flow_mask)
        # flow_mask = (flow_mask - flow_mask.min()) / (flow_mask.max() - flow_mask.min() + self.coeffs['EPS'])
        # x = torch.ones(num_hops) * (self.hard_edge_mask.sum() / self.flow_mask.shape[0])
        # x = self.hard_l_edge_mask.sum(dim=1) / self.flow_mask.shape[0]  # * self.N
        # x = self.hard_l_edge_mask.sum(dim=1) / self.N
        # x = x.pow(torch.arange(num_hops-1, -1, -1, device=x.device))
        # x = (1 / torch.tensor(range(1, num_hops+1))) / self.N
        # w = self.w.sigmoid() #* x
        # w = torch.ones_like(self.w)
        # w = F.softplus(self.w)
        w = torch.exp(self.w)
        # idx_pos = self.w > 0
        # idx_neg = ~idx_pos
        # w[idx_neg] *= self.w[idx_neg].sigmoid()
        # w[idx_pos] *= self.w[idx_pos] * 0.25 + .5
        # w = w / x
        # w[w > 1.] += torch.ones(1) - w[w > 1.].detach()
        # w[w < 0.] += torch.zeros(1) - w[w < 0.].detach()
        # w = F.softmax(self.w, dim=-1)
        # layer_indices = torch.arange(num_hops, dtype=torch.long, device=self.flow_mask.device)
        self.l_edge_mask = torch.mv(self.hard_flow_mask, flow_mask).view(num_hops, -1)
        # self.l_edge_mask[self.hard_l_edge_mask] /= self.l_edge_count[self.hard_l_edge_mask]
        # self.l_edge_mask[self.hard_l_edge_mask] = self.l_edge_mask[self.hard_l_edge_mask]
        self.l_edge_mask *= w.unsqueeze(1)
        # self.l_edge_mask /= x.unsqueeze(1)
        # self.l_edge_mask[self.hard_l_edge_mask] += self.coeffs['EPS']
        # self.l_edge_mask[self.hard_l_edge_mask] /= self.l_edge_count[self.hard_l_edge_mask]
        if self.l_edge:
            self.l_edge_mask[self.hard_l_edge_mask] = self.l_edge_mask[self.hard_l_edge_mask].sigmoid()
            # self.l_edge_mask /= torch.linalg.norm(self.l_edge_mask[self.hard_l_edge_mask].detach())
            # self.l_edge_mask[~self.hard_l_edge_mask] = self.l_edge_mask[self.hard_l_edge_mask].min()
            # hard_edge_mask = F.dropout(self.hard_edge_mask.float(), 0.25, training=self.training).bool()
            # self.l_edge_mask[:, ~hard_edge_mask] = 0
            # self.l_edge_mask[self.l_edge_mask == 0] = 1
            # self.l_edge_mask[self.hard_l_edge_mask] -= self.l_edge_mask[self.hard_l_edge_mask].min()
            # self.l_edge_mask[self.hard_l_edge_mask] /= self.l_edge_mask[self.hard_l_edge_mask].max().detach() + self.coeffs['EPS']
            # self.l_edge_mask /= self.l_edge_mask.max(dim=1, keepdim=True)[0]
        else:
            # self.l_edge_mask[self.hard_l_edge_mask] = self.l_edge_mask[self.hard_l_edge_mask].sigmoid()
            # self.l_edge_mask = F.normalize(self.l_edge_mask)
            # self.edge_mask = self.l_edge_mask.mean(dim=0)
            # self.l_edge_mask /= x.unsqueeze(1)
            self.edge_mask = self.l_edge_mask.mean(dim=0).sigmoid()
            # self.edge_mask[self.hard_edge_mask] -= self.edge_mask[self.hard_edge_mask].min()
            # self.edge_mask[self.hard_edge_mask] /= self.edge_mask[self.hard_edge_mask].max() + self.coeffs['EPS']
        # self.edge_mask = self.l_edge_mask.sum(dim=0)
        # self.edge_mask = (self.l_edge_mask * w.view(-1, 1)).sigmoid().mean(dim=0)

        # self.edge_mask = torch.zeros(self.l_edge_mask.shape[1])
        # self.edge_mask[self.hard_edge_mask] = self.l_edge_mask.sum(dim=0)[self.hard_edge_mask] / self.l_edge_count.sum(dim=0)[self.hard_edge_mask]
        # self.l_edge_mask[self.hard_l_edge_mask] /= self.l_edge_count[self.hard_l_edge_mask]

        # self.edge_mask[self.hard_edge_mask] = self.edge_mask[self.hard_edge_mask].sigmoid()
        # self.edge_mask[self.hard_edge_mask] -= self.edge_mask[self.hard_edge_mask].min()
        # self.edge_mask[self.hard_edge_mask] /= self.edge_mask[self.hard_edge_mask].max() + self.coeffs['EPS']
        # self.edge_mask[self.hard_edge_mask] += self.coeffs['EPS']
        # self.edge_mask /= self.edge_mask.abs().max().detach() + self.coeffs['EPS']
        # self.l_edge_mask = (self.l_edge_mask - self.l_edge_mask.min()) / (
        #             self.l_edge_mask.max() - self.l_edge_mask.min() + self.coeffs['EPS'])

        # loop_mask = torch.ones(self.l_edge_mask.shape[1], dtype=torch.bool)

        i = 0
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.explain = True
                if self.l_edge:
                    module._edge_mask = self.l_edge_mask[i]
                else:
                    module._edge_mask = self.edge_mask
                # module._loop_mask = loop_mask
                module._apply_sigmoid = False
                i += 1

    def _clean_model(self, model: torch.nn.Module):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True
        self.node_mask = self.hard_node_mask = None
        self.l_edge_mask = self.hard_l_edge_mask = None
        self.edge_mask = self.hard_edge_mask = None
        self.flow_ids = self.flow_indices = self.l_edge_count = None
        self.hard_flow_mask = None

    def forward(self, model: torch.nn.Module, x: Tensor, edge_index: Tensor, *, target: Tensor,
                index: Optional[Union[int, Tensor]] = None, **kwargs):
        self._init_masks(model, x, edge_index, index)
        self.N = x.shape[0]
        num_hops = self._num_hops(model)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.flow_mask is not None:
            parameters.append(self.flow_mask)
            parameters.append(self.w)
            # parameters.append({'params': self.w, 'lr': 1e-2})

        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        # optimizer = torch.optim.SGD(parameters, lr=self.lr)
        # opt = torch.optim.Adam([self.w], lr=1e-2)

        model.eval()
        for i in range(self.epochs):
            optimizer.zero_grad()
            # opt.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            self._set_masks(model)
            y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            # torch.nn.utils.clip_grad_value_(parameters, clip_value=25)
            optimizer.step()
            # opt.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                self.hard_node_mask = self.node_mask.grad != 0.0

        # self._fine_tune(model, x, edge_index, target=target, index=index)

        with torch.no_grad():
            if self.fidelity_plus:
                self.flow_mask.data = -self.flow_mask.data
            if self.l_edge:
                # self.l_edge = False
                self._set_masks(model)
                # self.l_edge = True
                # self.l_edge_mask[self.hard_l_edge_mask] -= self.l_edge_mask[self.hard_l_edge_mask].min()
                # self.l_edge_mask[self.hard_l_edge_mask] /= self.l_edge_mask[self.hard_l_edge_mask].max() + self.coeffs['EPS']
                # self.l_edge_mask = self.l_edge_mask.sigmoid()
                self.edge_mask = self.l_edge_mask.mean(dim=0)
            else:
                self._set_masks(model)
                self.l_edge_mask = self.l_edge_mask.sigmoid()
            # for l in range(self.l_edge_mask.size(0)):
            #     print(edge_index[:, torch.nonzero(self.hard_l_edge_mask[l])].view(2, -1))

            node_mask = self._post_process_mask(
                self.node_mask,
                self.hard_node_mask,
                apply_sigmoid=True,
            )
            l_edge_mask = self._post_process_mask(
                self.l_edge_mask,
                self.hard_l_edge_mask,
                apply_sigmoid=False,
            )
            # self.edge_mask[self.hard_edge_mask] -= self.edge_mask[self.hard_edge_mask].min()
            # self.edge_mask[self.hard_edge_mask] /= self.edge_mask[self.hard_edge_mask].max() + self.coeffs['EPS']
            # self.edge_mask[self.hard_edge_mask] += self.coeffs['EPS']
            edge_mask = self._post_process_mask(
                self.edge_mask,
                self.hard_edge_mask,
                apply_sigmoid=False,
            )

            flow_ids = self.flow_ids.detach()
            flow_mask = self.flow_mask.tanh().detach()
            flows = {'ids': flow_ids, 'score': flow_mask}

        self._clean_model(model)

        return Explanation(node_mask=node_mask,
                           edge_mask=edge_mask,
                           l_edge_mask=l_edge_mask,
                           flows=flows)

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:

        if self.fidelity_plus:
            if self.model_config.return_type == ModelReturnType.raw:
                y_hat = F.softmax(y_hat, dim=-1)
            elif self.model_config.return_type != ModelReturnType.probs:
                assert False
            loss = F.binary_cross_entropy(y_hat[:, y].view_as(y), torch.zeros_like(y, dtype=torch.float))
            # loss = (F.binary_cross_entropy(y_hat[:, y].view_as(y), torch.zeros_like(y, dtype=torch.float)) +
            #         F.binary_cross_entropy(y_hat.sum(dim=-1) - y_hat[:, y].view_as(y), torch.ones_like(y, dtype=torch.float)))
        else:
            if self.model_config.mode == ModelMode.binary_classification:
                loss = self._loss_binary_classification(y_hat, y)
            elif self.model_config.mode == ModelMode.multiclass_classification:
                loss = self._loss_multiclass_classification(y_hat, y)
            elif self.model_config.mode == ModelMode.regression:
                loss = self._loss_regression(y_hat, y)
            else:
                assert False

        # if self.model_config.mode == ModelMode.binary_classification:
        #     loss = self._loss_binary_classification(y_hat, y)
        # elif self.model_config.mode == ModelMode.multiclass_classification:
        #     loss = self._loss_multiclass_classification(y_hat, y)
        # elif self.model_config.mode == ModelMode.regression:
        #     loss = self._loss_regression(y_hat, y)
        # else:
        #     assert False
        # if self.fidelity_plus:
        #     loss = -loss

        if self.hard_edge_mask is not None:
            m = self.l_edge_mask[self.hard_l_edge_mask] if self.l_edge else self.edge_mask[self.hard_edge_mask]
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            # ent = -m * torch.log(m + self.coeffs['EPS']) - (
            #         1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            # loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                    1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def find_flows(self, model: torch.nn.Module, edge_index: Tensor, index: int = None):
        num_hops = self._num_hops(model)
        flow_ids = []
        # how many times the edge is used in a certain layer
        l_edge_count = torch.zeros(num_hops, edge_index.shape[1], device=edge_index.device)

        def dfs(candidate_indices: List, saved_indices: List = []):
            for nxt in candidate_indices:
                saved_indices.append(nxt)
                src, tgt = edge_index[:, nxt]
                nxt_indices = (edge_index[1, :] == src).nonzero().view(-1).tolist()
                if len(saved_indices) >= num_hops:
                    flow_ids.append(saved_indices[::-1])
                    l_edge_count[range(num_hops), flow_ids[-1]] += 1
                else:
                    dfs(nxt_indices, saved_indices)
                saved_indices.pop()

        if index is not None:
            init_indices = (edge_index[1, :].cpu() == index).nonzero().view(-1).tolist()
        else:
            init_indices = list(range(edge_index.shape[1]))
        dfs(init_indices)

        hard_flow_mask = torch.zeros(num_hops, edge_index.shape[1], len(flow_ids), dtype=torch.bool)
        i_layer = list(range(num_hops))
        for i_flow, i_edge in enumerate(flow_ids):
            i_flow = [i_flow] * num_hops
            hard_flow_mask[i_layer, i_edge, i_flow] = True

        hard_flow_mask = hard_flow_mask.view(-1, hard_flow_mask.shape[-1]).to_sparse_csr()  # (L * E) * F
        hard_flow_mask = hard_flow_mask.float().to(edge_index.device)

        flow_ids = torch.tensor(flow_ids, dtype=torch.long, device=edge_index.device)

        return flow_ids, hard_flow_mask, l_edge_count

    @staticmethod
    def flow_to_edge(flow_ids: Tensor, flow_mask: Tensor, num_edges: int):
        EPS = 10E-15

        num_hops = flow_ids.shape[1]
        l_edge_mask = torch.zeros(num_hops, num_edges)
        hard_l_edge_mask = torch.zeros_like(l_edge_mask, dtype=torch.bool)
        layer_indices = torch.tensor(range(num_hops), dtype=torch.long)
        for i in range(flow_mask.shape[0]):
            l_edge_mask[layer_indices, flow_ids[i]] += flow_mask[i]
            hard_l_edge_mask[layer_indices, flow_ids[i]] = True
        edge_mask = l_edge_mask.sum(dim=0)
        hard_edge_mask = hard_l_edge_mask[0]

        l_edge_mask[hard_l_edge_mask] += l_edge_mask[hard_l_edge_mask].min().abs() + EPS
        l_edge_mask[~hard_l_edge_mask] = 0
        edge_mask[hard_edge_mask] += edge_mask[hard_edge_mask].min().abs() + EPS
        edge_mask[~hard_edge_mask] = 0

        return edge_mask, hard_edge_mask, l_edge_mask, hard_l_edge_mask

    def supports(self) -> bool:
        return True
