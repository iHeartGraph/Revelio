from math import sqrt
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import MaskType, ModelMode, ModelReturnType
from torch_geometric.nn.conv import MessagePassing

# from .utils import gumbel_softmax, k_hop_flows, temp_mask

EPS = 1e-15


class Revelio(ExplainerAlgorithm):
    coeffs = {
        'edge_size': 0.,
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
        # whether to use layer edge to estimate message flow importance
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

        # initialize node mask if needed
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

        # initialize message flow mask
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
            )
            self.w = Parameter(torch.zeros(self.flow_ids.shape[1], requires_grad=True, device=device) * 0.1)
        else:
            assert False

    def _set_masks(self, model: torch.nn.Module):
        num_hops = self._num_hops(model)

        '''turn flow masks to layer edge masks'''
        flow_mask = self.flow_mask.tanh()
        # w = F.softplus(self.w)
        w = torch.exp(self.w)
        # idx_pos = self.w > 0
        # idx_neg = ~idx_pos
        # w[idx_neg] *= self.w[idx_neg].sigmoid()
        # w[idx_pos] *= self.w[idx_pos] * 0.25 + .5
        self.l_edge_mask = torch.mv(self.hard_flow_mask, flow_mask).view(num_hops, -1)
        self.l_edge_mask *= w.unsqueeze(1)
        if self.l_edge:
            self.l_edge_mask[self.hard_l_edge_mask] = self.l_edge_mask[self.hard_l_edge_mask].sigmoid()
        else:
            self.edge_mask = self.l_edge_mask.mean(dim=0).sigmoid()

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

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.flow_mask is not None:
            parameters.append(self.flow_mask)
            parameters.append(self.w)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        model.eval()
        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            self._set_masks(model)
            y_hat, y = model(h, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)

            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                self.hard_node_mask = self.node_mask.grad != 0.0

        with torch.no_grad():
            if self.fidelity_plus:
                self.flow_mask.data = -self.flow_mask.data
            if self.l_edge:
                self._set_masks(model)
                self.edge_mask = self.l_edge_mask.mean(dim=0)
            else:
                self._set_masks(model)
                self.l_edge_mask = self.l_edge_mask.sigmoid()

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
        else:
            if self.model_config.mode == ModelMode.binary_classification:
                loss = self._loss_binary_classification(y_hat, y)
            elif self.model_config.mode == ModelMode.multiclass_classification:
                loss = self._loss_multiclass_classification(y_hat, y)
            elif self.model_config.mode == ModelMode.regression:
                loss = self._loss_regression(y_hat, y)
            else:
                assert False

        if self.hard_edge_mask is not None:
            m = self.l_edge_mask[self.hard_l_edge_mask] if self.l_edge else self.edge_mask[self.hard_edge_mask]
            if self.fidelity_plus:
                m = 1 - m
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
        r"""
        find the message flows

        index is the central node
        for graph classification, index is set to None

        returns:
        flow_ids (int): [F, L] -> the sequence of edges a message flow passes
        hard_flow_mask (bool): [L, E, F] -> True indicates a message flow passes on a specific layer edge
        l_edge_count (int): [L, E] -> the number of message flows a layer edge carries

        comments:
        edge_index indicates the ids of edges
        flow_ids indicates the ids of message flows
        """
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

    def supports(self) -> bool:
        return True
