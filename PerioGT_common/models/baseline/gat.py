# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GAT-based model for regression and classification on graphs.
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import WeightAndSum


class MLPPredictor(nn.Module):
    """Two-layer MLP for regression or soft classification
    over multiple tasks from graph representations.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input graph features
    hidden_feats : int
        Number of graph features in hidden layers
    n_tasks : int
        Number of tasks, which is also the output size.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(MLPPredictor, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, feats):
        """Make prediction.

        Parameters
        ----------
        feats : FloatTensor of shape (B, M3)
            * B is the number of graphs in a batch
            * M3 is the input graph feature size, must match in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
        """
        return self.predict(feats)


class WeightedSumAndMax(nn.Module):
    r"""Apply weighted sum and max pooling to the node
    representations and concatenate the results.

    Parameters
    ----------
    in_feats : int
        Input node feature size
    """
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):
        """Readout

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        h_g : FloatTensor of shape (B, 2 * M1)
            * B is the number of graphs in the batch
            * M1 is the input node feature size, which must match
              in_feats in initialization
        """
        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g


# pylint: disable=W0221
class GATLayer(nn.Module):
    r"""Single GAT layer from `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features
    out_feats : int
        Number of output node features
    num_heads : int
        Number of attention heads
    feat_drop : float
        Dropout applied to the input features
    attn_drop : float
        Dropout applied to attention values of edges
    alpha : float
        Hyperparameter in LeakyReLU, which is the slope for negative values.
        Default to 0.2.
    residual : bool
        Whether to perform skip connection, default to True.
    agg_mode : str
        The way to aggregate multi-head attention results, can be either
        'flatten' for concatenating all-head results or 'mean' for averaging
        all head results.
    activation : activation function or None
        Activation function applied to the aggregated multi-head results, default to None.
    bias : bool
        Whether to use bias in the GAT layer.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph. Defaults to False.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop,
        attn_drop,
        alpha=0.2,
        residual=True,
        agg_mode="flatten",
        activation=None,
        bias=True,
        allow_zero_in_degree=False,
    ):
        super(GATLayer, self).__init__()

        self.gat_conv = GATConv(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=alpha,
            residual=residual,
            bias=bias,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        assert agg_mode in ["flatten", "mean"]
        self.agg_mode = agg_mode
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gat_conv.reset_parameters()

    def forward(self, bg, feats):
        """Update node representations

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              out_feats in initialization if self.agg_mode == 'mean' and
              out_feats * num_heads in initialization otherwise.
        """
        feats = self.gat_conv(bg, feats)
        if self.agg_mode == "flatten":
            feats = feats.flatten(1)
        else:
            feats = feats.mean(1)

        if self.activation is not None:
            feats = self.activation(feats)

        return feats


class GAT(nn.Module):
    r"""GAT from `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the output size of an attention head in the i-th GAT layer.
        ``len(hidden_feats)`` equals the number of GAT layers. By default, we use ``[32, 32]``.
    num_heads : list of int
        ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
        ``len(num_heads)`` equals the number of GAT layers. By default, we use 4 attention heads
        for each GAT layer.
    feat_drops : list of float
        ``feat_drops[i]`` gives the dropout applied to the input features in the i-th GAT layer.
        ``len(feat_drops)`` equals the number of GAT layers. By default, this will be zero for
        all GAT layers.
    attn_drops : list of float
        ``attn_drops[i]`` gives the dropout applied to attention values of edges in the i-th GAT
        layer. ``len(attn_drops)`` equals the number of GAT layers. By default, this will be zero
        for all GAT layers.
    alphas : list of float
        Hyperparameters in LeakyReLU, which are the slopes for negative values. ``alphas[i]``
        gives the slope for negative value in the i-th GAT layer. ``len(alphas)`` equals the
        number of GAT layers. By default, this will be 0.2 for all GAT layers.
    residuals : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GAT layer.
        ``len(residual)`` equals the number of GAT layers. By default, residual connection
        is performed for each GAT layer.
    agg_modes : list of str
        The way to aggregate multi-head attention results for each GAT layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
        GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
        all-head results for each GAT layer.
    activations : list of activation function or None
        ``activations[i]`` gives the activation function applied to the aggregated multi-head
        results for the i-th GAT layer. ``len(activations)`` equals the number of GAT layers.
        By default, no activation is applied for each GAT layer.
    biases : list of bool
        ``biases[i]`` gives whether to use bias for the i-th GAT layer. ``len(activations)``
        equals the number of GAT layers. By default, we use bias for all GAT layers.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph for all layers. By default, will not
        allow zero in degree nodes.
    """

    def __init__(
        self,
        in_feats,
        hidden_feats=None,
        num_heads=None,
        feat_drops=None,
        attn_drops=None,
        alphas=None,
        residuals=None,
        agg_modes=None,
        activations=None,
        biases=None,
        allow_zero_in_degree=False,
    ):
        super(GAT, self).__init__()

        if hidden_feats is None:
            hidden_feats = [32, 32]

        n_layers = len(hidden_feats)
        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0.0 for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0.0 for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [True for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ["flatten" for _ in range(n_layers - 1)]
            agg_modes.append("mean")
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        if biases is None:
            biases = [True for _ in range(n_layers)]
        lengths = [
            len(hidden_feats),
            len(num_heads),
            len(feat_drops),
            len(attn_drops),
            len(alphas),
            len(residuals),
            len(agg_modes),
            len(activations),
            len(biases),
        ]
        assert len(set(lengths)) == 1, (
            "Expect the lengths of hidden_feats, num_heads, "
            "feat_drops, attn_drops, alphas, residuals, "
            "agg_modes, activations, and biases to be the same, "
            "got {}".format(lengths)
        )
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                GATLayer(
                    in_feats,
                    hidden_feats[i],
                    num_heads[i],
                    feat_drops[i],
                    attn_drops[i],
                    alphas[i],
                    residuals[i],
                    agg_modes[i],
                    activations[i],
                    biases[i],
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )
            if agg_modes[i] == "flatten":
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] if agg_modes[-1] == 'mean' and
              hidden_sizes[-1] * num_heads[-1] otherwise.
        """
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats


# pylint: disable=W0221
class GATPredictor(nn.Module):
    r"""GAT-based model for regression and classification on graphs.

    GAT is introduced in `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__.
    This model is based on GAT and can be used for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the output size of an attention head in the i-th GAT layer.
        ``len(hidden_feats)`` equals the number of GAT layers. By default, we use ``[32, 32]``.
    num_heads : list of int
        ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
        ``len(num_heads)`` equals the number of GAT layers. By default, we use 4 attention heads
        for each GAT layer.
    feat_drops : list of float
        ``feat_drops[i]`` gives the dropout applied to the input features in the i-th GAT layer.
        ``len(feat_drops)`` equals the number of GAT layers. By default, this will be zero for
        all GAT layers.
    attn_drops : list of float
        ``attn_drops[i]`` gives the dropout applied to attention values of edges in the i-th GAT
        layer. ``len(attn_drops)`` equals the number of GAT layers. By default, this will be zero
        for all GAT layers.
    alphas : list of float
        Hyperparameters in LeakyReLU, which are the slopes for negative values. ``alphas[i]``
        gives the slope for negative value in the i-th GAT layer. ``len(alphas)`` equals the
        number of GAT layers. By default, this will be 0.2 for all GAT layers.
    residuals : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GAT layer.
        ``len(residual)`` equals the number of GAT layers. By default, residual connection
        is performed for each GAT layer.
    agg_modes : list of str
        The way to aggregate multi-head attention results for each GAT layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
        GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
        multi-head results for intermediate GAT layers and compute mean of multi-head results
        for the last GAT layer.
    activations : list of activation function or None
        ``activations[i]`` gives the activation function applied to the aggregated multi-head
        results for the i-th GAT layer. ``len(activations)`` equals the number of GAT layers.
        By default, ELU is applied for intermediate GAT layers and no activation is applied
        for the last GAT layer.
    biases : list of bool
        ``biases[i]`` gives whether to add bias for the i-th GAT layer. ``len(activations)``
        equals the number of GAT layers. By default, bias is added for all GAT layers.
    classifier_hidden_feats : int
        (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
        in the classifier. Default to 128.
    classifier_dropout : float
        (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
        Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 biases=None, classifier_hidden_feats=128, classifier_dropout=0., n_tasks=1,
                 predictor_hidden_feats=128, predictor_dropout=0.):
        super(GATPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GAT(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       num_heads=num_heads,
                       feat_drops=feat_drops,
                       attn_drops=attn_drops,
                       alphas=alphas,
                       residuals=residuals,
                       agg_modes=agg_modes,
                       activations=activations,
                       biases=biases)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return self.predict(graph_feats)