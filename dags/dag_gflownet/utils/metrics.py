"""
The code is adapted from:
https://github.com/larslorch/dibs/blob/master/dibs/metrics.py

MIT License

Copyright (c) 2021 Lars Lorch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import math
import networkx as nx

from scipy import stats
from tqdm.auto import tqdm
from sklearn import metrics
from collections import namedtuple
from itertools import permutations, combinations

from dag_gflownet.utils.graph import adjacencies_to_networkx, get_markov_blanket_graph


def expected_shd(posterior, ground_truth):
    """Compute the Expected Structural Hamming Distance.

    This function computes the Expected SHD between a posterior approximation
    given as a collection of samples from the posterior, and the ground-truth
    graph used in the original data generation process.

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    ground_truth : np.ndarray instance
        Adjacency matrix of the ground-truth graph. The array must have size
        `(N, N)`, where `N` is the number of variables in the graph.

    Returns
    -------
    e_shd : float
        The Expected SHD.
    """
    # Compute the pairwise differences
    diff = np.abs(posterior - np.expand_dims(ground_truth, axis=0))
    diff = diff + diff.transpose((0, 2, 1))

    # Ignore double edges
    diff = np.minimum(diff, 1)
    shds = np.sum(diff, axis=(1, 2)) / 2

    return np.mean(shds)


def expected_edges(posterior):
    """Compute the expected number of edges.

    This function computes the expected number of edges in graphs sampled from
    the posterior approximation.

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    Returns
    -------
    e_edges : float
        The expected number of edges.
    """
    num_edges = np.sum(posterior, axis=(1, 2))
    return np.mean(num_edges)


def threshold_metrics(posterior, ground_truth):
    """Compute threshold metrics (e.g. AUROC, Precision, Recall, etc...).

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    ground_truth : np.ndarray instance
        Adjacency matrix of the ground-truth graph. The array must have size
        `(N, N)`, where `N` is the number of variables in the graph.

    Returns
    -------
    metrics : dict
        The threshold metrics.
    """
    # Expected marginal edge features
    p_edge = np.mean(posterior, axis=0)
    p_edge_flat = p_edge.reshape(-1)
    
    gt_flat = ground_truth.reshape(-1)

    # Threshold metrics 
    fpr, tpr, _ = metrics.roc_curve(gt_flat, p_edge_flat)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(gt_flat, p_edge_flat)
    prc_auc = metrics.auc(recall, precision)
    ave_prec = metrics.average_precision_score(gt_flat, p_edge_flat)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'prc_auc': prc_auc,
        'ave_prec': ave_prec,
    }

Features = namedtuple('Features', ['edge', 'path', 'markov_blanket'])

def get_log_features(posterior, nodes, verbose=True):
    """Compute the log-features for edges, paths & Markov blankets."""
    features = Features(edge={}, path={}, markov_blanket={})
    num_samples = posterior.shape[0]

    # Initialize the features
    for edge in permutations(nodes, 2):
        features.edge[edge] = 0.
        features.path[edge] = 0.
    for edge in combinations(nodes, 2):
        features.markov_blanket[edge] = 0.

    for graph in tqdm(adjacencies_to_networkx(posterior, nodes),
            total=num_samples, disable=(not verbose)):
        # Get edge features
        for edge in graph.edges:
            features.edge[edge] += 1.

        # Get path features
        closure = nx.transitive_closure_dag(graph)
        for edge in closure.edges:
            features.path[edge] += 1.

        # Get Markov blanket features
        mb = get_markov_blanket_graph(graph)
        for edge in mb.edges:
            features.markov_blanket[edge] += 1.

    return Features(
        edge=dict((key, math.log(value) - math.log(num_samples) if value else -float('inf'))
            for (key, value) in features.edge.items()),
        path=dict((key, math.log(value) - math.log(num_samples) if value else -float('inf'))
            for (key, value) in features.path.items()),
        markov_blanket=dict((key, math.log(value) - math.log(num_samples) if value else -float('inf'))
            for (key, value) in features.markov_blanket.items())
    )


def corr_features(title, full_log_features, estimate_log_features):
    full_data, estimate_data = [], []
    for key in (full_log_features.keys() & estimate_log_features.keys()):
        full_data.append(full_log_features[key])
        estimate_data.append(estimate_log_features[key])
    full_data, estimate_data = np.asarray(full_data), np.asarray(estimate_data)

    logs = {
        f'{title}/rmse': metrics.mean_squared_error(np.exp(full_data), np.exp(estimate_data), squared=False),
        f'{title}/pearson_r': stats.pearsonr(np.exp(full_data), np.exp(estimate_data))[0],
    }

    if np.any(np.isinf(estimate_data)):
        logs.update({
            f'{title}/log/rmse': np.nan,
            f'{title}/log/pearson_r': np.nan,
        })
    else:
        logs.update({
            f'{title}/log/rmse': metrics.mean_squared_error(full_data, estimate_data, squared=False),
            f'{title}/log/pearson_r': stats.pearsonr(full_data, estimate_data)[0],
        })

    return logs


def jensen_shannon_divergence(full_posterior, posterior):
    # Convert to dictionaries to align distributions
    full_posterior_dict = full_posterior.to_dict()
    posterior_dict = posterior.to_dict()

    # Get an (arbitrary ordering of the graphs)
    graphs = list(full_posterior_dict.keys())
    graphs = sorted(graphs, key=len)

    # Get the two distributions aligned
    full_posterior, posterior = [], []
    for graph in graphs:
        full_posterior.append(full_posterior_dict[graph])
        posterior.append(posterior_dict[graph])
    full_posterior = np.array(full_posterior, dtype=np.float_)
    posterior = np.array(posterior, dtype=np.float_)

    # Compute the mean distribution
    mean = np.log(0.5) + np.logaddexp(full_posterior, posterior)

    # Compute the JSD
    KL_full_posterior = np.exp(full_posterior) * (full_posterior - mean)
    KL_posterior = np.exp(posterior) * (posterior - mean)
    return 0.5 * np.sum(KL_full_posterior + KL_posterior)
