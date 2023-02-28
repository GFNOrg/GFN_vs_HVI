import numpy as np
import jraph

from dag_gflownet.utils.np_utils import multirange


def to_graphs_tuple(adjacencies, pad=True):
    num_graphs, num_variables = adjacencies.shape[:2]
    n_node = np.full((num_graphs,), num_variables, dtype=np.int_)

    counts, senders, receivers = np.nonzero(adjacencies)
    n_edge = np.bincount(counts, minlength=num_graphs)

    # Node features: node indices
    nodes = np.tile(np.arange(num_variables), num_graphs)
    edges = np.ones_like(senders)

    graphs_tuple =  jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders + counts * num_variables,
        receivers=receivers + counts * num_variables,
        globals=None,
        n_node=n_node,
        n_edge=n_edge,
    )
    if pad:
        graphs_tuple = pad_graph_to_nearest_power_of_two(graphs_tuple)
    return graphs_tuple


def batch_sequences_to_graphs_tuple(num_variables, actions, lengths):
    batch_size, max_length = actions.shape
    # TODO: Pad the episode to be of length the closest power of 2 larger than
    # max(lengths), with a hard maximum at "actions.shape[1]" (d * (d - 1) // 2 + 1)
    # max_length = min(max_length,
    #     _nearest_bigger_power_of_two(np.max(lengths)) + 1)
    max_edges_in_sequence = max_length * (max_length - 1) // 2

    # Number of nodes:
    # The number of nodes in each graph is equal to the number of variables,
    # except the last graph at index "max_length" of each sequence, which is a
    # graph with a single node, used for padding.
    n_node = np.full((batch_size, max_length + 1), num_variables, dtype=np.int_)
    n_node[:, -1] = 1

    # Nodes:
    # The nodes are encoded as their integer index in [0, num_variables - 1].
    # The last graph of each sequence has a single node with index "0".
    nodes = np.zeros((batch_size, num_variables * max_length + 1), dtype=np.int_)
    nodes[:, :-1] = np.tile(np.arange(num_variables), (batch_size, max_length))

    # Number of edges:
    # the number of edges increases by one for each graph in the sequence, for
    # all sequences in the batch, starting from the empty graph, up to the
    # stop action. Then we pad the sequences with empty graphs (where n_edge = 0).
    # The last graph (with a single node) contains as many edges as there are to
    # sum the total number of edges in all graphs of the episodes to "max_edge_in_sequence".
    n_edge = np.tile(np.arange(max_length + 1), (batch_size, 1))
    n_edge = np.where(n_edge <= lengths[:, None], n_edge, 0)
    n_edge[:, -1] = max_edges_in_sequence - np.sum(n_edge, axis=1)

    # Edges:
    # All the edges are encoded with the same embedding (with index "1"). There
    # are a total of "max_edges_in_sequence" edges in all the graphs of a
    # sequence (including the padding graph).
    edges = np.ones((batch_size, max_edges_in_sequence), dtype=np.int_)

    arange = np.arange(1, max_length)
    indices, offsets = multirange(arange), np.repeat(arange, arange)
    senders, receivers = divmod(actions[:, indices], num_variables)

    senders = np.where(offsets <= lengths[:, None],
        senders + offsets * num_variables, max_length * num_variables)
    receivers = np.where(offsets <= lengths[:, None],
        receivers + offsets * num_variables, max_length * num_variables)

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=None,
        n_node=n_node,
        n_edge=n_edge
    )


def _nearest_bigger_power_of_two(x):
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(graphs_tuple):
    # Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(np.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(np.sum(graphs_tuple.n_edge))

    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to)
