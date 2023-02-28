import numpy as np
import networkx as nx
import string

from itertools import chain, product, islice, count, permutations

from numpy.random import default_rng
from pgmpy import models
from pgmpy.factors.continuous import LinearGaussianCPD


def sample_erdos_renyi_graph(
        num_variables,
        p=None,
        num_edges=None,
        nodes=None,
        create_using=models.BayesianNetwork,
        rng=default_rng()
    ):
    if p is None:
        if num_edges is None:
            raise ValueError('One of p or num_edges must be specified.')
        p = num_edges / ((num_variables * (num_variables - 1)) / 2.)
    
    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(
            product(uppercase, repeat=r) for r in count(1))
        nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    adjacency = rng.binomial(1, p=p, size=(num_variables, num_variables))
    adjacency = np.tril(adjacency, k=-1)  # Only keep the lower triangular part

    # Permute the rows and columns
    perm = rng.permutation(num_variables)
    adjacency = adjacency[perm, :]
    adjacency = adjacency[:, perm]

    graph = nx.from_numpy_array(adjacency, create_using=create_using)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    return graph


def sample_erdos_renyi_linear_gaussian(
        num_variables,
        p=None,
        num_edges=None,
        nodes=None,
        loc_edges=0.0,
        scale_edges=1.0,
        obs_noise=0.1,
        rng=default_rng()
    ):
    # Create graph structure
    graph = sample_erdos_renyi_graph(
        num_variables,
        p=p,
        num_edges=num_edges,
        nodes=nodes,
        create_using=models.LinearGaussianBayesianNetwork,
        rng=rng
    )

    # Create the model parameters
    factors = []
    for node in graph.nodes:
        parents = list(graph.predecessors(node))

        # Sample random parameters (from Normal distribution)
        theta = rng.normal(loc_edges, scale_edges, size=(len(parents) + 1,))
        theta[0] = 0.  # There is no bias term

        # Create factor
        factor = LinearGaussianCPD(node, theta, obs_noise, parents)
        factors.append(factor)

    graph.add_cpds(*factors)
    return graph


def _s(node1, node2):
    return (node2, node1) if (node1 > node2) else (node1, node2)

def get_markov_blanket(graph, node):
    parents = set(graph.predecessors(node))
    children = set(graph.successors(node))

    mb_nodes = parents | children
    for child in children:
        mb_nodes |= set(graph.predecessors(child))
    mb_nodes.discard(node)

    return mb_nodes


def get_markov_blanket_graph(graph):
    """Build an undirected graph where two nodes are connected if
    one node is in the Markov blanket of another.
    """
    # Make it a directed graph to control the order of nodes in each
    # edges, to avoid mapping the same edge to 2 entries in mapping.
    mb_graph = nx.DiGraph()
    mb_graph.add_nodes_from(graph.nodes)

    edges = set()
    for node in graph.nodes:
        edges |= set(_s(node, mb_node)
            for mb_node in get_markov_blanket(graph, node))
    mb_graph.add_edges_from(edges)

    return mb_graph


def adjacencies_to_networkx(adjacencies, nodes):
    mapping = dict(enumerate(nodes))
    for adjacency in adjacencies:
        graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        yield nx.relabel_nodes(graph, mapping, copy=False)


def get_valid_actions(graph):
    """Gets the list of valid actions.

    The valid actions correspond to directed edges that can be added to the
    current graph, such that adding any of those edges would still yield a
    DAG. In other words, those are edges that (1) are not already present in
    the graph, and (2) would not introduce a directed cycle.

    Parameters
    ----------
    graph : nx.DiGraph instance
        The current graph.

    Returns
    -------
    edges : set of tuples
        A set of directed edges, encoded as a tuple of nodes from `graph`,
        corresponding to the valid actions in the state `graph`.
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('The input graph is not a valid DAG.')

    all_edges = set(permutations(graph.nodes, 2))
    edges_already_present = set(graph.edges())

    # Build the transitive closure of the transpose graph
    closure = nx.transitive_closure_dag(graph.reverse())
    edges_cycle = set(closure.edges())

    return all_edges - (edges_already_present | edges_cycle)


def valid_actions_to_mask(valid_actions, nodelist):
    """Converts a list of valid actions into a mask matrix.

    Parameters
    ----------
    valid_actions : set of tuples
        A set of directed edges, encoded as a tuple of nodes from `nodes`,
        corresponding to the valid actions.

    nodelist : list
        The list of nodes; this list is required to ensure consistent
        encoding of nodes in the rows and columns of the mask matrix.

    Returns
    -------
    mask : np.ndarray of shape `(num_nodes, num_nodes)`
        The (binary) mask matrix, with value 1 for valid actions, and 0
        otherwise.
    """
    num_nodes = len(nodelist)
    mask = np.zeros((num_nodes, num_nodes), dtype=np.int_)

    for i, source in enumerate(nodelist):
        for j, target in enumerate(nodelist):
            mask[i, j] = int((source, target) in valid_actions)

    return mask


def graph_to_adjacency(graph, nodelist):
    """Converts a graph into its adjacency matrix.
    
    Parameters
    ----------
    graph : nx.DiGraph instance
        The graph.

    nodelist : list
        The list of nodes; this list is required to ensure consistent
        encoding of nodes in the rows and columns of the adjacency matrix.

    Returns
    -------
    adjacency : np.ndarray of shape `(num_nodes, num_nodes)`
        The adjacency matrix of `graph`.
    """
    return nx.to_numpy_array(graph, nodelist, dtype=np.int_, weight=None)


def get_children(graph, gfn_cache, nodelist):
    """Gets all the children of a graph.

    This function returns a set of the next states, from a particular state
    `graph`, with its corresponding log-probability. Note that the set of
    children includes the stop action, encoded as a `None` action, for which
    the child graph is the same as the current graph.

    Parameters
    ----------
    graph : nx.DiGraph instance
        The current graph.

    gfn_cache : dict
        The cache of log-probabilities returned by the GFlowNet. See
        `dag_gflownet.utils.gflownet.get_gflownet_cache` for details.

    nodelist : list
        The list of nodes; this list is required to ensure consistent
        encoding of nodes in the rows and columns of the adjacency matrix.

    Returns
    -------
    children : set of tuples
        The set of all the next state from the current graph, with their
        corresponding log-probability. Each child is represented as
        `(next_graph, action, log_prob)`, where `next_graph` is a nx.DiGraph
        instance, `action` is the edge added (as a tuple of nodes), and
        `log_prob` is the log-probability of this action. Not that the "stop"
        action is encoded as the action `None`.
    """
    node2idx = dict((node, idx) for (idx, node) in enumerate(nodelist))
    valid_actions = get_valid_actions(graph)
    num_variables = len(nodelist)

    log_pi = gfn_cache[frozenset(graph.edges())]
    children = {(graph, None, log_pi[-1])}  # The stop action
    for (source, target) in valid_actions:
        action = node2idx[source] * num_variables + node2idx[target]
        next_graph = graph.copy()
        next_graph.add_edge(source, target)
        children.add((next_graph, (source, target), log_pi[action]))

    return children
