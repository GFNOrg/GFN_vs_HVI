import numpy as np
import networkx as nx

from scipy.special import logsumexp
from dataclasses import dataclass
from collections import defaultdict
from pgmpy.estimators import ExhaustiveSearch
from pgmpy.utils.mathext import powerset
from itertools import permutations, product
from tqdm import tqdm

from dag_gflownet.utils.graph import get_markov_blanket_graph

# https://oeis.org/A003024
NUM_DAGS = [1, 1, 3, 25, 543, 29281, 3781503]

class GraphCollection:
    def __init__(self):
        self.edges, self.lengths = [], []
        self.mapping = defaultdict(int)
        self.mapping.default_factory = lambda: len(self.mapping)

    def append(self, graph):
        self.edges.extend([self.mapping[edge] for edge in graph.edges()])
        self.lengths.append(graph.number_of_edges())

    def freeze(self):
        self.edges = np.asarray(self.edges, dtype=np.int_)
        self.lengths = np.asarray(self.lengths, dtype=np.int_)
        self.mapping = [edge for (edge, _)
            in sorted(self.mapping.items(), key=lambda x: x[1])]
        return self

    def is_frozen(self):
        return isinstance(self.mapping, list)
    
    def to_dict(self, prefix=None):
        prefix = f'{prefix}_' if (prefix is not None) else ''
        return ({
            f'{prefix}edges': self.edges,
            f'{prefix}lengths': self.lengths,
            f'{prefix}mapping': self.mapping
        })

    def load(self, edges, lengths, mapping):
        self.edges = edges
        self.lengths = lengths
        self.mapping = dict((tuple(edge), idx) for (idx, edge) in enumerate(mapping))
        return self.freeze()


@dataclass
class FullPosterior:
    log_probas: np.ndarray
    graphs: GraphCollection
    closures: GraphCollection
    markov: GraphCollection

    def to_dict(self):
        # Ensure that "graphs" has been frozen
        if not self.graphs.is_frozen():
            raise ValueError('Graphs must be frozen. Call "graphs.freeze()".')

        offset, output = 0, dict()
        for length, log_prob in zip(self.graphs.lengths, self.log_probas):
            edges_indices = self.graphs.edges[offset:offset + length]
            edges = [self.graphs.mapping[idx] for idx in edges_indices]
            output[frozenset(edges)] = log_prob
            offset += length

        return output

    def save(self, filename):
        with open(filename, 'wb') as f:
            np.savez(f, log_probas=self.log_probas,
                **self.graphs.to_dict(prefix='graphs'),
                **self.closures.to_dict(prefix='closures'),
                **self.markov.to_dict(prefix='markov')
            )

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
            log_probas = data['log_probas']
            graphs = GraphCollection().load(
                data['graphs_edges'],
                data['graphs_lengths'],
                data['graphs_mapping']
            )
            closures = GraphCollection().load(
                data['closures_edges'],
                data['closures_lengths'],
                data['closures_mapping']
            )
            markov = GraphCollection().load(
                data['markov_edges'],
                data['markov_lengths'],
                data['markov_mapping']
            )
        return cls(
            log_probas=log_probas,
            graphs=graphs,
            closures=closures,
            markov=markov
        )


def get_full_posterior(data, scorer, verbose=True):
    estimator = ExhaustiveSearch(data, scoring_method=scorer, use_cache=False)

    log_probas = []
    graphs = GraphCollection()
    closures = GraphCollection()
    markov = GraphCollection()
    with tqdm(estimator.all_dags(), 
            total=NUM_DAGS[data.shape[1]], disable=(not verbose)) as pbar:
        for graph in pbar:  # Enumerate all possible DAGs
            score = estimator.scoring_method.score(graph)
            log_probas.append(score)

            graphs.append(graph)
            closures.append(nx.transitive_closure_dag(graph))
            markov.append(get_markov_blanket_graph(graph))

    # Normalize the log-joint distribution to get the posterior
    log_probas = np.asarray(log_probas, dtype=np.float_)
    log_probas -= logsumexp(log_probas)

    return FullPosterior(
        log_probas=log_probas,
        graphs=graphs.freeze(),
        closures=closures.freeze(),
        markov=markov.freeze()
    )


def get_gfn_exact_posterior(gfn_state_graph, verbose=True):
    # Get the source graph
    in_degrees = gfn_state_graph.in_degree(gfn_state_graph)
    source_graphs = [gfn_state_graph.nodes[node]['graph'] for node, in_degree
        in in_degrees if in_degree == 0]
    assert len(source_graphs) == 1
    assert len(source_graphs[0].edges) == 0
    num_variables = len(source_graphs[0])

    log_probas = []
    graphs = GraphCollection()
    closures = GraphCollection()
    markov = GraphCollection()

    for node in tqdm(nx.topological_sort(gfn_state_graph),
            total=NUM_DAGS[num_variables], disable=(not verbose)):
        graph = gfn_state_graph.nodes[node]['graph']
        log_probas.append(gfn_state_graph.nodes[node]['terminal_log_flow'])

        graphs.append(graph)
        closures.append(nx.transitive_closure_dag(graph))
        markov.append(get_markov_blanket_graph(graph))

    # The log-posterior is already normalized
    log_probas = np.asarray(log_probas, dtype=np.float_)

    return FullPosterior(
        log_probas,
        graphs=graphs.freeze(),
        closures=closures.freeze(),
        markov=markov.freeze()
    )


def _get_log_features(graphs, log_probas):
    indices = np.zeros_like(graphs.lengths)
    indices[1:] = np.cumsum(graphs.lengths[:-1])

    features = dict()
    for index, edge in enumerate(graphs.mapping):
        if not np.any(graphs.edges == index):
            continue
        has_feat = np.add.reduceat(graphs.edges == index, indices)

        # Edge case: the first graph is the empty graph, it has no edge
        if graphs.lengths[0] == 0:
            has_feat[0] = 0
        assert np.sum(graphs.edges == index) == np.sum(has_feat)
        
        has_feat = has_feat.astype(np.bool_)
        features[edge] = logsumexp(log_probas[has_feat])

    return features

def get_edge_log_features(posterior):
    return _get_log_features(posterior.graphs, posterior.log_probas)


def get_path_log_features(posterior):
    return _get_log_features(posterior.closures, posterior.log_probas)


def get_markov_blanket_log_features(posterior):
    return _get_log_features(posterior.markov, posterior.log_probas)


def all_dags(num_variables, nodelist=None):
    # Adapted from: https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/estimators/ExhaustiveSearch.py
    if nodelist is None:
        nodelist = list(range(num_variables))
    edges = list(permutations(nodelist, 2))  # n*(n-1) possible directed edges
    all_graphs = powerset(edges)  # 2^(n*(n-1)) graphs

    for graph_edges in all_graphs:
        graph = nx.DiGraph(graph_edges)
        graph.add_nodes_from(nodelist)
        if nx.is_directed_acyclic_graph(graph):
            yield graph


def all_hashes(num_variables):
    hashes = {
        edge: 2 ** i
        for (i, edge)
        in enumerate(product(range(num_variables), repeat=2))
    }
    for graph in all_dags(num_variables):
        yield sum(hashes[edge] for edge in graph.edges)
