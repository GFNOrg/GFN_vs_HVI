import sys
import networkx as nx

from functools import lru_cache
from collections import namedtuple
from abc import ABC, abstractmethod
from pgmpy.estimators import StructureScore

LocalScore = namedtuple('LocalScore', ['key', 'score', 'prior'])

class BaseScore(ABC, StructureScore):
    """Base class for the scorer.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset.

    prior : `BasePrior` instance
        The prior over graphs p(G).
    """
    def __init__(self, data, prior):
        self.data = data
        self.prior = prior
        self.column_names = list(data.columns)
        self.column_names_to_idx = dict((name, idx)
            for (idx, name) in enumerate(self.column_names))
        self.num_variables = len(self.column_names)
        self.prior.num_variables = self.num_variables
        self._cache_local_scores = None

    def __call__(self, index, in_queue, out_queue, error_queue):
        try:
            while True:
                data = in_queue.get()
                if data is None:
                    break

                target, indices, indices_after = data
                local_score_before, local_score_after = self.get_local_scores(
                    target, indices, indices_after=indices_after)

                out_queue.put((True, *local_score_after))
                if local_score_before is not None:
                    out_queue.put((True, *local_score_before))

        except (KeyboardInterrupt, Exception):
            error_queue.put((index,) + sys.exc_info()[:2])
            out_queue.put((False, None, None, None))

    @abstractmethod
    def get_local_scores(self, target, indices, indices_after=None):
        pass

    @property
    def cache_local_scores(self):
        if self._cache_local_scores is None:
            self._cache_local_scores = lru_cache()(self.get_local_scores)
        return self._cache_local_scores

    def score(self, graph):
        graph = nx.relabel_nodes(graph, self.column_names_to_idx)
        score = 0
        for node in graph.nodes():
            _, local_score = self.cache_local_scores(
                node, tuple(graph.predecessors(node)))
            score += local_score.score + local_score.prior
        return score


class BasePrior(ABC):
    """Base class for the prior over graphs p(G).
    
    Any subclass of `BasePrior` must return the contribution of log p(G) for a
    given variable with `num_parents` parents. We assume that the prior is modular.
    
    Parameters
    ----------
    num_variables : int (optional)
        The number of variables in the graph. If not specified, this gets
        populated inside the scorer class.
    """
    def __init__(self, num_variables=None):
        self._num_variables = num_variables
        self._log_prior = None

    def __call__(self, num_parents):
        return self.log_prior[num_parents]

    @property
    @abstractmethod
    def log_prior(self):
        pass

    @property
    def num_variables(self):
        if self._num_variables is None:
            raise RuntimeError('The number of variables is not defined.')
        return self._num_variables

    @num_variables.setter
    def num_variables(self, value):
        self._num_variables = value
