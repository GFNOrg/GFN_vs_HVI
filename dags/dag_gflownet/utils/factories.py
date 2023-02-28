from numpy.random import default_rng

from dag_gflownet.scores import BGeScore, priors
from dag_gflownet.utils.data import get_data
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.episodic_replay_buffer import EpisodicReplayBuffer


def get_prior(name, **kwargs):
    prior = {
        'uniform': priors.UniformPrior,
        'erdos_renyi': priors.ErdosRenyiPrior,
        'edge': priors.EdgePrior,
        'fair': priors.FairPrior
    }
    return prior[name](**kwargs)


def get_scorer(args, rng=default_rng()):
    # Get the data
    graph, data, score = get_data(args.graph, args, rng=rng)

    # Get the prior
    prior = get_prior(args.prior, **args.prior_kwargs)

    # Get the scorer
    scores = {'bge': BGeScore}
    scorer = scores[score](data, prior, **args.scorer_kwargs)

    return scorer, data, graph


def get_replay_cls(loss_name):
    if loss_name == 'db':
        cls_ = ReplayBuffer
    else:
        cls_ = EpisodicReplayBuffer
    return cls_
