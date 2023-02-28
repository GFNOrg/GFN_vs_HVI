import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
import jax
import os
import math

from tqdm import trange
from numpy.random import default_rng

from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet, GFNParameters
from dag_gflownet.utils.factories import get_scorer, get_replay_cls
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.metrics import get_log_features
from dag_gflownet.utils.jraph_utils import to_graphs_tuple
from dag_gflownet.utils.exhaustive import (get_full_posterior,
    get_edge_log_features, get_path_log_features, get_markov_blanket_log_features)


def main(args):
    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    # Create the environment
    scorer, data, graph = get_scorer(args, rng=rng)
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer,
        num_workers=args.num_workers,
        context=args.mp_context
    )

    # Create the replay buffer
    replay_cls = get_replay_cls(args.loss)
    replay = replay_cls(
        args.replay_capacity,
        num_variables=env.num_variables,
    )

    # Create the GFlowNet & initialize parameters
    gflownet = DAGGFlowNet(loss=args.loss, delta=args.delta, stop_gamma=args.stop_gamma)
    optimizer = optax.multi_transform({
        'log_Z': optax.sgd(args.lr_logZ, momentum=args.momentum_logZ),
        'model': optax.adam(args.lr),
    }, GFNParameters(model='model', log_Z='log_Z'))
    params, state = gflownet.init(
        subkey,
        optimizer,
        replay.dummy['graphs'],
        replay.dummy['masks'],
    )

    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - args.min_exploration),
        transition_steps=args.num_iterations // 2,
        transition_begin=args.prefill,
    ))

    # For small enough graphs, evaluate the full posterior
    if env.num_variables < 6:
        full_posterior = get_full_posterior(data, scorer, verbose=True)

        full_edge_log_features = get_edge_log_features(full_posterior)
        full_path_log_features = get_path_log_features(full_posterior)
        full_markov_log_features = get_markov_blanket_log_features(full_posterior)

    # Training loop
    indices = None
    observations = env.reset()
    with trange(args.prefill + args.num_iterations, desc='Training') as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            epsilon = exploration_schedule(iteration)
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            actions, key, logs = gflownet.act(params, key, observations, epsilon)
            next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
            indices = replay.add(
                observations,
                actions,
                logs,
                next_observations,
                delta_scores,
                dones,
                indices=indices
            )
            observations = next_observations

            if iteration >= args.prefill:
                # Update the parameters of the GFlowNet
                if replay.can_sample(args.batch_size):
                    samples = replay.sample(batch_size=args.batch_size, rng=rng)
                    params, state, logs = gflownet.step(params, state, samples)

                pbar.set_postfix(loss=f"{logs['loss']:.2f}", epsilon=f"{epsilon:.2f}", log_Z=f"{params.log_Z:.3f}")

    # Evaluate the posterior estimate
    posterior, _ = posterior_estimate(
        gflownet,
        params,
        env,
        key,
        num_samples=args.num_samples_posterior,
        desc='Sampling from posterior'
    )

    # Compute the metrics
    ground_truth = nx.to_numpy_array(graph, weight=None)

    if env.num_variables < 6:
        log_features = get_log_features(posterior, data.columns)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser(description='DAG-GFlowNet for Strucure Learning (Off-Policy training).')

    # Environment
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=8,
        help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--scorer_kwargs', type=json.loads, default='{}',
        help='Arguments of the scorer.')
    environment.add_argument('--prior', type=str, default='uniform',
        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
        help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--prior_kwargs', type=json.loads, default='{}',
        help='Arguments of the prior over graphs.')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-5,
        help='Learning rate (default: %(default)s)')
    optimization.add_argument('--lr_logZ', type=float, default=1e-1,
        help='Learning rate for log(Z) (default: %(default)s)')
    optimization.add_argument('--momentum_logZ', type=float, default=0.8,
        help='Momentum for log(Z) (default: %(default)s)')
    optimization.add_argument('--delta', type=float, default=1.,
        help='Value of delta for Huber loss (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=32,
        help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=100_000,
        help='Number of iterations (default: %(default)s)')
    optimization.add_argument('--loss', type=str, default='db',
        choices=['db', 'tb_off_policy', 'hvi_off_policy'],
        help='Type of loss (default: %(default)s)')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=100_000,
        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=1000,
        help='Number of iterations with a random policy to prefill '
             'the replay buffer (default: %(default)s)')
    
    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,
        help='Minimum value of epsilon-exploration (default: %(default)s)')
    exploration.add_argument('--stop_gamma', type=float, default=0.,
        help='How often termination is trigger during exploration (default: %(default)s)')
    
    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--num_samples_posterior', type=int, default=1000,
        help='Number of samples for the posterior estimate (default: %(default)s)')
    misc.add_argument('--seed', type=int, default=0,
        help='Random seed (default: %(default)s)')
    misc.add_argument('--num_workers', type=int, default=4,
        help='Number of workers (default: %(default)s)')
    misc.add_argument('--mp_context', type=str, default='spawn',
        help='Multiprocessing context (default: %(default)s)')
    misc.add_argument('--log_every', type=int, default=50,
        help='Frequency for logging (default: %(default)s)')
    misc.add_argument('--log_posterior_every', type=int, default=1000,
        help='Frequency of evaluation of the posterior estimate (default: %(default)s)')

    subparsers = parser.add_subparsers(help='Type of graph', dest='graph')

    # Erdos-Renyi Linear-Gaussian graphs
    er_lingauss = subparsers.add_parser('erdos_renyi_lingauss')
    er_lingauss.add_argument('--num_variables', type=int, required=True,
        help='Number of variables')
    er_lingauss.add_argument('--num_edges', type=int, required=True,
        help='Average number of edges')
    er_lingauss.add_argument('--num_samples', type=int, required=True,
        help='Number of samples')

    args = parser.parse_args()

    main(args)
