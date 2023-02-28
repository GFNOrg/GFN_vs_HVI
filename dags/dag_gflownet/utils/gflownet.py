import numpy as np
import jax.numpy as jnp
import optax

from tqdm.auto import trange
from jax import nn, lax, jit

from dag_gflownet.utils.jraph_utils import to_graphs_tuple
from dag_gflownet.utils.exhaustive import all_dags
from dag_gflownet.utils.graph import (get_valid_actions, graph_to_adjacency,
    valid_actions_to_mask)


MASKED_VALUE = -1e5


def mask_logits(logits, masks):
    return masks * logits + (1. - masks) * MASKED_VALUE


def detailed_balance_loss(
        log_pi_t,
        log_pi_tp1,
        actions,
        delta_scores,
        num_edges,
        delta=1.
    ):
    r"""Detailed balance loss.

    This function computes the detailed balance loss, in the specific case
    where all the states are complete. This loss function is given by:

    $$ L(\theta; s_{t}, s_{t+1}) = \left[\log\frac{
        R(s_{t+1})P_{B}(s_{t} \mid s_{t+1})P_{\theta}(s_{f} \mid s_{t})}{
        R(s_{t})P_{\theta}(s_{t+1} \mid s_{t})P_{\theta}(s_{f} \mid s_{t+1})
    }\right]^{2} $$

    In practice, to avoid gradient explosion, we use the Huber loss instead
    of the L2-loss (the L2-loss can be emulated with a large value of delta).
    Moreover, we do not backpropagate the error through $P_{\theta}(s_{f} \mid s_{t+1})$,
    which is computed using a target network.

    Parameters
    ----------
    log_pi_t : jnp.DeviceArray
        The log-probabilities $\log P_{\theta}(s' \mid s_{t})$, for all the
        next states $s'$, including the terminal state $s_{f}$. This array
        has size `(B, N ** 2 + 1)`, where `B` is the batch-size, and `N` is
        the number of variables in a graph.

    log_pi_tp1 : jnp.DeviceArray
        The log-probabilities $\log P_{\theta}(s' \mid s_{t+1})$, for all the
        next states $s'$, including the terminal state $s_{f}$. This array
        has size `(B, N ** 2 + 1)`, where `B` is the batch-size, and `N` is
        the number of variables in a graph. In practice, `log_pi_tp1` is
        computed using a target network with parameters $\theta'$.

    actions : jnp.DeviceArray
        The actions taken to go from state $s_{t}$ to state $s_{t+1}$. This
        array has size `(B, 1)`, where `B` is the batch-size.

    delta_scores : jnp.DeviceArray
        The delta-scores between state $s_{t}$ and state $s_{t+1}$, given by
        $\log R(s_{t+1}) - \log R(s_{t})$. This array has size `(B, 1)`, where
        `B` is the batch-size.

    num_edges : jnp.DeviceArray
        The number of edges in $s_{t}$. This array has size `(B, 1)`, where `B`
        is the batch-size.

    delta : float (default: 1.)
        The value of delta for the Huber loss.

    Returns
    -------
    loss : jnp.DeviceArray
        The detailed balance loss averaged over a batch of samples.

    logs : dict
        Additional information for logging purposes.
    """
    # Compute the forward log-probabilities
    log_pF = jnp.take_along_axis(log_pi_t, actions, axis=-1)

    # Compute the backward log-probabilities
    log_pB = -jnp.log1p(num_edges)

    error = (jnp.squeeze(delta_scores + log_pB - log_pF, axis=-1)
        + log_pi_t[:, -1] - log_pi_tp1[:, -1])
    loss = jnp.mean(optax.huber_loss(error, delta=delta))

    logs = {
        'error': error,
        'loss': loss,
    }
    return (loss, logs)


def trajectory_balance_loss(
        log_pi,
        log_Z,
        actions,
        log_rewards,
        num_edges,
    ):
    # Mask the log-probabilities, based on the sequence lengths
    seq_masks = (jnp.arange(log_pi.shape[1]) <= num_edges[:, None])
    log_pi = jnp.where(seq_masks[..., None], log_pi, 0.)

    # Compute the forward log-probabilities
    log_pF = jnp.take_along_axis(log_pi, actions[..., None], axis=-1)
    log_pF = jnp.sum(log_pF, axis=(1, 2))

    # Compute the backward log-probabilities (fixed p_B)
    log_pB = -lax.lgamma(num_edges + 1.)  # -log(n!)

    error = log_Z + log_pF - log_rewards - log_pB
    loss = jnp.mean(optax.l2_loss(error))

    logs = {
        'error': error,
        'loss': loss,
    }
    return (loss, logs)


def log_policy(logits, stop, masks):
    masks = masks.reshape(logits.shape)
    masked_logits = mask_logits(logits, masks)
    can_continue = jnp.any(masks, axis=-1, keepdims=True)

    logp_continue = (nn.log_sigmoid(-stop)
        + nn.log_softmax(masked_logits, axis=-1))
    logp_stop = nn.log_sigmoid(stop)

    # In case there is no valid action other than stop
    logp_continue = jnp.where(can_continue, logp_continue, MASKED_VALUE)
    logp_stop = logp_stop * can_continue

    return jnp.concatenate((logp_continue, logp_stop), axis=-1)


def uniform_log_policy(masks, stop_gamma):
    masks = masks.reshape(masks.shape[0], -1)
    num_edges = jnp.sum(masks, axis=-1, keepdims=True)

    if stop_gamma > 0:
        logp_action = jnp.log1p(-stop_gamma) - jnp.log(num_edges)
        logp_stop = jnp.full((masks.shape[0], 1), jnp.log(stop_gamma))
        logp_continue = mask_logits(logp_action, masks)
    else:
        logp_stop = -jnp.log1p(num_edges)
        logp_continue = mask_logits(logp_stop, masks)

    return jnp.concatenate((logp_continue, logp_stop), axis=-1)

def posterior_estimate(
        gflownet,
        params,
        env,
        key,
        num_samples=1000,
        verbose=True,
        **kwargs
    ):
    """Get the posterior estimate of DAG-GFlowNet as a collection of graphs
    sampled from the GFlowNet.

    Parameters
    ----------
    gflownet : `DAGGFlowNet` instance
        Instance of a DAG-GFlowNet.

    params : dict
        Parameters of the neural network for DAG-GFlowNet. This must be a dict
        that can be accepted by the Haiku model in the `DAGGFlowNet` instance.

    env : `GFlowNetDAGEnv` instance
        Instance of the environment.

    key : jax.random.PRNGKey
        Random key for sampling from DAG-GFlowNet.

    num_samples : int (default: 1000)
        The number of samples in the posterior approximation.

    verbose : bool
        If True, display a progress bar for the sampling process.

    Returns
    -------
    posterior : np.ndarray instance
        The posterior approximation, given as a collection of adjacency matrices
        from graphs sampled with the posterior approximation. This array has
        size `(B, N, N)`, where `B` is the number of sample graphs in the
        posterior approximation, and `N` is the number of variables in a graph.

    logs : dict
        Additional information for logging purposes.
    """
    samples = []
    observations = env.reset()
    with trange(num_samples, disable=(not verbose), **kwargs) as pbar:
        while len(samples) < num_samples:
            order = observations['order']
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            actions, key, _ = gflownet.act(params, key, observations, 1.)
            observations, _, dones, _ = env.step(np.asarray(actions))

            samples.extend([order[i] for i, done in enumerate(dones) if done])
            pbar.update(min(num_samples - pbar.n, np.sum(dones).item()))
    orders = np.stack(samples[:num_samples], axis=0)
    logs = {
        'orders': orders,
    }
    return ((orders >= 0).astype(np.int_), logs)


def get_gflownet_cache(gflownet, params, nodelist, batch_size=256):
    """Cache the results of the GFlowNet for all the states.

    This function caches the log-probabilities for all the actions and for
    all the states of the GFlowNet.

    Parameters
    ----------
    gflownet : DAGGFlowNet instance
        The GFlowNet class, containing the definition of the model.

    params : GFNParameters instance
        The parameters of the GFlowNet (i.e., the parameters of the model, as
        well as log(Z)).

    nodelist : list
        The list of nodes; this list is required to ensure consistent
        encoding of nodes in the rows and columns of the adjacency matrix.

    batch_size : int
        The batch-size of the inputs of the GFlowNet.

    Returns
    -------
    cache : dict of (frozenset, np.ndarray)
        The cache of log-probabilities returned by the GFlowNet. The keys of
        the cache are the graphs (encoded as a frozenset of their edges), and
        the corresponding value is an array of size `(num_variables ** 2 + 1,)`
        containing the log-probabilities of all the actions in that state
        (including the "stop" action, at the last index).
    """
    gfn_apply = jit(gflownet.model.apply)
    cache = dict()

    graphs, adjacencies, masks = [], [], []
    for graph in all_dags(len(nodelist), nodelist=nodelist):
        valid_actions = get_valid_actions(graph)

        graphs.append(graph)
        adjacencies.append(graph_to_adjacency(graph, nodelist))
        masks.append(valid_actions_to_mask(valid_actions, nodelist))

        if len(graphs) >= batch_size:
            adjacencies = np.stack(adjacencies, axis=0).astype(np.float32)
            masks = np.stack(masks, axis=0).astype(np.float32)
            graph_tuples = to_graphs_tuple(adjacencies)
            log_pis = gfn_apply(params.model, graph_tuples, masks)

            for graph, log_pi in zip(graphs, np.asarray(log_pis)):
                cache[frozenset(graph.edges())] = log_pi

            # Clear batch
            graphs, adjacencies, masks = [], [], []

    if graphs:
        adjacencies = np.stack(adjacencies, axis=0).astype(np.float32)
        masks = np.stack(masks, axis=0).astype(np.float32)
        graph_tuples = to_graphs_tuple(adjacencies)
        log_pis = gfn_apply(params.model, graph_tuples, masks)

        for graph, log_pi in zip(graphs, np.asarray(log_pis)):
            cache[frozenset(graph.edges())] = log_pi

    return cache
