import jax.numpy as jnp
import haiku as hk
import optax

from collections import namedtuple
from functools import partial
from jax import grad, random, jit, vmap

from dag_gflownet.nets.gnn.gflownet import gflownet
from dag_gflownet.utils.gflownet import (uniform_log_policy,
    detailed_balance_loss, trajectory_balance_loss)
from dag_gflownet.utils.hvi import hvi_off_policy_loss, hvi_on_policy_loss
from dag_gflownet.utils.jnp_utils import batch_random_choice


GFNParameters = namedtuple('GFNParameters', ['model', 'log_Z'])


class DAGGFlowNet:
    """DAG-GFlowNet.

    Parameters
    ----------
    model : callable (optional)
        The neural network for the GFlowNet. The model must be a callable
        to feed into hk.transform, that takes a single adjacency matrix and
        a single mask as an input. Default to an architecture based on
        Linear Transformers.

    delta : float (default: 1.)
        The value of delta for the Huber loss used in the detailed balance
        loss (in place of the L2 loss) to avoid gradient explosion.
    """
    def __init__(self, model=None, loss='db', delta=1., stop_gamma=0.):
        if model is None:
            model = gflownet

        self.model = hk.without_apply_rng(hk.transform(model))
        self._loss = loss
        self.delta = delta
        self.stop_gamma = stop_gamma

        self._optimizer = None

    def loss(self, params, samples):
        if self._loss == 'db':
            log_pi_t = self.model.apply(
                params.model, samples['graphs'], samples['masks'])
            log_pi_tp1 = self.model.apply(
                params.model, samples['next_graphs'], samples['next_masks'])

            loss, logs = detailed_balance_loss(
                log_pi_t,
                log_pi_tp1,
                samples['actions'],
                samples['delta_scores'],
                samples['num_edges'],
                delta=self.delta
            )
        else:
            v_model = vmap(self.model.apply, in_axes=(None, 0, 0))
            log_pi = v_model(params.model, samples['graphs'], samples['masks'])

            if self._loss.startswith('tb'):
                loss, logs = trajectory_balance_loss(
                    log_pi,
                    params.log_Z,
                    samples['actions'],
                    samples['scores'],
                    samples['num_edges'],
                )
            elif self._loss == 'hvi_off_policy':
                loss, logs = hvi_off_policy_loss(
                    log_pi,
                    samples['log_pi'],
                    samples['actions'],
                    samples['scores'],
                    samples['num_edges'],
                )
            elif self._loss == 'hvi_on_policy':
                loss, logs = hvi_on_policy_loss(
                    log_pi,
                    samples['actions'],
                    samples['scores'],
                    samples['num_edges'],
                )
            else:
                raise ValueError(f'Unknown loss function: {self._loss}')
        return (loss, logs)

    @partial(jit, static_argnums=(0,))
    def act(self, params, key, observations, epsilon):
        masks = observations['mask'].astype(jnp.float32)
        graphs = observations['graph']
        batch_size = masks.shape[0]
        key, subkey1, subkey2 = random.split(key, 3)

        # Get the GFlowNet policy
        log_pi = self.model.apply(params.model, graphs, masks)

        # Get uniform policy
        log_uniform = uniform_log_policy(masks, self.stop_gamma)

        # Mixture of GFlowNet policy and uniform policy
        is_exploration = random.bernoulli(
            subkey1, p=1. - epsilon, shape=(batch_size, 1))
        log_pi = jnp.where(is_exploration, log_uniform, log_pi)

        # Sample actions
        actions = batch_random_choice(subkey2, jnp.exp(log_pi), masks)
        log_pi_actions = jnp.take_along_axis(log_pi, actions[..., None], axis=1)

        logs = {
            'is_exploration': is_exploration.astype(jnp.int32),
            'log_pi': jnp.squeeze(log_pi_actions, axis=1),
        }
        return (actions, key, logs)

    @partial(jit, static_argnums=(0,))
    def step(self, params, state, samples):
        grads, logs = grad(self.loss, has_aux=True)(params, samples)

        # Update the online params
        updates, state = self.optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)

        return (params, state, logs)

    def init(self, key, optimizer, graph, mask):
        # Set the optimizer
        self._optimizer = optax.chain(optimizer, optax.zero_nans())
        params = GFNParameters(
            model=self.model.init(key, graph, mask),
            log_Z=jnp.array(0.),
        )
        state = self.optimizer.init(params)
        return (params, state)

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise RuntimeError('The optimizer is not defined. To train the '
                               'GFlowNet, you must call `DAGGFlowNet.init` first.')
        return self._optimizer
