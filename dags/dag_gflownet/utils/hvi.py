import jax.numpy as jnp

from jax import lax, nn


def hvi_off_policy_loss(
        log_pi,
        log_behavior,
        actions,
        log_rewards,
        num_edges,
    ):
    batch_size, max_length = log_pi.shape[:2]

    # Compute the forward log-probabilities
    log_pF = jnp.take_along_axis(log_pi, actions[..., None], axis=-1)
    log_pF = jnp.squeeze(log_pF, axis=2)

    # Compute the backward log-probabilities (fixed p_B)
    log_pB = -jnp.log1p(jnp.arange(max_length))
    log_pB = jnp.repeat(log_pB[None, :], batch_size, axis=0)
    log_pB = log_pB.at[jnp.arange(batch_size), num_edges].set(log_rewards)

    # Mask the padding in each sequence
    seq_masks = (jnp.arange(max_length) <= num_edges[:, None])
    log_pF = jnp.where(seq_masks, log_pF, 0.)
    log_pB = jnp.where(seq_masks, log_pB, 0.)
    log_behavior = jnp.where(seq_masks, log_behavior, 0.)

    # Compute the returns for Off-Policy Policy Gradient
    hvi_returns = jnp.sum(log_pF - log_pB, axis=1, keepdims=True)

    # Compute the Weighted Importance Sampling ratios
    log_w_is = jnp.sum(log_pF - log_behavior, axis=1, keepdims=True)
    w_is = jnp.exp(log_w_is) / batch_size

    # Compute the baseline (average returns over the batch of trajectories)
    hvi_baseline = jnp.mean(hvi_returns)

    hvi_error = w_is * hvi_returns
    error = lax.stop_gradient(w_is * (hvi_returns - hvi_baseline))
    loss = jnp.sum(log_pF * error)

    logs = {
        'returns': hvi_returns,
        'error': hvi_error,
        'loss': loss,
        'log_w_is': log_w_is,
    }
    return (loss, logs)


def hvi_on_policy_loss(
        log_pi,
        actions,
        log_rewards,
        num_edges,
    ):
    max_length = log_pi.shape[1]

    # Mask the log-probabilities, based on the sequence lengths
    seq_masks = (jnp.arange(max_length) <= num_edges[:, None])
    log_pi = jnp.where(seq_masks[..., None], log_pi, 0.)

    # Compute the forward log-probabilities
    log_pF = jnp.take_along_axis(log_pi, actions[..., None], axis=-1)
    log_pF = jnp.sum(log_pF, axis=(1, 2))

    # Compute the backward log-probabilities (fixed p_B)
    log_pB = -lax.lgamma(num_edges + 1.)  # -log(n!)

    hvi_rewards = log_pF - log_rewards - log_pB
    hvi_baseline = jnp.mean(hvi_rewards)
    error = lax.stop_gradient(hvi_rewards - hvi_baseline)
    loss = jnp.mean(log_pF * error)

    logs = {
        'rewards': hvi_rewards,
        'error': error,
        'loss': loss,
    }
    return (loss, logs)
