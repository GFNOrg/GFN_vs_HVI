import torch
import os
from gfn.losses import TBParametrization
from gfn.estimators import LogitPFEstimator
from gfn.samplers import DiscreteActionsSampler
import math


def get_metadata(load_from=None):
    if load_from is not None:
        try:
            with open(os.path.join(load_from, "metadata.txt"), "r") as f:
                lines = f.readlines()
                iteration = int(lines[0].split(":")[1].strip())
                wandb_id = lines[1].split(":")[1].strip()
        except FileNotFoundError:
            print("No metadata file found, starting from scratch")
            iteration = 0
            wandb_id = None
    else:
        iteration = 0
        wandb_id = None
    return (iteration, wandb_id)


def cosine_annealing_schedule(iteration, init, final, last_update):
    """
    A cosine annealing schedule that starts at init and ends at final after last_update iterations
    init is the max value
    """
    if iteration >= last_update:
        return final
    else:
        return final + (init - final) * 0.5 * (
            1 + math.cos(math.pi * iteration / last_update)
        )


def temperature_epsilon_schedule(
    iteration,
    init_temp,
    init_epsilon,
    final_temp,
    final_epsilon,
    last_update,
    scheduler_type="linear",
):
    """
    A temperature and epsilon schedule that starts at init_temp and ends at final_temp after last_update iterations
    """
    if iteration >= last_update:
        return final_temp, final_epsilon
    else:
        if scheduler_type == "linear":
            return (
                init_temp + (final_temp - init_temp) * iteration / last_update,
                init_epsilon + (final_epsilon - init_epsilon) * iteration / last_update,
            )
        elif scheduler_type == "cosine":
            return (
                cosine_annealing_schedule(
                    iteration, init_temp, final_temp, last_update
                ),
                cosine_annealing_schedule(
                    iteration, init_epsilon, final_epsilon, last_update
                ),
            )


def save(
    parametrization,
    optimizer_pf,
    optimizer_pb,
    optimizer_Z,
    scheduler_pf,
    scheduler_pb,
    scheduler_Z,
    replay_buffer,
    iteration,
    wandb_id,
    save_path,
):
    """
    It saves the model, optimizer, scheduler, and iteration number to a folder

    :param parametrization: the model
    :param optimizer: the optimizer for the parameters of the model
    :param optimizer_Z: optimizer for the latent space
    :param scheduler: the learning rate scheduler
    :param scheduler_Z: the scheduler for the Z optimizer
    :param iteration: the current iteration of the training loop
    :param wandb_id: the id of the run in wandb
    :param save_path: the path to save the model to
    """
    parametrization.save_state_dict(save_path)
    torch.save(optimizer_pf.state_dict(), os.path.join(save_path, "optimizer.pt"))
    if optimizer_pf is not None and scheduler_pb is not None:
        torch.save(
            optimizer_pb.state_dict(), os.path.join(save_path, "optimizer_pb.pt")
        )
        torch.save(
            scheduler_pb.state_dict(), os.path.join(save_path, "scheduler_pb.pt")
        )
    torch.save(optimizer_Z.state_dict(), os.path.join(save_path, "optimizer_Z.pt"))
    if scheduler_pf is not None and scheduler_Z is not None:
        torch.save(scheduler_pf.state_dict(), os.path.join(save_path, "scheduler.pt"))
        torch.save(scheduler_Z.state_dict(), os.path.join(save_path, "scheduler_Z.pt"))
    if replay_buffer is not None:
        replay_buffer.save(save_path)
    with open(os.path.join(save_path, "metadata.txt"), "w") as f:
        f.write("iteration: {}\n".format(iteration))
        f.write("wandb_id: {}\n".format(wandb_id))


def deep_iter(data, ix=tuple()):
    "Iterates over a multi-dimensional tensor, copied from https://stackoverflow.com/questions/59332694/replacement-of-nditer-for-numpy-array-for-pytorch-tensor"
    try:
        for i, element in enumerate(data):
            yield from deep_iter(element, ix + (i,))
    except:
        yield ix, data


def all_indices(dim, height):
    if dim == 1:
        return [(i,) for i in range(height)]
    return [(i, *j) for i in range(height) for j in all_indices(dim - 1, height)]


def get_exact_P_T(env, logit_PF: LogitPFEstimator, cheap=False):
    """This function evaluates the exact terminating state distribution P_T for HyperGrid.
    P_T(s') = u(s') P_F(s_f | s') where u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1
    """

    grid = env.build_grid()
    ndim = env.ndim
    height = env.height
    action_sampler = DiscreteActionsSampler(logit_PF, temperature=1.0)
    probabilities = action_sampler.get_probs(grid)
    u = torch.ones(grid.batch_shape)
    if cheap:
        indices = all_indices(ndim, height)
        for index in indices[1:]:
            parents = [
                index[:i] + (index[i] - 1,) + index[i + 1 :] + (i,)
                for i in range(len(index))
                if index[i] > 0
            ]
            parents = torch.tensor(parents).T.numpy().tolist()
            u[index] = torch.sum(u[parents[:-1]] * probabilities[parents])

    else:
        iter_u = list(deep_iter(u))
        for grid_ix, _ in iter_u:
            if grid_ix == (0,) * ndim:
                continue
            else:
                index = tuple(grid_ix)
                parents = [
                    index[:i] + (index[i] - 1,) + index[i + 1 :] + (i,)
                    for i in range(len(index))
                    if index[i] > 0
                ]
                parents = torch.tensor(parents).T.numpy().tolist()
                u[index] = torch.sum(u[parents[:-1]] * probabilities[parents])
    return (u * probabilities[..., -1]).view(-1).detach().cpu()


def JSD(P, Q):
    """Computes the Jensen-Shannon divergence between two distributions P and Q"""
    M = 0.5 * (P + Q)
    return 0.5 * (torch.sum(P * torch.log(P / M)) + torch.sum(Q * torch.log(Q / M)))


def get_number_of_modes(P_T, true_dist_pmf, env_dim, env_height):
    """This function returns the number of modes in the distribution P_T"""
    true_dist_pmf = true_dist_pmf.cpu().clone()
    P_T = P_T.cpu().clone()
    n_pixels_per_mode = round(env_height / 10) ** env_dim
    modes_idx = set(
        torch.argsort(true_dist_pmf, descending=True)[
            : (2**env_dim) * n_pixels_per_mode
        ].numpy()
    )
    P_T_modes_idx = set(
        torch.argsort(P_T, descending=True)[
            : (2**env_dim) * n_pixels_per_mode
        ].numpy()
    )
    return len(modes_idx.intersection(P_T_modes_idx)) / n_pixels_per_mode


def get_validation_info(env, parametrization, cheap=False):
    true_logZ = env.log_partition
    true_dist_pmf = env.true_dist_pmf.cpu()

    logZ = None
    if isinstance(parametrization, TBParametrization):
        logZ = parametrization.logZ.tensor.item()

    P_T = get_exact_P_T(env, parametrization.logit_PF, cheap=cheap)
    l1_dist = (P_T - true_dist_pmf).abs().mean().item()
    jsd = JSD(P_T, true_dist_pmf).item()
    number_of_modes = get_number_of_modes(P_T, true_dist_pmf, env.ndim, env.height)

    validation_info = {
        "l1_dist": l1_dist,
        "jsd": jsd,
        "modes_found": number_of_modes,
        "modes_found_rounded": round(number_of_modes),
    }
    if logZ is not None:
        validation_info["logZ_error"] = abs(logZ - true_logZ)

    return (
        validation_info,
        true_dist_pmf.view(*[env.height for _ in range(env.ndim)]),
        P_T.view(*[env.height for _ in range(env.ndim)]),
    )
