from gfn.envs import HyperGrid
import torch
import os
from gfn.parametrizations import TBParametrization
from gfn.estimators import LogitPFEstimator, LogitPBEstimator, LogZEstimator
from gfn.samplers import LogitPFActionsSampler
from gfn.containers import ReplayBuffer, Transitions
import math


def make_tb_parametrization(env, PB, load_from=None):
    """
    It creates a TrajectoryBalance parametrization

    :param env: the environment we're working in
    :param load_from: If you want to load a previously saved model, you can pass the path to the saved
    model here
    :return: A parametrization of the model.
    """
    logZ_tensor = torch.tensor(0.0)
    logZ = LogZEstimator(logZ_tensor)
    logit_PF = LogitPFEstimator(env=env, module_name="NeuralNet")
    logit_PB = LogitPBEstimator(
        env=env,
        module_name="NeuralNet" if PB in ["learnable", "tied"] else "Uniform",
        torso=logit_PF.module.torso if PB == "tied" else None,
    )
    parametrization = TBParametrization(logit_PF, logit_PB, logZ)
    if load_from is not None:
        parametrization.load_state_dict(load_from)
    return parametrization


def make_buffer(env, capacity, load_from=None):
    """
    It creates a buffer for the environment. If load_from is not None, it loads the buffer from the
    path specified by load_from.

    :param env: the environment
    :param capacity: the capacity of the buffer
    :param load_from: the path to load the buffer from
    :return: the buffer
    """
    buffer = ReplayBuffer(env, capacity, objects="trajectories")
    if load_from is not None:
        buffer.load(load_from)
    return buffer


def make_optimizers(
    parametrization,
    lr,
    lr_PB,
    lr_Z,
    schedule,
    total_iterations,
    scheduler_type="None",
    multi_step_milestones=None,
    load_from=None,
):
    """
    It creates two optimizers, one for the parameters of the model and one for the log-partition
    function, and two schedulers, one for each optimizer

    :param parametrization: the parametrization of the model
    :param mode: "tb" or "tb_Z"
    :param lr: learning rate for the parameters
    :param lr_Z: learning rate for the logZ parameter
    :param schedule: the learning rate decay schedule, gamma if "linear" which is MultiStepLR, or the ratio between max_lr and min_lr if "cosine"
    :param load_from: the directory to load the optimizers from. If None, then the optimizers are initialized from
    scratch
    :return: optimizer, optimizer_Z, scheduler, scheduler_Z
    """
    params_pf = parametrization.logit_PF.module.parameters()
    optimizer_pf = torch.optim.Adam(params_pf, lr=lr)
    optimizer_pb = None
    if parametrization.logit_PB.module_name == "NeuralNet":
        params_pb = parametrization.logit_PB.module.parameters()
        optimizer_pb = torch.optim.Adam(params_pb, lr=lr_PB)
    optimizer_Z = torch.optim.Adam([parametrization.logZ.tensor], lr=lr_Z)
    scheduler_pb = None
    scheduler_pf = None
    scheduler_Z = None
    if scheduler_type == "multi_step":
        assert multi_step_milestones is not None
        gamma = schedule ** (1 / (multi_step_milestones - 1))
        milestones = (
            torch.linspace(
                int(total_iterations / multi_step_milestones),
                total_iterations,
                multi_step_milestones,
            )
            .int()
            .numpy()
        )
        scheduler_pf = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_pf, milestones=milestones, gamma=gamma
        )
        if optimizer_pb is not None:
            scheduler_pb = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_pb, milestones=milestones, gamma=gamma
            )
        scheduler_Z = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_Z, milestones=milestones, gamma=gamma
        )
    elif scheduler_type == "cosine":
        scheduler_pf = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_pf, T_max=total_iterations, eta_min=lr * schedule
        )
        if optimizer_pb is not None:
            scheduler_pb = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_pb, T_max=total_iterations, eta_min=lr_PB * schedule
            )
        scheduler_Z = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_Z, T_max=total_iterations, eta_min=lr_Z * schedule
        )
    elif scheduler_type == "plateau":
        scheduler_pf = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_pf, factor=schedule, patience=300, threshold=1e-3
        )
        if optimizer_pb is not None:
            scheduler_pb = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_pb, factor=schedule, patience=300, threshold=1e-3
            )
        scheduler_Z = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_Z, factor=schedule, patience=300, threshold=1e-3
        )
    elif scheduler_type == "None":
        pass
    else:
        raise ValueError("Unknown scheduler type")
    if load_from is not None:
        try:
            optimizer_pf.load_state_dict(
                torch.load(os.path.join(load_from, "optimizer.pt"))
            )
            if optimizer_pb is not None and os.path.exists(
                os.path.join(load_from, "optimizer_pb.pt")
            ):
                optimizer_pb.load_state_dict(
                    torch.load(os.path.join(load_from, "optimizer_pb.pt"))
                )
            optimizer_Z.load_state_dict(
                torch.load(os.path.join(load_from, "optimizer_Z.pt"))
            )
            if scheduler_pf is not None:
                scheduler_pf.load_state_dict(
                    torch.load(os.path.join(load_from, "scheduler.pt"))
                )
            if scheduler_pb is not None and os.path.exists(
                os.path.join(load_from, "scheduler_pb.pt")
            ):
                scheduler_pb.load_state_dict(
                    torch.load(os.path.join(load_from, "scheduler_pb.pt"))
                )
            if scheduler_Z is not None:
                scheduler_Z.load_state_dict(
                    torch.load(os.path.join(load_from, "scheduler_Z.pt"))
                )
        except FileNotFoundError:
            print("Could not load optimizers -- starting from scratch")
    return (
        optimizer_pf,
        optimizer_pb,
        optimizer_Z,
        scheduler_pf,
        scheduler_pb,
        scheduler_Z,
    )


def evaluate_trajectories(
    args, parametrization, loss_fn, trajectories, temperature, epsilon
):
    if args.mode == "modified_db":
        transitions = Transitions.from_trajectories(trajectories)
        scores = loss_fn.get_modified_scores(transitions)
        logPF_trajectories, logPB_trajectories = None, None
    else:
        logPF_trajectories, logPB_trajectories, scores = loss_fn.get_scores(
            trajectories
        )
    importance_sampling_weights = 1.0
    on_policy_importance_sampling_weights = 1.0
    baseline = 0.0
    if args.mode in [
        "symmetric_cycles",
        "forward_kl",
        "reverse_kl",
        "rws",
        "reverse_rws",
    ]:
        if args.baseline == "local":
            baseline = scores.mean().detach()
        elif args.baseline == "global":
            baseline = -parametrization.logZ.tensor.detach()
        else:
            baseline = 0.0
        if args.sampling_mode == "off_policy":
            tempered_logPF_trajectories = trajectories.log_pfs
            importance_sampling_weights = torch.exp(
                -tempered_logPF_trajectories + logPF_trajectories  # type: ignore
            ).detach()
        on_policy_importance_sampling_weights = (
            torch.exp(-scores) / torch.exp(-scores).sum()
        ).detach()

    return (
        scores,
        baseline,
        importance_sampling_weights,
        on_policy_importance_sampling_weights,
        logPF_trajectories,
        logPB_trajectories,
    )


def evaluate_loss(
    args,
    parametrization,
    scores,
    baseline,
    importance_sampling_weights,
    on_policy_importance_sampling_weights,
    logPF_trajectories,
    logPB_trajectories,
):
    if args.mode == "tb":
        loss = (scores + parametrization.logZ.tensor).pow(2)
    elif args.mode == "modified_db":
        loss = scores.pow(2)
    elif args.mode == "symmetric_cycles":
        loss = (
            logPF_trajectories
            * (scores.detach() - baseline - on_policy_importance_sampling_weights)
            - logPB_trajectories
            - logPB_trajectories  # type: ignore
            * on_policy_importance_sampling_weights
            * (scores.detach() - baseline)
        )

    elif args.mode == "reverse_kl":
        loss = logPF_trajectories * (scores.detach() - baseline) - logPB_trajectories
    elif args.mode == "reverse_rws":
        loss_pf = logPF_trajectories * (scores.detach() - baseline)
        loss_pb = (
            -on_policy_importance_sampling_weights  # type: ignore
            * logPB_trajectories  # type: ignore
            * (scores.detach() - baseline)
        )
        loss = loss_pf + loss_pb
    elif args.mode == "forward_kl":
        loss = -logPB_trajectories * (scores.detach() - baseline) - logPF_trajectories  # type: ignore
        loss = loss * on_policy_importance_sampling_weights
    elif args.mode == "rws":
        loss_pf = -logPF_trajectories * on_policy_importance_sampling_weights  # type: ignore
        loss_pb = -logPB_trajectories  # type: ignore
        loss = loss_pf + loss_pb
    else:
        raise NotImplementedError("Only TB is implemented for now")

    loss = loss * importance_sampling_weights
    loss = loss.mean()

    return loss


def get_gradients_log(
    parametrization, trajectories_sampler, args, loss_fn, actions_sampler
):
    gradients_log = {}
    logit_PF_parameters = list(parametrization.logit_PF.module.parameters())
    if parametrization.logit_PB.module_name == "NeuralNet":
        logit_PB_parameters = list(parametrization.logit_PB.module.parameters())
    for p in logit_PF_parameters:
        p.grad.zero_()
    if parametrization.logit_PB.module_name == "NeuralNet":
        for p in logit_PB_parameters:  # type: ignore
            p.grad.zero_()
    trajectories = trajectories_sampler.sample(1024)
    (
        scores,
        baseline,
        importance_sampling_weights,
        on_policy_importance_sampling_weights,
        logPF_trajectories,
        logPB_trajectories,
    ) = evaluate_trajectories(
        args,
        parametrization,
        loss_fn,
        trajectories,
        actions_sampler.temperature,
        actions_sampler.epsilon,
    )
    loss_big_batch = evaluate_loss(
        args,
        parametrization,
        scores,
        baseline,
        importance_sampling_weights,
        on_policy_importance_sampling_weights,
        logPF_trajectories,
        logPB_trajectories,
    )
    loss_big_batch.backward(retain_graph=True)
    gradients_big_batch = [p.grad.clone() for p in logit_PF_parameters]

    for K in [4, 6]:
        num_small_batches = int(1024 / 2**K)
        small_batches = []
        for k in range(num_small_batches):
            small_batches.append(trajectories[k * 2**K : (k + 1) * 2**K])
        per_batch_cosine_similarities = []
        for i, small_batch in enumerate(small_batches):
            for p in logit_PF_parameters:
                p.grad.zero_()
            if parametrization.logit_PB.module_name == "NeuralNet":
                for p in logit_PB_parameters:  # type: ignore
                    p.grad.zero_()
            parametrization.logZ.tensor.grad.zero_()
            (
                scores,
                baseline,
                importance_sampling_weights,
                on_policy_importance_sampling_weights,
                logPF_trajectories,
                logPB_trajectories,
            ) = evaluate_trajectories(
                args,
                parametrization,
                loss_fn,
                small_batch,
                actions_sampler.temperature,
                actions_sampler.epsilon,
            )
            loss_small_batch = evaluate_loss(
                args,
                parametrization,
                scores,
                baseline,
                importance_sampling_weights,
                on_policy_importance_sampling_weights,
                logPF_trajectories,
                logPB_trajectories,
            )

            loss_small_batch.backward(retain_graph=True)
            gradients_small_batch = [p.grad.clone() for p in logit_PF_parameters]
            cosine_similarities = [
                (gradient_big * gradient_small).sum()
                / (gradient_big.norm() * gradient_small.norm())
                for gradient_big, gradient_small in zip(
                    gradients_big_batch, gradients_small_batch
                )
            ]
            cosine_similarities = torch.stack(cosine_similarities)
            # cosine_similarity = cosine_similarities.mean()
            per_batch_cosine_similarities.append(cosine_similarities)
        per_batch_cosine_similarities = torch.stack(per_batch_cosine_similarities)
        gradients_log[
            f"cosine_similarity_K={K}"
        ] = per_batch_cosine_similarities.mean().item()

    return gradients_log
