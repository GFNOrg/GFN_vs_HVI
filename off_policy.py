import os
import torch
from argparse import ArgumentParser
from tqdm import tqdm, trange

import wandb
import gfn

from gfn.envs import HyperGrid
from gfn.samplers import LogitPFActionsSampler, TrajectoriesSampler
from gfn.estimators import LogitPFEstimator, LogitPBEstimator, LogZEstimator
from gfn.losses import TrajectoryBalance, DetailedBalance
from gfn.parametrizations import TBParametrization
from gfn.containers import Trajectories, Transitions


from gfn.validate import validate

from utils import (
    make_tb_parametrization,
    make_optimizers,
    get_metadata,
    save,
    get_validation_info,
    make_buffer,
    temperature_epsilon_schedule,
    evaluate_trajectories,
    evaluate_loss,
)

from all_configs import all_configs_list_of_dicts, total_configs
from get_failed_jobs_configs import get_failed_configs_list

parser = ArgumentParser()

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_cuda", action="store_true", default=False)

# 1 - Environment specific arguments
parser.add_argument(
    "--env", type=str, choices=["hard", "easy", "medium", "manual"], default="hard"
)
parser.add_argument("--ndim", type=int, default=None)
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--R0", type=float, default=None)

# 2 - Training specific arguments
parser.add_argument("--n_trajectories", type=int, default=1000000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_Z", type=float, default=1e-1)
parser.add_argument("--schedule", type=float, default=1.0)

# 3 - GFN specific arguments
parser.add_argument(
    "--sampling_mode",
    type=str,
    choices=["on_policy", "off_policy", "pure_off_policy"],
    default="on_policy",
)
parser.add_argument("--init_temperature", type=float, default=None)
parser.add_argument("--final_temperature", type=float, default=1.0)
parser.add_argument("--init_epsilon", type=float, default=None)
parser.add_argument("--final_epsilon", type=float, default=0.0)
parser.add_argument("--exploration_phase_ends_by", type=int, default=None)
parser.add_argument(
    "--prefill",
    type=int,
    default=100,
    help="Number of iterations where the buffer is filled before training",
)

# 4 - Learning specific arguments
parser.add_argument(
    "--mode",
    type=str,
    choices=[
        "tb",
        "forward_kl",
        "reverse_kl",
        "rws",
        "reverse_rws",
        "modified_db",
        "symmetric_cycles",
    ],
    default="tb",
)
parser.add_argument("--PB", type=str, choices=["uniform", "learnable", "tied"])
parser.add_argument(
    "--baseline", type=str, choices=["None", "local", "global"], default="None"
)

# 5 - Validation specific arguments
parser.add_argument("--validation_interval", type=int, default=100)
parser.add_argument(
    "--gradient_estimation_interval", type=int, default=0
)  # 1000 would be a good value

# 6 - Logging and checkpointing specific arguments
parser.add_argument("--log_directory", type=str, default="gfn_vs_hvi_complete_test")
parser.add_argument("--wandb", type=str, default=None)

# 7 - Slurm arguments
parser.add_argument("--config_id", type=int, default=None)  # Should start with 1
parser.add_argument("--task_id", type=int, default=None)
parser.add_argument("--total", type=int, default=None)
parser.add_argument("--offset", type=int, default=None)
parser.add_argument("--failed_runs", action="store_true", default=False)


parser.add_argument(
    "--early_stop", type=int, default=20
)  # Number of successive logs such that if there is no improvement, we stop

args = parser.parse_args()


# TODO: create the variable config_id if args.config_id is not None, or if this is launched by SLURM
config_id = None
slurm_proc_id = os.environ.get("SLURM_PROCID")
if args.config_id is not None or (slurm_proc_id is not None and args.total is not None):
    assert args.wandb is not None
    print("Total number of configs:", total_configs)
    if args.config_id is None:
        args.config_id = (
            int(slurm_proc_id) * args.total + args.offset + args.task_id  # type: ignore
        )
    config_id = args.config_id
    if args.failed_runs:
        failed_configs = get_failed_configs_list(args.wandb)
        print("Total number of failed configs:", len(failed_configs))
        print(f"Getting the {args.config_id}th config from the failed configs list")
        config_id = failed_configs[config_id - 1]
        args.config_id = config_id
    config = all_configs_list_of_dicts[config_id - 1]
    for key in config:
        setattr(args, key, config[key])
    done_configs_file = os.path.join("configs", f"{args.wandb}_done_configs")
    started_configs_file = os.path.join("configs", f"{args.wandb}_started_configs")
    failed_configs_file = os.path.join("configs", f"{args.wandb}_failed_configs")
    if os.path.exists(done_configs_file):
        with open(done_configs_file, "r+") as f:
            done_configs = f.read().splitlines()
    else:
        done_configs = []
    if os.path.exists(started_configs_file):
        with open(started_configs_file, "r+") as f1:
            started_configs = f1.read().splitlines()
    else:
        started_configs = []
    if str(config_id) not in started_configs:
        with open(started_configs_file, "a+") as f2:
            f2.write(f"{config_id}\n")
    if str(config_id) in done_configs:
        print("Config already done")
        exit(0)

try:
    run_name = "temporary_run" if config_id is None else f"{args.wandb}_{config_id}"
    save_path = os.path.join(
        os.environ["SCRATCH_PATH"], args.log_directory, "models", run_name
    )

    loading_model = os.path.exists(save_path) and run_name != "temporary_run"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.manual_seed(args.seed)
    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    print(
        f"config_id: {config_id} - run_name: {run_name} - device: {device_str} \n args: {args}"
    )

    ## 1- Create the environment and models
    if args.env == "hard":
        (ndim, height, R0) = (2, 64, 0.001)
    elif args.env == "medium":
        (ndim, height, R0) = (4, 8, 0.01)
    elif args.env == "easy":
        (ndim, height, R0) = (4, 8, 0.1)
    else:
        (ndim, height, R0) = (args.ndim, args.height, args.R0)
    env = HyperGrid(ndim, height, R0)

    parametrization = make_tb_parametrization(
        env, args.PB, load_from=save_path if loading_model else None
    )
    actions_sampler = LogitPFActionsSampler(
        estimator=parametrization.logit_PF, temperature=1.0
    )
    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    if args.mode == "modified_db":
        loss_fn = DetailedBalance(parametrization)
    else:
        loss_fn = TrajectoryBalance(parametrization)

    optimizer, optimizer_Z, scheduler, scheduler_Z = make_optimizers(
        parametrization,
        args.mode,
        args.lr,
        args.lr_Z,
        args.schedule,
        load_from=save_path if loading_model else None,
    )

    replay_buffer = None
    if args.sampling_mode == "pure_off_policy":
        replay_buffer = make_buffer(
            env, capacity=10000, load_from=save_path if loading_model else None
        )

    iteration, wandb_id = get_metadata(load_from=save_path if loading_model else None)

    use_wandb = args.wandb is not None
    if use_wandb:
        os.environ["WANDB_DIR"] = os.path.join(
            os.environ["SCRATCH_PATH"], args.log_directory
        )
        wandb.init(project=args.wandb, id=wandb_id, resume="allow")
        wandb.config.update(args, allow_val_change=True)
        if config_id is not None:
            wandb.run.name = f"{args.wandb}_{config_id}"  # type: ignore

    n_iterations = args.n_trajectories // args.batch_size

    best_jsd = 100.324
    best_jsd_iteration = -1
    for i in trange(iteration, n_iterations):
        if args.sampling_mode != "on_policy":
            temperature, epsilon = temperature_epsilon_schedule(
                i,
                args.init_temperature or 2.0,
                args.init_epsilon or 1.0,
                args.final_temperature,
                args.final_epsilon,
                args.exploration_phase_ends_by or 100,
            )
            actions_sampler.temperature = temperature
            actions_sampler.epsilon = epsilon
        trajectories = trajectories_sampler.sample(args.batch_size)
        if replay_buffer is not None:
            replay_buffer.add(trajectories)
            if i < args.prefill:
                continue
            trajectories = replay_buffer.sample(args.batch_size)

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

        if args.mode != "tb":
            optimizer_Z.zero_grad()  # type: ignore
            loss_Z = (parametrization.logZ.tensor + scores.detach()).pow(2).mean()
            loss_Z.backward()
            optimizer_Z.step()  # type: ignore
            scheduler_Z.step()  # type: ignore

        optimizer.zero_grad()

        loss = evaluate_loss(
            args,
            parametrization,
            scores,
            baseline,
            importance_sampling_weights,
            on_policy_importance_sampling_weights,
            logPF_trajectories,
            logPB_trajectories,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % args.validation_interval == 0:
            save(
                parametrization,
                optimizer,
                optimizer_Z,
                scheduler,
                scheduler_Z,
                replay_buffer,
                i,
                wandb.run.id if use_wandb else None,  # type: ignore
                save_path,
            )
        gradients_log = {}
        if (
            args.gradient_estimation_interval != 0
            and i % args.gradient_estimation_interval == 0
        ):
            logit_PF_parameters = list(parametrization.logit_PF.module.parameters())
            for p in logit_PF_parameters:
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
            loss_big_batch.backward()
            gradients_big_batch = [p.grad.clone() for p in logit_PF_parameters]

            for K in [4, 6]:
                num_small_batches = int(1024 / 2**K)
                small_batches = []
                for k in range(num_small_batches):
                    small_batches.append(trajectories[k * 2**K : (k + 1) * 2**K])
                per_batch_cosine_similarities = []
                for small_batch in small_batches:
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

                    for p in logit_PF_parameters:
                        p.grad.zero_()

                    loss_small_batch.backward()
                    gradients_small_batch = [
                        p.grad.clone() for p in logit_PF_parameters
                    ]
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
                per_batch_cosine_similarities = torch.stack(
                    per_batch_cosine_similarities
                )
                gradients_log[
                    f"cosine_similarity_K={K}"
                ] = per_batch_cosine_similarities.mean().item()

        if i % args.validation_interval == 0:
            to_log = {"states_visited": (i + 1) * args.batch_size, "loss": loss.item()}
            validation_info = get_validation_info(env, parametrization)
            to_log.update(validation_info)
            to_log.update(gradients_log)
            if use_wandb:
                wandb.log({k: v for k, v in to_log.items() if k != "jsd"}, step=i)
                wandb.log({"jsd": to_log["jsd"]}, step=i)
            tqdm.write(f"{i}: {to_log}")
            if to_log["jsd"] < best_jsd:
                best_jsd = to_log["jsd"]
                best_jsd_iteration = i
            if i - best_jsd_iteration >= args.early_stop * args.validation_interval:
                print("Early stopping")
                break

    if config_id is not None:
        with open(done_configs_file, "a+") as f:  # type: ignore
            f.write(f"{config_id}\n")
except Exception as e:
    print(e)
    if config_id is not None:
        with open(failed_configs_file, "a+") as f:  # type: ignore
            f.write(f"{config_id}\n")
    if len(os.listdir(save_path)) == 0:
        print(f"Removing empty folder: {save_path}")
        os.rmdir(save_path)
        print("Done")
