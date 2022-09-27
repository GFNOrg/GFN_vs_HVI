import os
import torch
from argparse import ArgumentParser
from tqdm import tqdm, trange

import wandb
import gfn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


from gfn.envs import HyperGrid
from gfn.samplers import (
    LogitPFActionsSampler,
    TrajectoriesSampler,
    LogitPBActionsSampler,
)
from gfn.estimators import LogitPFEstimator, LogitPBEstimator, LogZEstimator
from gfn.losses import TrajectoryBalance, DetailedBalance
from gfn.parametrizations import TBParametrization
from gfn.containers import Trajectories, Transitions


from gfn.validate import validate

from utils import (
    get_metadata,
    save,
    get_validation_info,
    temperature_epsilon_schedule,
)
from learn_utils import (
    make_tb_parametrization,
    make_optimizers,
    make_buffer,
    evaluate_trajectories,
    evaluate_loss,
    get_gradients_log,
)

import io
from PIL import Image

from paper_configs import all_configs_dict, all_extra_configs_dict
from get_failed_jobs_configs import get_failed_configs_list


parser = ArgumentParser()

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_cuda", action="store_true", default=False)

# 1 - Environment specific arguments
parser.add_argument(
    "--env",
    type=str,
    choices=["very_hard", "hard", "big", "easy", "medium", "manual"],
    default="hard",
)
parser.add_argument("--ndim", type=int, default=2)
parser.add_argument("--height", type=int, default=8)
parser.add_argument("--R0", type=float, default=0.1)
parser.add_argument("--reward_cos", action="store_true", default=False)

# 2 - Training specific arguments
parser.add_argument("--n_trajectories", type=int, default=1000000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_PB", type=float, default=1e-3)
parser.add_argument("--lr_Z", type=float, default=1e-1)
parser.add_argument(
    "--lr_scheduling",
    type=str,
    choices=["multi_step", "cosine", "plateau", "None"],
    default="None",
)
parser.add_argument(
    "--schedule",
    type=float,
    default=1.0,
    help="schedule * lr is the lowest lr, unless it's plateau in which case it's the multiplicative factor",
)
parser.add_argument(
    "--multi_step_milestones",
    type=int,
    default="2",
    help="e.g. if it's 10, then the lr will be annealed at max_iterations/10 and 2 * max_iterations/10, etc...",
)

# 3 - GFN specific arguments
parser.add_argument(
    "--sampling_mode",
    type=str,
    choices=["on_policy", "off_policy", "pure_off_policy", "off_policy_with_replay"],
    default="on_policy",
)
# The following arguments are only used when sampling_mode is "off_policy" or "pure_off_policy"
parser.add_argument("--init_temperature", type=float, default=1.0)
parser.add_argument("--final_temperature", type=float, default=1.0)
parser.add_argument("--init_epsilon", type=float, default=0.0)
parser.add_argument("--final_epsilon", type=float, default=0.0)
parser.add_argument(
    "--exploration_phase_ends_by",
    type=int,
    default=-1,
    help="If positive, it's number of iterations. If negative, exploration stops after -1/this * total_trajectories, and stays at the last value",
)
parser.add_argument(
    "--exploration_scheduling", type=str, choices=["linear", "cosine"], default="cosine"
)
parser.add_argument(
    "--temperature_sf",
    action="store_true",
    default=False,
    help="if true, it's sf_temperature",
)


parser.add_argument("--replay_capacity", type=int, default=10000)
parser.add_argument(
    "--temperature_sf_string",
    type=str,
    default="False",
    help="if True, it sets the args.temperature_sf to True. This is useful for wandb.",
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
parser.add_argument("--validation_interval", type=int, default=0)
parser.add_argument(
    "--gradient_estimation_interval", type=int, default=1000
)  # 1000 would be a good value

# 6 - Logging and checkpointing specific arguments
parser.add_argument("--wandb", type=str, default="hvi_paper_extra")
parser.add_argument("--no_wandb", action="store_true", default=False)

# 7 - Misc
parser.add_argument("--config_id", type=int, default=0)
parser.add_argument("--task_id", type=int, default=None)
parser.add_argument("--total", type=int, default=None)
parser.add_argument("--offset", type=int, default=None)
parser.add_argument("--failed_runs", action="store_true", default=False)

parser.add_argument(
    "--early_stop", type=int, default=20
)  # Number of successive logs such that if there is no improvement, we stop

args = parser.parse_args()

config_id = None
slurm_proc_id = os.environ.get("SLURM_PROCID")
print("slurm_proc_id:", slurm_proc_id)
print("args.total:", args.total)
print("args.offset:", args.offset)
if slurm_proc_id is not None and args.total is not None and args.config_id == 0:
    print("Total number of configs:", len(all_configs_dict))
    args.config_id = (
        int(slurm_proc_id) * args.total + args.offset + args.task_id  # type: ignore
    )
    print("args.config_id:", args.config_id)
config_id = args.config_id
if args.failed_runs:
    failed_configs = get_failed_configs_list(args.wandb)
    print("Total number of failed configs:", len(failed_configs))
    print(f"Getting the {args.config_id}th config from the failed configs list")
    config_id = failed_configs[config_id - 1]
    args.config_id = config_id

if config_id is not None and config_id != 0:
    config = all_extra_configs_dict[config_id - 1]
    for key in config:
        setattr(args, key, config[key])

if args.temperature_sf_string == "True":
    args.temperature_sf = True


run_name = (
    "temporary_run"
    if config_id is None or config_id == 0
    else f"{args.wandb}_{config_id}"
)
save_path = os.path.join(os.environ["SCRATCH_PATH"], args.wandb, "models", run_name)

loading_model = os.path.exists(save_path) and run_name != "temporary_run"
if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.seed == 0:
    seed = torch.randint(0, 1000000, (1,)).item()
else:
    seed = args.seed
torch.manual_seed(seed)
device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

print(
    f"config_id: {config_id} - run_name: {run_name} - device: {device_str} \n args: {args}"
)

## 1- Create the environment and models
if args.env == "hard":
    (ndim, height, R0) = (2, 64, 0.001)
elif args.env == "very_hard":
    (ndim, height, R0) = (2, 128, 0.001)
elif args.env == "big":
    (ndim, height, R0) = (2, 128, 0.1)
elif args.env == "medium":
    (ndim, height, R0) = (4, 8, 0.01)
elif args.env == "easy":
    (ndim, height, R0) = (4, 8, 0.1)
else:
    (ndim, height, R0) = (args.ndim, args.height, args.R0)
env = HyperGrid(ndim, height, R0, reward_cos=args.reward_cos)

parametrization = make_tb_parametrization(
    env, args.PB, load_from=save_path if loading_model else None
)
actions_sampler = LogitPFActionsSampler(
    estimator=parametrization.logit_PF, temperature=1.0
)
backward_actions_sampler = LogitPBActionsSampler(estimator=parametrization.logit_PB)
trajectories_sampler = TrajectoriesSampler(
    env, actions_sampler, backward_actions_sampler=backward_actions_sampler
)
if args.mode == "modified_db":
    loss_fn = DetailedBalance(parametrization)
else:
    loss_fn = TrajectoryBalance(
        parametrization, on_policy=(args.sampling_mode == "on_policy")
    )

n_iterations = args.n_trajectories // args.batch_size

(
    optimizer_pf,
    optimizer_pb,
    optimizer_Z,
    scheduler_pf,
    scheduler_pb,
    scheduler_Z,
) = make_optimizers(
    parametrization,
    args.lr,
    args.lr_PB,
    args.lr_Z,
    args.schedule,
    total_iterations=n_iterations,
    scheduler_type=args.lr_scheduling,
    multi_step_milestones=args.multi_step_milestones,
    load_from=save_path if loading_model else None,
)

replay_buffer = None
if (
    args.sampling_mode == "pure_off_policy"
    or args.sampling_mode == "off_policy_with_replay"
):
    replay_buffer = make_buffer(
        env,
        capacity=args.replay_capacity,
        load_from=save_path if loading_model else None,
    )

iteration, wandb_id = get_metadata(load_from=save_path if loading_model else None)

use_wandb = not args.no_wandb
if use_wandb:
    os.environ["WANDB_DIR"] = os.path.join(os.environ["SCRATCH_PATH"], args.wandb)
    wandb.init(project=args.wandb, id=wandb_id, resume="allow")
    wandb.config.update(args, allow_val_change=True)
    if config_id is not None:
        wandb.run.name = f"{args.wandb}_{config_id}"  # type: ignore

best_jsd = float("inf")
best_jsd_iteration = -1
if args.exploration_phase_ends_by < 0:
    exploration_phase_ends_by = int(
        -1 / args.exploration_phase_ends_by * n_iterations
    )  # one half, one third, etc. of the total number of iterations
else:
    exploration_phase_ends_by = args.exploration_phase_ends_by
current_jsd = torch.tensor(float("inf"))
for i in trange(iteration, n_iterations):
    if args.sampling_mode != "on_policy":
        temperature, epsilon = temperature_epsilon_schedule(
            i,
            args.init_temperature,
            args.init_epsilon,
            args.final_temperature,
            args.final_epsilon,
            last_update=exploration_phase_ends_by,
            scheduler_type=args.exploration_scheduling,
        )  # type: ignore
        if args.temperature_sf:
            actions_sampler.sf_temperature = temperature
        else:
            actions_sampler.temperature = temperature
        actions_sampler.epsilon = epsilon

    trajectories = trajectories_sampler.sample(args.batch_size)
    if replay_buffer is not None:
        replay_buffer.add(trajectories)
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

    optimizer_pf.zero_grad()
    optimizer_Z.zero_grad()
    if optimizer_pb is not None:
        optimizer_pb.zero_grad()

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
    if args.mode != "tb":
        loss_Z = (parametrization.logZ.tensor + scores.detach()).pow(2).mean()
        loss += loss_Z

    loss.backward()
    optimizer_pf.step()
    optimizer_Z.step()
    if scheduler_pf is not None and scheduler_Z is not None:
        if args.lr_scheduling == "plateau":
            scheduler_pf.step(current_jsd)  # type: ignore
            scheduler_Z.step(current_jsd)  # type: ignore
        else:
            scheduler_pf.step()  # type: ignore
            scheduler_Z.step()  # type: ignore
    if optimizer_pb is not None and scheduler_pb is not None:
        optimizer_pb.step()
        if args.lr_scheduling == "plateau":
            scheduler_pb.step(current_jsd)  # type: ignore
        else:
            scheduler_pb.step()  # type: ignore

    if (
        args.gradient_estimation_interval != 0
        and i % args.gradient_estimation_interval == 0
    ) or i == n_iterations - 1:
        gradients_log = get_gradients_log(
            parametrization, trajectories_sampler, args, loss_fn, actions_sampler
        )
    else:
        gradients_log = {}
    if i % args.validation_interval == 0 or i == n_iterations - 1:
        if run_name != "temporary_run":
            save(
                parametrization,
                optimizer_pf,
                optimizer_pb,
                optimizer_Z,
                scheduler_pf,
                scheduler_pb,
                scheduler_Z,
                replay_buffer,
                i,
                wandb.run.id if use_wandb else None,  # type: ignore
                save_path,
            )
        to_log = {"states_visited": (i + 1) * args.batch_size, "loss": loss.item()}
        validation_info, true_dist, P_T = get_validation_info(env, parametrization)
        current_jsd = validation_info["jsd"]
        to_log.update(validation_info)
        to_log.update(gradients_log)

        if use_wandb:
            wandb.log(to_log, step=i)
            if env.ndim == 2:
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                ax.imshow(P_T.numpy())
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                pillow_image = Image.open(buf)
                plt.clf()
                plt.close()
                wandb.log({"P_T": wandb.Image(pillow_image)}, step=i)
                wandb.log(
                    {
                        "temperature": actions_sampler.temperature,
                        "sf_temperature": actions_sampler.sf_temperature,
                        "epsilon": actions_sampler.epsilon,
                        "lr": optimizer_pf.param_groups[0]["lr"],
                    },
                    step=i,
                )

        tqdm.write(f"{i}: {to_log} / {2 ** ndim}")
        if to_log["jsd"] < best_jsd:
            best_jsd = to_log["jsd"]
            best_jsd_iteration = i
        if (
            args.early_stop > 0
            and i - best_jsd_iteration >= args.early_stop * args.validation_interval
        ):
            print("Early stopping")
            break
