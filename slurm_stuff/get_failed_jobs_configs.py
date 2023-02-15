import os


def get_failed_configs_list(wandb):
    failed_configs_file = os.path.join("configs", f"{wandb}_failed_configs")
    done_configs_file = os.path.join("configs", f"{wandb}_done_configs")

    if os.path.exists(failed_configs_file):
        with open(failed_configs_file, "r+") as f:
            failed_configs = f.read().splitlines()
    else:
        failed_configs = []
    if os.path.exists(done_configs_file):
        with open(done_configs_file, "r+") as f:
            done_configs = f.read().splitlines()
    else:
        done_configs = []

    failed_configs = set(failed_configs) - set(done_configs)
    failed_configs = [int(failed_config) for failed_config in failed_configs]
    failed_configs = sorted(failed_configs)

    return failed_configs


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--wandb",
        type=str,
        default="gfn_vs_hvi_complete",
        help="Name of the experiment",
    )

    args = parser.parse_args()
    print(get_failed_configs_list(args.wandb))
