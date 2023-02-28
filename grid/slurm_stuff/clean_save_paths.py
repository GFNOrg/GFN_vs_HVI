import os


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    s = 0
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            print(f"Removing empty folder: {path}")
            os.rmdir(path)
            s = s + 1
    print(f"Removed {s} empty folders")


def clean_failed_configs_file(path):
    done_configs_file = os.path.join("configs", f"{path}_done_configs")
    failed_configs_file = os.path.join("configs", f"{path}_failed_configs")
    with open(done_configs_file, "r+") as f:
        done_configs = f.read().splitlines()
    with open(failed_configs_file, "r+") as f1:
        failed_configs = f1.read().splitlines()
    print(f"Done configs: {len(done_configs)}")
    print(f"Failed configs: {len(failed_configs)}")
    new_failed_configs = set(failed_configs) - set(done_configs)
    new_failed_configs = list(new_failed_configs)
    print(f"New failed configs: {len(new_failed_configs)}")
    failed_configs_file_2 = os.path.join("configs", f"{path}_failed_configs")
    with open(failed_configs_file_2, "w+") as f2:
        f2.write("\n".join(new_failed_configs))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="gfn_vs_hvi_complete")
    args = parser.parse_args()

    path = os.path.join(os.environ["SCRATCH_PATH"], args.path, "models")
    remove_empty_folders(path)
    clean_failed_configs_file(args.path)
    print("OK")
