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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="gfn_vs_hvi_complete")
    args = parser.parse_args()

    path = os.path.join(os.environ["SCRATCH_PATH"], args.path, "models")
    remove_empty_folders(path)
    print("OK")
