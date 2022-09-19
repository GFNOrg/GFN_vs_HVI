import os


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


if __name__ == "__main__":
    path = os.path.join(os.environ["SCRATCH_PATH"], "gfn_vs_hvi", "models")
    remove_empty_folders(path)
    print("OK")
