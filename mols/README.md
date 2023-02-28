This code is forked from [this repository](https://github.com/GFNOrg/gflownet/tree/trajectory_balance).

## Molecule experiments

Additional requirements:
- `pandas rdkit torch_geometric h5py ray`
- a few biochemistry programs, see `mols/Programs/README`

For `rdkit` in particular we found it to be easier to install through (mini)conda, but `rdkit-pypi` also works on `pip` in a vanilla python virtual environment. [`torch_geometric`](https://github.com/rusty1s/pytorch_geometric) has non-trivial installation instructions.

If you have CUDA 10.1 configured, you can run `pip install -r requirements.txt`. You can also change `requirements.txt` to match your CUDA version. (Replace cu101 to cuXXX, where XXX is your CUDA version).

We compress the 300k molecule dataset for size. To uncompress it, run `cd mols/data/; gunzip docked_mols.h5.gz`.
