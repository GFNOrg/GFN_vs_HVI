This repo include the code for the HyperGrid experiments shown in the "GFlowNets and Variational Inference" paper.

The files [paper_configs](./slurm_stuff/paper_configs.py) and [small_configs](./slurm_stuff/small_configs.py) shows the configurations used for the [main file](./train.py) in order to obtain the paper results.


# To use

First install the [gfn library](https://github.com/saleml/gfn):
```
git clone https://github.com/saleml/gfn.git
conda create -n gfn python=3.10
conda activate gfn
cd gfn
pip install -e .
```

Then:
```
conda activate gfn
pip install tqdm
```
and optionally `pip install wandb`

You should then be able to run [train.py]](./train.py) with `python train.py --no_wandb --env manual --ndim 2 --height 16 --mode reverse_kl` for example
