import torch
from gfn.estimators import LogitPFEstimator
from gfn.containers import Trajectories


def get_tempered_tb_is_weights(
    logit_PF: LogitPFEstimator, trajectories: Trajectories, temperature: float
):
    """This function evaluates the importance samplign weights for a batch of trajectories sampled from a tempered
    version of P_F(tau) rather than P_F(tau)"""
    pass
