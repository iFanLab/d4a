import numpy as np
import warnings
import random
import os
import torch


def moment_diff(sx1, sx2, k, p=2):
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    return torch.norm(ss1 - ss2, p=p)


def central_moment_discrepancy(x1, x2, k, a=0, b=1, p=2):
    """
    CMD_k, refer to <https://arxiv.org/pdf/1702.08811.pdf>

    Parameters
    ==========
    a : float
        lower bound
    b : float
        upper bound
    """
    coef = b - a
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    loss_distr_shift = [torch.norm(mx1 - mx2, p=p) / coef]
    for i in range(2, k + 1):
        loss_distr_shift.append(moment_diff(sx1, sx2, i, p=p) / (coef**i))
    return loss_distr_shift


def setup(args):
    # close warning output
    warnings.filterwarnings("ignore")

    # output format
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False, profile='default')

    set_seed(args.seed)
    device = set_device(args.gpu)

    # dir
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    return {
        'device': device
    }


def set_seed(seed=0):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(gpu=-1):
    """Set GPU or CPU device to run code."""
    gpu = int(gpu)
    device = f'cuda:{gpu}' if torch.cuda.is_available() and gpu >= 0 else 'cpu'
    return device


def to_mask(*indices, num=None):
    if num is None:
        num = int(max(indices) + 1)
    masks = []
    for idx in indices:
        mask = torch.zeros(num).bool()
        mask[idx] = 1
        masks.append(mask)
    return masks


def save_checkpoint(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_checkpoint(model, filepath):
    model.load_state_dict(torch.load(filepath))
