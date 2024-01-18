import random
import torch
import numpy as np

# Time out related class and function
class OutOfTimeException(Exception):
    pass


def timeouthandler(signum, frame):
    raise OutOfTimeException


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
