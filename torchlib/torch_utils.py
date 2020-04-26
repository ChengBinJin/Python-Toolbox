import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from collections import OrderedDict


def get_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = np.sum(np.prod(p.size()) for p in model_parameters)
    params = params / (1000 * 1000)
    return params
