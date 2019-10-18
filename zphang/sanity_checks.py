import copy
import math

import numpy as np
import scipy.stats

import torch


def copy_state_dict(model):
    return copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})


def set_no_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def get_parametered_layer_names(model):
    return [
        name
        for name, module in model.named_children()
        if len(list(module.parameters()))
    ]


def chain_getattr(obj, attr_name):
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def randomize_layer(layer):
    # Just taking random initialization schemes
    for name, param in layer.named_parameters():
        if len(param.shape) == 4:
            # Conv kernel
            n = param.shape[0] * param.shape[1] * param.shape[3]
            param.normal_(0, math.sqrt(2. / n))
        elif len(param.shape) == 2:
            # Linear weight
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif len(param.shape) == 1:
            # Linear bias
            bound = 1 / math.sqrt(len(param))
            torch.nn.init.uniform_(param, -bound, bound)
        else:
            raise RuntimeError()


def get_model_device(model):
    parameter_list = list(model.parameters())
    if parameter_list:
        return parameter_list[0].device
    else:
        return torch.device("cpu")


def cascading_parameter_randomization_generator(model):
    device = get_model_device(model)
    state_dict_copy = copy_state_dict(model)
    reversed_layer_names = get_parametered_layer_names(model)[::-1]
    for layer_name in reversed_layer_names:
        layer = getattr(model, layer_name)
        randomize_layer(layer)
        yield layer_name
    model.load_state_dict(state_dict_copy)
    model.to(device)


def independent_parameter_randomization_generator(model):
    device = get_model_device(model)
    reversed_layer_names = get_parametered_layer_names(model)[::-1]
    for layer_name in reversed_layer_names:
        layer = getattr(model, layer_name)
        layer_state_dict_copy = copy_state_dict(layer)
        randomize_layer(layer)
        yield layer_name
        layer.load_state_dict(layer_state_dict_copy)
        layer.to(device)


def spearman_comparison(masks1, masks2, reduce=True):
    assert masks1.shape == masks2.shape
    ls = []
    for i in range(masks1.shape[0]):
        correl = scipy.stats.spearmanr(
            masks1[i].reshape(-1),
            masks2[i].reshape(-1),
        ).correlation
        if np.isnan(correl):
            # Should mean one of the inputs is all 1s.
            # Use a heuristic here:
            if len(np.unique(masks1[i])) == len(np.unique(masks2[i])) == 1:
                correl = 1
            else:
                correl = 0

        ls.append(correl)
    if reduce:
        return np.mean(ls)
    else:
        return np.array(ls)
