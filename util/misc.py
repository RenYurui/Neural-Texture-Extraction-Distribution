import torch
from torch._six import container_abcs, string_classes


def requires_grad(model, require=True):
    r""" Set a model to require gradient or not.

    Args:
        model (nn.Module): Neural network model.
        require (bool): Whether the network requires gradient or not.

    Returns:

    """
    for p in model.parameters():
        p.requires_grad = require


def to_device(data, device):
    r"""Move all tensors inside data to device.

    Args:
        data (dict, list, or tensor): Input data.
        device (str): 'cpu' or 'cuda'.
    """
    assert device in ['cpu', 'cuda']
    if isinstance(data, torch.Tensor):
        data = data.to(torch.device(device))
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {key: to_device(data[key], device) for key in data}
    elif isinstance(data, container_abcs.Sequence) and \
            not isinstance(data, string_classes):
        return [to_device(d, device) for d in data]
    else:
        return data


def to_cuda(data):
    r"""Move all tensors inside data to gpu.

    Args:
        data (dict, list, or tensor): Input data.
    """
    return to_device(data, 'cuda')


def to_cpu(data):
    r"""Move all tensors inside data to cpu.

    Args:
        data (dict, list, or tensor): Input data.
    """
    return to_device(data, 'cpu')


def to_half(data):
    r"""Move all floats to half.

    Args:
        data (dict, list or tensor): Input data.
    """
    if isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        data = data.half()
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {key: to_half(data[key]) for key in data}
    elif isinstance(data, container_abcs.Sequence) and \
            not isinstance(data, string_classes):
        return [to_half(d) for d in data]
    else:
        return data


def to_float(data):
    r"""Move all halfs to float.

    Args:
        data (dict, list or tensor): Input data.
    """
    if isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        data = data.float()
        return data
    elif isinstance(data, container_abcs.Mapping):
        return {key: to_float(data[key]) for key in data}
    elif isinstance(data, container_abcs.Sequence) and \
            not isinstance(data, string_classes):
        return [to_float(d) for d in data]
    else:
        return data





