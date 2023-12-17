"""
decorator
"""
import typing as t
import torch
from torchsummary import summary
from thop import profile

__all__ = ['log_profile' ]


def log_profile(mode: str, shape: t.Tuple = (3, 640, 640), batch_size: int = 1):
    """
    print some profile of model, such as Params, FLOPs
    """
    def wrapper(func: t.Callable):
        def inner(*args, **kwargs):
            model = func(*args, **kwargs)
            if getattr(model, 'model'):
                m = model.model
            device = kwargs.get('device', 'cpu')
            m = m.to(device)
            x = torch.randn((batch_size, *shape), device=device)

            if mode == 'thop':
                flops, params = profile(m, (x,))
                print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
                print("params=", str(params / 1e6) + '{}'.format("M"))
            if mode == 'torchsummary':
                summary(m, shape, batch_size=batch_size, device=device)
            return model
        return inner
    return wrapper
