import typing as t

from functools import wraps

__all__ = ['log_wrap', ]

def log_wrap(title: t.Optional[str] = None, stream: t.Optional[str] = None):
    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            nonlocal title
            if title is None:
                title = func.__name__
            file_path = kwargs.pop('file_path', None)

            res = func(*args, **kwargs)
            if stream == 'file' and file_path:
                with open(file_path, 'a+') as f:
                    f.write(f'{title}\n')
                    for i, rs in enumerate(res[-2:]):
                        rs: t.List
                        for r in rs:
                            if isinstance(r, str):
                                f.write(f'{r}\n')
                            elif isinstance(r, (list, tuple)) and r:
                                key, values = r[0], r[1:]
                                f.write(f'{key}: {" ".join([v for v in values])}\n')
                    f.write('------------------------------------------------------------\n')
                    f.write('------------------------------------------------------------\n')
            elif stream == 'std':
                for i, rs in enumerate(res[-2:]):
                    rs: t.List
                    for r in rs:
                        if isinstance(r, str):
                            print(r)
                        elif isinstance(r, (list, tuple)):
                            key, values = r[0], r[1:]
                            print(f'{key}: {" ".join([v for v in values])}\n')
                print('------------------------------------------------------------')
                print('------------------------------------------------------------')
            return res
        return inner
    return wrapper