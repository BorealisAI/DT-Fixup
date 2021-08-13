# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import importlib
import pkgutil
from contextlib import contextmanager
from typing import TypeVar, Union, Generator
from pathlib import Path

PathType = Union[os.PathLike, str]
T = TypeVar("T")
ContextManagerFunctionReturnType = Generator[T, None, None]

class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.
        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


@contextmanager
def push_python_path(path: PathType) -> ContextManagerFunctionReturnType[None]:
    """
    Source: https://github.com/allenai/allennlp/blob/main/allennlp/common/util.py
    """
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)

def import_module_and_submodules(package_name: str) -> None:
    """
    Source: https://github.com/allenai/allennlp/blob/main/allennlp/common/util.py
    """
    importlib.invalidate_caches()

    with push_python_path("."):
        module = importlib.import_module(package_name)
        path = getattr(module, "__path__", [])
        path_string = "" if not path else path[0]

        for module_finder, name, _ in pkgutil.walk_packages(path):
            if path_string and module_finder.path != path_string:
                continue
            subpackage = f"{package_name}.{name}"
            import_module_and_submodules(subpackage)

def print_dict(f, d, prefix="    ", incr_prefix="    "):
    if not isinstance(d, dict):
        f.write("%s%s\n" % (prefix, d))
        if isinstance(d, tuple):
            for x in d:
                if isinstance(x, dict):
                    print_dict(f, x, prefix + incr_prefix, incr_prefix)
        return
    sorted_keys = sorted(d.keys())
    for k in sorted_keys:
        v = d[k]
        if isinstance(v, dict):
            f.write("%s%s:\n" % (prefix, k))
            print_dict(f, v, prefix + incr_prefix, incr_prefix)
        elif isinstance(v, list):
            f.write("%s%s:\n" % (prefix, k))
            for x in v:
                print_dict(f, x, prefix + incr_prefix, incr_prefix)
        else:
            f.write("%s%s: %s\n" % (prefix, k, v))
