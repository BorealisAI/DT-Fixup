# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Type, Callable, Dict, Tuple, List
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Dict[str, Tuple[Type, str]]] = defaultdict(dict)

def register(kind: str, name: str, constructor: str = None):
    kind_register = _REGISTRY[kind]

    def decorator(cls: Type):
        if name in kind_register:
            raise LookupError('{} already registered as kind {}'.format(name, kind))
        kind_register[name] = (cls, constructor)
        return cls

    return decorator

def lookup(kind: str, name: str) -> Callable:
    if kind not in _REGISTRY:
        raise KeyError('Nothing registered under "{}"'.format(kind))
    logger.debug(f"instantiating {name} of {kind}")
    cls, constructor = _REGISTRY[kind][name]
    if not constructor:
        return cls
    else:
        return getattr(cls, constructor)

def list_available(kind: str) -> List[str]:
    if kind not in _REGISTRY:
        raise KeyError('Nothing registered under "{}"'.format(kind))
    keys = list(_REGISTRY[kind].keys())
    return keys
