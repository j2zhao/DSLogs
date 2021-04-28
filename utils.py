import random
import string
from collections.abc import Collection
from typing import NewType, Tuple


def rand_string(N):
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(N))

def coll_function(func, obj):
    if isinstance(obj, Tuple):
        outputs = []
        for obj_ in obj:
            outputs.append(coll_function(func, obj_))
        return tuple(outputs)
    else:
        return func(obj)