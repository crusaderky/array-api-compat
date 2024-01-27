# Several names are not included in the above import *
import torch as _torch
from torch import *  # noqa: F401, F403

from .._internal import _get_all_public_members


def exlcude(name):
    if (
        name.startswith("_")
        or name.endswith("_")
        or "cuda" in name
        or "cpu" in name
        or "backward" in name
    ):
        return True
    return False


_torch_all = _get_all_public_members(_torch, exclude=exlcude, extend_all=True)

for _name in _torch_all:
    globals()[_name] = getattr(_torch, _name)


from ..common._helpers import (  # noqa: E402
    array_namespace,
    device,
    get_namespace,
    is_array_api_obj,
    size,
    to_device,
)

# These imports may overwrite names from the import * above.
from ._aliases import (  # noqa: E402
    add,
    all,
    any,
    arange,
    astype,
    atan2,
    bitwise_and,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    broadcast_arrays,
    broadcast_to,
    can_cast,
    concat,
    divide,
    empty,
    equal,
    expand_dims,
    eye,
    flip,
    floor_divide,
    full,
    greater,
    greater_equal,
    isdtype,
    less,
    less_equal,
    linspace,
    logaddexp,
    matmul,
    matrix_transpose,
    max,
    mean,
    min,
    multiply,
    newaxis,
    nonzero,
    not_equal,
    ones,
    permute_dims,
    pow,
    prod,
    remainder,
    reshape,
    result_type,
    roll,
    sort,
    squeeze,
    std,
    subtract,
    sum,
    take,
    tensordot,
    tril,
    triu,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
    var,
    vecdot,
    where,
    zeros,
)

__all__ = []

__all__ += _torch_all

__all__ += [
    "is_array_api_obj",
    "array_namespace",
    "get_namespace",
    "device",
    "to_device",
    "size",
]

__all__ += [
    "add",
    "all",
    "any",
    "arange",
    "astype",
    "atan2",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "concat",
    "divide",
    "empty",
    "equal",
    "expand_dims",
    "eye",
    "flip",
    "floor_divide",
    "full",
    "greater",
    "greater_equal",
    "isdtype",
    "less",
    "less_equal",
    "linspace",
    "logaddexp",
    "matmul",
    "matrix_transpose",
    "max",
    "mean",
    "min",
    "multiply",
    "newaxis",
    "nonzero",
    "not_equal",
    "ones",
    "permute_dims",
    "pow",
    "prod",
    "remainder",
    "reshape",
    "result_type",
    "roll",
    "sort",
    "squeeze",
    "std",
    "subtract",
    "sum",
    "take",
    "tensordot",
    "tril",
    "triu",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "var",
    "vecdot",
    "where",
    "zeros",
]


# See the comment in the numpy __init__.py
__import__(__package__ + ".linalg")

__array_api_version__ = "2022.12"
