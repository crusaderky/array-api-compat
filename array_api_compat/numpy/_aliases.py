from __future__ import annotations

from typing import Optional, Union

from .._internal import get_xp
from ..common import _aliases
from ..common._typing import NestedSequence, SupportsBufferProtocol
from ._info import __array_namespace_info__
from ._typing import Array, Device, DType

import numpy as np

bool = np.bool_

# Basic renames
acos = np.arccos
acosh = np.arccosh
asin = np.arcsin
asinh = np.arcsinh
atan = np.arctan
atan2 = np.arctan2
atanh = np.arctanh
bitwise_left_shift = np.left_shift
bitwise_invert = np.invert
bitwise_right_shift = np.right_shift
concat = np.concatenate
pow = np.power

arange = get_xp(np)(_aliases.arange)
empty = get_xp(np)(_aliases.empty)
empty_like = get_xp(np)(_aliases.empty_like)
eye = get_xp(np)(_aliases.eye)
full = get_xp(np)(_aliases.full)
full_like = get_xp(np)(_aliases.full_like)
linspace = get_xp(np)(_aliases.linspace)
ones = get_xp(np)(_aliases.ones)
ones_like = get_xp(np)(_aliases.ones_like)
zeros = get_xp(np)(_aliases.zeros)
zeros_like = get_xp(np)(_aliases.zeros_like)
UniqueAllResult = get_xp(np)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(np)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(np)(_aliases.UniqueInverseResult)
unique_all = get_xp(np)(_aliases.unique_all)
unique_counts = get_xp(np)(_aliases.unique_counts)
unique_inverse = get_xp(np)(_aliases.unique_inverse)
unique_values = get_xp(np)(_aliases.unique_values)
std = get_xp(np)(_aliases.std)
var = get_xp(np)(_aliases.var)
cumulative_sum = get_xp(np)(_aliases.cumulative_sum)
cumulative_prod = get_xp(np)(_aliases.cumulative_prod)
clip = get_xp(np)(_aliases.clip)
permute_dims = get_xp(np)(_aliases.permute_dims)
reshape = get_xp(np)(_aliases.reshape)
argsort = get_xp(np)(_aliases.argsort)
sort = get_xp(np)(_aliases.sort)
nonzero = get_xp(np)(_aliases.nonzero)
ceil = get_xp(np)(_aliases.ceil)
floor = get_xp(np)(_aliases.floor)
trunc = get_xp(np)(_aliases.trunc)
matmul = get_xp(np)(_aliases.matmul)
matrix_transpose = get_xp(np)(_aliases.matrix_transpose)
tensordot = get_xp(np)(_aliases.tensordot)
sign = get_xp(np)(_aliases.sign)
finfo = get_xp(np)(_aliases.finfo)
iinfo = get_xp(np)(_aliases.iinfo)


def _supports_buffer_protocol(obj):
    try:
        memoryview(obj)
    except TypeError:
        return False
    return True


# asarray also adds the copy keyword, which is not present in numpy 1.0.
# asarray() is different enough between numpy, cupy, and dask, the logic
# complicated enough that it's easier to define it separately for each module
# rather than trying to combine everything into one function in common/
def asarray(
    obj: (
        Array 
        | bool | int | float | complex 
        | NestedSequence[bool | int | float | complex] 
        | SupportsBufferProtocol
    ),
    /,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    copy: "Optional[Union[bool, np._CopyMode]]" = None,
    **kwargs,
) -> Array:
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device for NumPy: {device!r}")

    if hasattr(np, '_CopyMode'):
        if copy is None:
            copy = np._CopyMode.IF_NEEDED
        elif copy is False:
            copy = np._CopyMode.NEVER
        elif copy is True:
            copy = np._CopyMode.ALWAYS
    else:
        # Not present in older NumPys. In this case, we cannot really support
        # copy=False.
        if copy is False:
            raise NotImplementedError("asarray(copy=False) requires a newer version of NumPy.")

    return np.array(obj, copy=copy, dtype=dtype, **kwargs)


def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: bool = True,
    device: Optional[Device] = None,
) -> Array:
    return x.astype(dtype=dtype, copy=copy)


# count_nonzero returns a python int for axis=None and keepdims=False
# https://github.com/numpy/numpy/issues/17562
def count_nonzero(x: Array, axis=None, keepdims=False) -> Array:
    result = np.count_nonzero(x, axis=axis, keepdims=keepdims)
    if axis is None and not keepdims:
        return np.asarray(result)
    return result


# These functions are completely new here. If the library already has them
# (i.e., numpy 2.0), use the library version instead of our wrapper.
if hasattr(np, 'vecdot'):
    vecdot = np.vecdot
else:
    vecdot = get_xp(np)(_aliases.vecdot)

if hasattr(np, 'isdtype'):
    isdtype = np.isdtype
else:
    isdtype = get_xp(np)(_aliases.isdtype)

if hasattr(np, 'unstack'):
    unstack = np.unstack
else:
    unstack = get_xp(np)(_aliases.unstack)

__all__ = _aliases.__all__ + ['__array_namespace_info__', 'asarray', 'astype',
                              'acos', 'acosh', 'asin', 'asinh', 'atan',
                              'atan2', 'atanh', 'bitwise_left_shift',
                              'bitwise_invert', 'bitwise_right_shift',
                              'bool', 'concat', 'count_nonzero', 'pow']

_all_ignore = ['np', 'get_xp']
