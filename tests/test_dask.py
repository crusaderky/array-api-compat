import dask
import numpy as np
import pytest
import dask.array as da

from array_api_compat import array_namespace


@pytest.fixture
def no_compute():
    """
    Cause the test to raise if at any point anything calls compute() or persist(),
    e.g. as it can be triggered implicitly by __bool__, __array__, etc.
    """
    def get(dsk, *args, **kwargs):
        raise AssertionError("Called compute() or persist()")
    
    with dask.config.set(scheduler=get):
        yield


def test_no_compute(no_compute):
    """Test the no_compute fixture"""
    a = da.asarray(True)
    with pytest.raises(AssertionError, match="Called compute"):
        bool(a)


def test_asarray_no_compute(no_compute):
    a = da.arange(10)
    xp = array_namespace(a)  # wrap in array_api_compat.dask.array

    xp.asarray(a)
    xp.asarray(a, dtype=np.int16)
    xp.asarray(a, dtype=a.dtype)
    xp.asarray(a, copy=True)
    xp.asarray(a, copy=True, dtype=np.int16)
    xp.asarray(a, copy=True, dtype=a.dtype)


def test_clip_no_compute(no_compute):
    a = da.arange(10)
    xp = array_namespace(a)  # wrap in array_api_compat.dask.array

    xp.clip(a)
    xp.clip(a, 1)
    xp.clip(a, 1, 8)
