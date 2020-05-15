import pytest

from pynum import ndarray


@pytest.mark.parametrize('shape, offset, strides', [
    ((), 0, ()),
    ((3,), 0, (1,)),
    ((3,), 5, (1,)),
    ((1, 2, 3,), 0, (6, 3, 1,)),
])
def test_indexer_init(shape, offset, strides):
    _ = ndarray.Indexer(shape, offset, strides)


@pytest.mark.parametrize('shape, offset, strides, expt_error', [
    ((), 0, (1,), ValueError),
    ((1,), 0, (), ValueError),

    (1, 0, (), TypeError),
    ((), 0, 1, TypeError),
])
def test_indexer_init_raises(shape, offset, strides, expt_error):
    with pytest.raises(expt_error):
        _ = ndarray.Indexer(shape, offset, strides)


@pytest.mark.parametrize('shape, expt_shape, expt_offset, expt_strides', [
    ((), (), 0, ()),
    ((3,), (3,), 0, (1,)),
    ((1, 2, 3,), (1, 2, 3,), 0, (6, 3, 1,)),
])
def test_indexer_make_basic(shape, expt_shape, expt_offset, expt_strides):
    indexer = ndarray.Indexer.make_basic(shape)

    assert indexer._shape == expt_shape
    assert indexer._offset == expt_offset
    assert indexer._strides == expt_strides
