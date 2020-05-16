import pytest

from pynum import ndarray


@pytest.mark.parametrize('shape, offset, strides', [
    ((), 0, ()),
    ((3,), 0, (1,)),
    ((3,), 5, (1,)),
    ((1, 2, 3,), 0, (6, 3, 1,)),
    ((3,), 2, (-1,))
])
def test_indexer_init(shape, offset, strides):
    _ = ndarray.Indexer(shape, offset, strides)


@pytest.mark.parametrize('shape, offset, strides, expt_error', [
    ((), 0, (1,), ValueError),
    ((1,), 0, (), ValueError),

    (1, 0, (), TypeError),
    ((), 0, 1, TypeError),

    ((1.,), 0, (1,), TypeError),
    ((1,), 0., (1,), TypeError),
    ((1,), 0, (1.,), TypeError),

    ((-1,), 0, (1,), ValueError),
    ((1,), -1, (1,), ValueError),
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


@pytest.mark.parametrize('indexer, dim', [
    (ndarray.Indexer.make_basic(shape=()), 0),

    (ndarray.Indexer.make_basic(shape=(3,)), 0),
    (ndarray.Indexer.make_basic(shape=(3,)), 1),
    (ndarray.Indexer.make_basic(shape=(3,)), -1),

    (ndarray.Indexer.make_basic(shape=(5, 3, 7)), 0),
    (ndarray.Indexer.make_basic(shape=(5, 3, 7)), 1),
    (ndarray.Indexer.make_basic(shape=(5, 3, 7)), 2),
    (ndarray.Indexer.make_basic(shape=(5, 3, 7)), 3),
    (ndarray.Indexer.make_basic(shape=(5, 3, 7)), -1),
    (ndarray.Indexer.make_basic(shape=(5, 3, 7)), -2),
    (ndarray.Indexer.make_basic(shape=(5, 3, 7)), -3),
])
def test_indexer_added_dim(indexer, dim):
    new_indexer = indexer.added_dim(dim)
    assert new_indexer._shape == (
        indexer._shape[:dim] + (1,) + indexer._shape[dim:]
    )
    assert (
        new_indexer._strides[:dim if dim >= 0 else dim-1]
        + new_indexer._strides[dim+1 if dim >= 0 else dim:]
    ) == indexer._strides
    assert new_indexer is not indexer


@pytest.mark.parametrize('indexer, dim, expt_error', [
    (ndarray.Indexer.make_basic(shape=(3,)), 2, IndexError),
    (ndarray.Indexer.make_basic(shape=(3,)), -2, IndexError),

    (ndarray.Indexer.make_basic(shape=(5, 3, 7)), 4, IndexError),
    (ndarray.Indexer.make_basic(shape=(5, 3, 7)), -4, IndexError),
])
def test_indexer_added_dim_raises(indexer, dim, expt_error):
    with pytest.raises(expt_error):
        _ = indexer.added_dim(dim)


@pytest.mark.parametrize('indexer, index, expt_indexer', [
    (
        ndarray.Indexer.make_basic(shape=(3,)), 0,
        ndarray.Indexer(shape=(), offset=0, strides=()),
    ),
    (
        ndarray.Indexer.make_basic(shape=(3,)), 1,
        ndarray.Indexer(shape=(), offset=1, strides=()),
    ),
    (
        ndarray.Indexer.make_basic(shape=(3,)), slice(1, None),
        ndarray.Indexer(shape=(2,), offset=1, strides=(1,)),
    ),
    (
        ndarray.Indexer.make_basic(shape=(3,)), slice(None, None, -1),
        ndarray.Indexer(shape=(3,), offset=2, strides=(-1,)),
    ),

    (
        ndarray.Indexer.make_basic(shape=(3,)), (Ellipsis, 1),
        ndarray.Indexer(shape=(), offset=1, strides=()),
    ),
    (
        ndarray.Indexer.make_basic(shape=(3,)), (Ellipsis, slice(1, None)),
        ndarray.Indexer(shape=(2,), offset=1, strides=(1,)),
    ),

    (
        ndarray.Indexer.make_basic(shape=(5, 3, 7)), 0,
        ndarray.Indexer(shape=(3, 7), offset=0, strides=(7, 1)),
    ),
    (
        ndarray.Indexer.make_basic(shape=(5, 3, 7)), 3,
        ndarray.Indexer(shape=(3, 7), offset=63, strides=(7, 1)),
    ),

    (
        ndarray.Indexer.make_basic(shape=(5, 3, 7)), slice(None),
        ndarray.Indexer(shape=(5, 3, 7), offset=0, strides=(21, 7, 1)),
    ),
    (
        ndarray.Indexer.make_basic(shape=(5, 3, 7)), slice(1, None),
        ndarray.Indexer(shape=(4, 3, 7), offset=21, strides=(21, 7, 1)),
    ),
    (
        ndarray.Indexer.make_basic(shape=(5, 3, 7)), slice(None, None, -1),
        ndarray.Indexer(shape=(5, 3, 7), offset=84, strides=(-21, 7, 1)),
    ),

    (
        ndarray.Indexer.make_basic(shape=(5, 3, 7)),
        (slice(None), 1),
        ndarray.Indexer(shape=(5, 7), offset=7, strides=(21, 1)),
    ),
    (
        ndarray.Indexer.make_basic(shape=(5, 3, 7)),
        (slice(None), slice(1, None)),
        ndarray.Indexer(shape=(5, 2, 7), offset=7, strides=(21, 7, 1)),
    ),

    (
        ndarray.Indexer.make_basic(shape=(5, 3, 7)),
        (Ellipsis, 1),
        ndarray.Indexer(shape=(5, 3), offset=1, strides=(21, 7)),
    ),
    (
        ndarray.Indexer.make_basic(shape=(5, 3, 7)),
        (Ellipsis, slice(1, None)),
        ndarray.Indexer(shape=(5, 3, 6), offset=1, strides=(21, 7, 1)),
    ),
])
def test_indexer_sliced(indexer, index, expt_indexer):
    assert indexer.sliced(index) == expt_indexer
