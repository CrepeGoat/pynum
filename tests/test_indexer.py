import pytest

from pynum import indexer


@pytest.mark.parametrize('shape, expt_indices', [
    ((), [()]),
    ((3,), [(0,), (1,), (2,)]),
    ((2, 7,), [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    ]),
])
def test_nd_indices(shape, expt_indices):
    assert list(indexer._nd_indices(shape)) == expt_indices


@pytest.mark.parametrize('array, nd_indices, expt_values', [
    (1, [()], [1]),
    ([1, 2, 3], [(0,), (1,), (2,), ()], [1, 2, 3, [1, 2, 3]]),
    (
        [[1, 2, 3], [4, 5, 6]],
        [(0, 0), (1, 2), (0,), ()],
        [1, 6, [1, 2, 3], [[1, 2, 3], [4, 5, 6]]]
    ),
])
def test_nd_getitem(array, nd_indices, expt_values):
    assert [
        indexer._nd_getitem(array, nd_index)
        for nd_index in nd_indices
    ] == expt_values


@pytest.mark.parametrize('array, expt_shape', [
    (1, ()),
    ([1, 2, 3], (3,)),
    ([[1, 2, 3], [4, 5, 6]], (2, 3)),
])
def test_nd_shape(array, expt_shape):
    assert indexer._nd_shape(array) == expt_shape


@pytest.mark.parametrize('array, expt_error', [
    ([1, 2, 3, [4, 5, 6]], ValueError),
])
def test_nd_shape_raises(array, expt_error):
    with pytest.raises(expt_error):
        _ = indexer._nd_shape(array)


###############################################################################

@pytest.mark.parametrize('shape, offset, strides', [
    ((), 0, ()),
    ((3,), 0, (1,)),
    ((3,), 5, (1,)),
    ((1, 2, 3,), 0, (6, 3, 1,)),
    ((3,), 2, (-1,))
])
def test_indexer_init(shape, offset, strides):
    _ = indexer.Indexer(shape, offset, strides)


@pytest.mark.parametrize('shape, offset, strides, expt_error', [
    ((), 0, (1,), ValueError),
    ((1,), 0, (), ValueError),

    (1, 0, (), TypeError),
    ((), 0, 1, TypeError),

    ((1.,), 0, (1,), TypeError),
    ((1,), 0., (1,), TypeError),
    ((1,), 0, (1.,), TypeError),

    ((-1,), 0, (1,), ValueError),
])
def test_indexer_init_raises(shape, offset, strides, expt_error):
    with pytest.raises(expt_error):
        _ = indexer.Indexer(shape, offset, strides)


@pytest.mark.parametrize('shape, expt_shape, expt_offset, expt_strides', [
    ((), (), 0, ()),
    ((3,), (3,), 0, (1,)),
    ((1, 2, 3,), (1, 2, 3,), 0, (6, 3, 1,)),
])
def test_indexer_make_basic(shape, expt_shape, expt_offset, expt_strides):
    idxr = indexer.Indexer.make_basic(shape)

    assert idxr._shape == expt_shape
    assert idxr._offset == expt_offset
    assert idxr._strides == expt_strides


@pytest.mark.parametrize('shape', [(), (3,), (1, 2, 3,)])
def test_indexer_basic_iter_len(shape):
    idxr = indexer.Indexer.make_basic(shape)
    assert tuple(idxr) == tuple(range(len(idxr)))


@pytest.mark.parametrize('idxr, expt_indices', [
    (indexer.Indexer(shape=(3,), offset=2, strides=(1,)), (2, 3, 4)),
    (indexer.Indexer(shape=(3,), offset=0, strides=(3,)), (0, 3, 6)),

    (
        indexer.Indexer(shape=(3, 2), offset=0, strides=(5, 7)),
        (0, 7, 5, 12, 10, 17),
    ),
])
def test_indexer_iter_len(idxr, expt_indices):
    assert tuple(idxr) == expt_indices
    assert len(idxr) == len(expt_indices)


@pytest.mark.parametrize('idxr', [
    indexer.Indexer.make_basic(shape=()),
    indexer.Indexer.make_basic(shape=(3,)),
    indexer.Indexer.make_basic(shape=(2, 3,)),

    indexer.Indexer(shape=(5,), offset=3, strides=(-1,)),
    indexer.Indexer(shape=(3, 5), offset=-2, strides=(-5, 15)),
])
def test_indexer_min(idxr):
    assert min(idxr) == idxr.min


@pytest.mark.parametrize('idxr', [
    indexer.Indexer.make_basic(shape=()),
    indexer.Indexer.make_basic(shape=(3,)),
    indexer.Indexer.make_basic(shape=(2, 3,)),

    indexer.Indexer(shape=(5,), offset=3, strides=(-1,)),
    indexer.Indexer(shape=(3, 5), offset=-2, strides=(-5, 15)),
])
def test_indexer_max(idxr):
    assert max(idxr) == idxr.max


@pytest.mark.parametrize('idxr1, idxr2', [
    (
        indexer.Indexer.make_basic(shape=()),
        indexer.Indexer.make_basic(shape=()),
    ),
    (
        indexer.Indexer.make_basic(shape=(3,)),
        indexer.Indexer.make_basic(shape=(3,)),
    ),
    (
        indexer.Indexer.make_basic(shape=(3,)),
        indexer.Indexer.make_basic(shape=(5,)),
    ),
    (
        indexer.Indexer.make_basic(shape=(3, 5)),
        indexer.Indexer.make_basic(shape=(5, 3)),
    ),
    (
        indexer.Indexer.make_basic(shape=(3, 5)),
        indexer.Indexer.make_basic(shape=(1, 3, 5)),
    ),

    (
        indexer.Indexer(shape=(3, 5), offset=0, strides=(5, 1)),
        indexer.Indexer(shape=(3, 5), offset=2, strides=(5, 1)),
    ),
    (
        indexer.Indexer(shape=(3, 5), offset=0, strides=(1, 5)),
        indexer.Indexer(shape=(3, 5), offset=0, strides=(5, 1)),
    ),
])
def test_indexer_eq(idxr1, idxr2):
    assert (idxr1 == idxr2) == (
        idxr1._shape == idxr2._shape
        and tuple(idxr1) == tuple(idxr2)
    )


@pytest.mark.parametrize('idxr, dim', [
    (indexer.Indexer.make_basic(shape=()), 0),

    (indexer.Indexer.make_basic(shape=(3,)), 0),
    (indexer.Indexer.make_basic(shape=(3,)), 1),
    (indexer.Indexer.make_basic(shape=(3,)), -1),

    (indexer.Indexer.make_basic(shape=(5, 3, 7)), 0),
    (indexer.Indexer.make_basic(shape=(5, 3, 7)), 1),
    (indexer.Indexer.make_basic(shape=(5, 3, 7)), 2),
    (indexer.Indexer.make_basic(shape=(5, 3, 7)), 3),
    (indexer.Indexer.make_basic(shape=(5, 3, 7)), -1),
    (indexer.Indexer.make_basic(shape=(5, 3, 7)), -2),
    (indexer.Indexer.make_basic(shape=(5, 3, 7)), -3),
])
def test_indexer_added_dim(idxr, dim):
    new_idxr = idxr.added_dim(dim)
    assert new_idxr._shape == (
        idxr._shape[:dim] + (1,) + idxr._shape[dim:]
    )
    assert (
        new_idxr._strides[:dim if dim >= 0 else dim-1]
        + new_idxr._strides[dim+1 if dim >= 0 else dim:]
    ) == idxr._strides
    assert new_idxr is not idxr


@pytest.mark.parametrize('idxr, dim, expt_error', [
    (indexer.Indexer.make_basic(shape=(3,)), 2, IndexError),
    (indexer.Indexer.make_basic(shape=(3,)), -2, IndexError),

    (indexer.Indexer.make_basic(shape=(5, 3, 7)), 4, IndexError),
    (indexer.Indexer.make_basic(shape=(5, 3, 7)), -4, IndexError),
])
def test_indexer_added_dim_raises(idxr, dim, expt_error):
    with pytest.raises(expt_error):
        _ = idxr.added_dim(dim)


@pytest.mark.parametrize('idxr, index, expt_indexer', [
    (
        indexer.Indexer.make_basic(shape=()), (),
        indexer.Indexer(shape=(), offset=0, strides=()),
    ),

    (
        indexer.Indexer.make_basic(shape=(3,)), 0,
        indexer.Indexer(shape=(), offset=0, strides=()),
    ),
    (
        indexer.Indexer.make_basic(shape=(3,)), 1,
        indexer.Indexer(shape=(), offset=1, strides=()),
    ),
    (
        indexer.Indexer.make_basic(shape=(3,)), slice(1, None),
        indexer.Indexer(shape=(2,), offset=1, strides=(1,)),
    ),
    (
        indexer.Indexer.make_basic(shape=(3,)), slice(None, None, -1),
        indexer.Indexer(shape=(3,), offset=2, strides=(-1,)),
    ),

    (
        indexer.Indexer.make_basic(shape=(3,)), (Ellipsis, 1),
        indexer.Indexer(shape=(), offset=1, strides=()),
    ),
    (
        indexer.Indexer.make_basic(shape=(3,)), (Ellipsis, slice(1, None)),
        indexer.Indexer(shape=(2,), offset=1, strides=(1,)),
    ),

    (
        indexer.Indexer.make_basic(shape=(5, 3, 7)), 0,
        indexer.Indexer(shape=(3, 7), offset=0, strides=(7, 1)),
    ),
    (
        indexer.Indexer.make_basic(shape=(5, 3, 7)), 3,
        indexer.Indexer(shape=(3, 7), offset=63, strides=(7, 1)),
    ),

    (
        indexer.Indexer.make_basic(shape=(5, 3, 7)), slice(None),
        indexer.Indexer(shape=(5, 3, 7), offset=0, strides=(21, 7, 1)),
    ),
    (
        indexer.Indexer.make_basic(shape=(5, 3, 7)), slice(1, None),
        indexer.Indexer(shape=(4, 3, 7), offset=21, strides=(21, 7, 1)),
    ),
    (
        indexer.Indexer.make_basic(shape=(5, 3, 7)), slice(None, None, -1),
        indexer.Indexer(shape=(5, 3, 7), offset=84, strides=(-21, 7, 1)),
    ),

    (
        indexer.Indexer.make_basic(shape=(5, 3, 7)),
        (slice(None), 1),
        indexer.Indexer(shape=(5, 7), offset=7, strides=(21, 1)),
    ),
    (
        indexer.Indexer.make_basic(shape=(5, 3, 7)),
        (slice(None), slice(1, None)),
        indexer.Indexer(shape=(5, 2, 7), offset=7, strides=(21, 7, 1)),
    ),

    (
        indexer.Indexer.make_basic(shape=(5, 3, 7)),
        (Ellipsis, 1),
        indexer.Indexer(shape=(5, 3), offset=1, strides=(21, 7)),
    ),
    (
        indexer.Indexer.make_basic(shape=(5, 3, 7)),
        (Ellipsis, slice(1, None)),
        indexer.Indexer(shape=(5, 3, 6), offset=1, strides=(21, 7, 1)),
    ),
])
def test_indexer_sliced(idxr, index, expt_indexer):
    assert idxr.sliced(index) == expt_indexer


def test_indexer_mutually_broadcasted():
    idxr1 = indexer.Indexer.make_basic(shape=(1, 3))
    idxr2 = indexer.Indexer.make_basic(shape=(5, 2, 1))

    new_idxr1, new_idxr2 = indexer.Indexer.mutually_broadcasted(idxr1, idxr2)

    assert new_idxr1._shape == new_idxr2._shape == (5, 2, 3)
    assert all(
        new_idxr1.sliced((i, slice(j, j+1), slice(None))) == idxr1
        for i in range(5)
        for j in range(2)
    )
    assert all(
        new_idxr2.sliced((slice(None), slice(None), slice(i, i+1))) == idxr2
        for i in range(3)
    )


@pytest.mark.parametrize('idxr1, idxr2, expt_error', [
    (indexer.Indexer.make_basic(shape=(1, 2)), 1, TypeError),
    ([1, 2, 3], indexer.Indexer.make_basic(shape=(1, 2)), TypeError),

    (
        indexer.Indexer.make_basic(shape=(2, 3, 4)),
        indexer.Indexer.make_basic(shape=(4, 6)),
        ValueError,
    ),
    (
        indexer.Indexer.make_basic(shape=(1, 2, 3)),
        indexer.Indexer.make_basic(shape=(6,)),
        ValueError,
    ),
    (
        indexer.Indexer.make_basic(shape=(1, 2)),
        indexer.Indexer.make_basic(shape=(3, 4)),
        ValueError,
    ),
])
def test_indexer_mutually_broadcasted_raises(idxr1, idxr2, expt_error):
    with pytest.raises(expt_error):
        indexer.Indexer.mutually_broadcasted(idxr1, idxr2)


def test_indexer_broadcasted_to():
    idxr_sub = indexer.Indexer.make_basic(shape=(3, 1))
    dom_shape = (5, 3, 2)

    new_idxr_sub = idxr_sub.broadcasted_to(dom_shape)

    assert new_idxr_sub._shape == dom_shape
    assert all(
        new_idxr_sub.sliced((i, slice(None), slice(j, j+1))) == idxr_sub
        for i in range(5)
        for j in range(2)
    )


@pytest.mark.parametrize('idxr_sub, new_shape, expt_error', [
    (indexer.Indexer.make_basic(shape=(1, 2)), 1, TypeError),

    (indexer.Indexer.make_basic(shape=(4, 6)), (2, 3, 4), ValueError),
    (indexer.Indexer.make_basic(shape=(6,)), (1, 2, 3), ValueError),
    (indexer.Indexer.make_basic(shape=(1, 2)), (3, 4), ValueError),
    (indexer.Indexer.make_basic(shape=(1, 2)), (3, 1), ValueError),
])
def test_indexer_broadcasted_to_raises(idxr_sub, new_shape, expt_error):
    with pytest.raises(expt_error):
        idxr_sub.broadcasted_to(new_shape)

