import pytest

from pynum import ndarray


@pytest.mark.parametrize('shape, expt_indices', [
    ((), [()]),
    ((3,), [(0,), (1,), (2,)]),
    ((2, 7,), [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    ]),
])
def test_nd_indices(shape, expt_indices):
    assert list(ndarray._nd_indices(shape)) == expt_indices


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
        ndarray._nd_getitem(array, nd_index)
        for nd_index in nd_indices
    ] == expt_values


@pytest.mark.parametrize('array, expt_shape', [
    (1, ()),
    ([1, 2, 3], (3,)),
    ([[1, 2, 3], [4, 5, 6]], (2, 3)),
])
def test_nd_shape(array, expt_shape):
    assert ndarray._nd_shape(array) == expt_shape


@pytest.mark.parametrize('array, expt_error', [
    ([1, 2, 3, [4, 5, 6]], ValueError),
])
def test_nd_shape_raises(array, expt_error):
    with pytest.raises(expt_error):
        _ = ndarray._nd_shape(array)


###############################################################################

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


@pytest.mark.parametrize('shape', [(), (3,), (1, 2, 3,)])
def test_indexer_basic_iter_len(shape):
    indexer = ndarray.Indexer.make_basic(shape)
    assert tuple(indexer) == tuple(range(len(indexer)))


@pytest.mark.parametrize('indexer, expt_indices', [
    (ndarray.Indexer(shape=(3,), offset=2, strides=(1,)), (2, 3, 4)),
    (ndarray.Indexer(shape=(3,), offset=0, strides=(3,)), (0, 3, 6)),

    (
        ndarray.Indexer(shape=(3, 2), offset=0, strides=(5, 7)),
        (0, 7, 5, 12, 10, 17),
    ),
])
def test_indexer_iter_len(indexer, expt_indices):
    assert tuple(indexer) == expt_indices
    assert len(indexer) == len(expt_indices)


@pytest.mark.parametrize('indexer1, indexer2', [
    (
        ndarray.Indexer.make_basic(shape=()),
        ndarray.Indexer.make_basic(shape=()),
    ),
    (
        ndarray.Indexer.make_basic(shape=(3,)),
        ndarray.Indexer.make_basic(shape=(3,)),
    ),
    (
        ndarray.Indexer.make_basic(shape=(3,)),
        ndarray.Indexer.make_basic(shape=(5,)),
    ),
    (
        ndarray.Indexer.make_basic(shape=(3, 5)),
        ndarray.Indexer.make_basic(shape=(5, 3)),
    ),
    (
        ndarray.Indexer.make_basic(shape=(3, 5)),
        ndarray.Indexer.make_basic(shape=(1, 3, 5)),
    ),

    (
        ndarray.Indexer(shape=(3, 5), offset=0, strides=(5, 1)),
        ndarray.Indexer(shape=(3, 5), offset=2, strides=(5, 1)),
    ),
    (
        ndarray.Indexer(shape=(3, 5), offset=0, strides=(1, 5)),
        ndarray.Indexer(shape=(3, 5), offset=0, strides=(5, 1)),
    ),
])
def test_indexer_eq(indexer1, indexer2):
    assert (indexer1 == indexer2) == (
        indexer1._shape == indexer2._shape
        and tuple(indexer1) == tuple(indexer2)
    )


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
        ndarray.Indexer.make_basic(shape=()), (),
        ndarray.Indexer(shape=(), offset=0, strides=()),
    ),

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


###############################################################################



###############################################################################

@pytest.mark.skip('function too simple')
def test_ndarray_init():
    pass


@pytest.mark.parametrize('values, expt_result', [
    (
        1, ndarray.NDArray([1], ndarray.Indexer.make_basic(shape=()))
    ),
    (
        [1, 2, 3], ndarray.NDArray(
            [1, 2, 3],
            ndarray.Indexer.make_basic(shape=(3,))
        ),
    ),
    (
        [[1, 2, 3], [4, 5, 6]], ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            ndarray.Indexer.make_basic(shape=(2, 3))
        ),
    ),
])
def test_ndarray_from_values(values, expt_result):
    result = ndarray.NDArray.from_values(values)
    assert result.shape == expt_result.shape
    assert all(
        result._flat_array[i] == expt_result._flat_array[j]
        for i, j in zip(result._indexer, expt_result._indexer)
    )


@pytest.mark.parametrize('array, expt_result', [
    (
        ndarray.NDArray([1], ndarray.Indexer.make_basic(shape=())), 1
    ),
    (
        ndarray.NDArray(
            [1, 2, 3],
            ndarray.Indexer.make_basic(shape=(3,))
        ), [1, 2, 3],
    ),
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            ndarray.Indexer.make_basic(shape=(2, 3))
        ), [[1, 2, 3], [4, 5, 6]],
    ),
])
def test_ndarray_to_list(array, expt_result):
    assert array.to_list() == expt_result


@pytest.mark.parametrize('array1, array2', [
    # 0D
    (
        ndarray.NDArray([], ndarray.Indexer.make_basic(shape=())),
        ndarray.NDArray([], ndarray.Indexer.make_basic(shape=())),
    ),
    (
        ndarray.NDArray([], ndarray.Indexer.make_basic(shape=())),
        ndarray.NDArray([1, 2, 3], ndarray.Indexer.make_basic(shape=())),
    ),
    (
        ndarray.NDArray([], ndarray.Indexer.make_basic(shape=())),
        ndarray.NDArray([1, 2, 3], ndarray.Indexer.make_basic(shape=(1,))),
    ),

    # 1D
    (
        ndarray.NDArray([1, 2, 3], ndarray.Indexer.make_basic(shape=(3,))),
        ndarray.NDArray([1, 2, 3], ndarray.Indexer.make_basic(shape=(3,))),
    ),
    (
        ndarray.NDArray([1, 2, 3], ndarray.Indexer.make_basic(shape=(3,))),
        ndarray.NDArray(
            [0, 0, 1, 0, 2, 0, 3],
            ndarray.Indexer(shape=(3,), offset=2, strides=(2,)),
        ),
    ),
    (
        ndarray.NDArray([1, 2, 3], ndarray.Indexer.make_basic(shape=(3,))),
        ndarray.NDArray([2, 2, 3], ndarray.Indexer.make_basic(shape=(3,))),
    ),
    (
        ndarray.NDArray([1, 2, 3], ndarray.Indexer.make_basic(shape=(2,))),
        ndarray.NDArray(
            [1, 2, 3],
            ndarray.Indexer(shape=(2,), offset=1, strides=(1,))),
    ),
    (
        ndarray.NDArray([1, 2, 3], ndarray.Indexer.make_basic(shape=(3,))),
        ndarray.NDArray([3, 2, 1], ndarray.Indexer.make_basic(shape=(2,))),
    ),
    (
        ndarray.NDArray([1, 2, 3], ndarray.Indexer.make_basic(shape=(3,))),
        ndarray.NDArray([1, 2, 3], ndarray.Indexer.make_basic(shape=(1, 3))),
    ),

    # 2D
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            ndarray.Indexer.make_basic(shape=(2, 3))
        ),
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            ndarray.Indexer.make_basic(shape=(2, 3))
        ),
    ),
    (
        ndarray.NDArray(
            [1, 2, 0, 3, 4, 0],
            ndarray.Indexer(shape=(2, 2), offset=0, strides=(3, 1))
        ),
        ndarray.NDArray(
            [1, 0, 3, 2, 0, 4, 0, 0],
            ndarray.Indexer(shape=(2, 2), offset=0, strides=(2, 3))
        ),
    ),
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            ndarray.Indexer.make_basic(shape=(3, 2))
        ),
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            ndarray.Indexer.make_basic(shape=(2, 3))
        ),
    ),
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 0],
            ndarray.Indexer.make_basic(shape=(2, 3))
        ),
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            ndarray.Indexer.make_basic(shape=(2, 3))
        ),
    ),
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            ndarray.Indexer.make_basic(shape=(2, 2))
        ),
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            ndarray.Indexer(shape=(2, 2), offset=1, strides=(2, 1))
        ),
    ),
    (
        ndarray.NDArray([1, 2, 3, 4], ndarray.Indexer.make_basic(shape=(4,))),
        ndarray.NDArray([1, 2, 3, 4], ndarray.Indexer.make_basic(shape=(2, 2))),
    ),
])
def test_ndarray_eq(array1, array2):
    expt_result = (
        array1.shape == array2.shape
        and all(
            array1._flat_array[i1] == array2._flat_array[i2]
            for i1, i2 in zip(array1._indexer, array2._indexer)
        )
    )
    assert (array1 == array2) == expt_result


def test_ndarray_getitem(array, index, expt_result):
    result = array[index]
    assert result.shape == expt_result.shape
    assert all(
        result._flat_array[i] == expt_result._flat_array[j]
        for i, j in zip(result._indexer, expt_result._indexer)
    )
