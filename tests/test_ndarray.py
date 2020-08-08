import operator

import pytest
from unittest import mock
import sympy as sp

from pynum import ndarray, indexer


@pytest.mark.skip('function too simple')
def test_ndarray_init():
    pass


@pytest.mark.parametrize('flat_array, indexer, expt_error', [
    (
        [1, 2, 3], indexer.Indexer(shape=(4,), offset=0, strides=(1,)),
        ValueError
    ),
    (
        [1, 2, 3], indexer.Indexer(shape=(3,), offset=-1, strides=(1,)),
        ValueError
    ),
    (
        [1, 2, 3], indexer.Indexer(shape=(3,), offset=1, strides=(1,)),
        ValueError
    ),
    (
        [1, 2, 3], indexer.Indexer(shape=(3,), offset=0, strides=(-1,)),
        ValueError
    ),
    (
        [1, 2, 3], indexer.Indexer(shape=(3,), offset=0, strides=(2,)),
        ValueError
    ),
    (
        [1, 2, 3, 4, 5, 6],
        indexer.Indexer(shape=(3, 3,), offset=0, strides=(3, 1,)),
        ValueError
    ),
    (
        [1, 2, 3, 4, 5, 6],
        indexer.Indexer(shape=(2, 3,), offset=-1, strides=(3, 1,)),
        ValueError
    ),
    (
        [1, 2, 3, 4, 5, 6],
        indexer.Indexer(shape=(2, 3,), offset=1, strides=(3, 1,)),
        ValueError
    ),
    (
        [1, 2, 3, 4, 5, 6],
        indexer.Indexer(shape=(2, 3,), offset=0, strides=(3, 2,)),
        ValueError
    ),
])
def test_ndarray_init_raises(flat_array, indexer, expt_error):
    with pytest.raises(expt_error):
        _ = ndarray.NDArray(flat_array, indexer)


@pytest.mark.parametrize('values, expt_result', [
    (
        1, ndarray.NDArray([1], indexer.Indexer.make_basic(shape=()))
    ),
    (
        [1, 2, 3], ndarray.NDArray(
            [1, 2, 3],
            indexer.Indexer.make_basic(shape=(3,))
        ),
    ),
    (
        [[1, 2, 3], [4, 5, 6]], ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer.make_basic(shape=(2, 3))
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
        ndarray.NDArray([1], indexer.Indexer.make_basic(shape=())), 1
    ),
    (
        ndarray.NDArray(
            [1, 2, 3],
            indexer.Indexer.make_basic(shape=(3,))
        ), [1, 2, 3],
    ),
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer.make_basic(shape=(2, 3))
        ), [[1, 2, 3], [4, 5, 6]],
    ),
])
def test_ndarray_to_list(array, expt_result):
    assert array.to_list() == expt_result


@pytest.mark.parametrize('array1, array2', [
    # 0D
    (
        ndarray.NDArray([1], indexer.Indexer.make_basic(shape=())),
        ndarray.NDArray([1], indexer.Indexer.make_basic(shape=())),
    ),
    (
        ndarray.NDArray([1], indexer.Indexer.make_basic(shape=())),
        ndarray.NDArray([1, 2, 3], indexer.Indexer.make_basic(shape=())),
    ),
    (
        ndarray.NDArray([1], indexer.Indexer.make_basic(shape=())),
        ndarray.NDArray([1, 2, 3], indexer.Indexer.make_basic(shape=(1,))),
    ),

    # 1D
    (
        ndarray.NDArray([1, 2, 3], indexer.Indexer.make_basic(shape=(3,))),
        ndarray.NDArray([1, 2, 3], indexer.Indexer.make_basic(shape=(3,))),
    ),
    (
        ndarray.NDArray([1, 2, 3], indexer.Indexer.make_basic(shape=(3,))),
        ndarray.NDArray(
            [0, 0, 1, 0, 2, 0, 3],
            indexer.Indexer(shape=(3,), offset=2, strides=(2,)),
        ),
    ),
    (
        ndarray.NDArray([1, 2, 3], indexer.Indexer.make_basic(shape=(3,))),
        ndarray.NDArray([2, 2, 3], indexer.Indexer.make_basic(shape=(3,))),
    ),
    (
        ndarray.NDArray([1, 2, 3], indexer.Indexer.make_basic(shape=(2,))),
        ndarray.NDArray(
            [1, 2, 3],
            indexer.Indexer(shape=(2,), offset=1, strides=(1,))),
    ),
    (
        ndarray.NDArray([1, 2, 3], indexer.Indexer.make_basic(shape=(3,))),
        ndarray.NDArray([3, 2, 1], indexer.Indexer.make_basic(shape=(2,))),
    ),
    (
        ndarray.NDArray([1, 2, 3], indexer.Indexer.make_basic(shape=(3,))),
        ndarray.NDArray([1, 2, 3], indexer.Indexer.make_basic(shape=(1, 3))),
    ),

    # 2D
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer.make_basic(shape=(2, 3))
        ),
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer.make_basic(shape=(2, 3))
        ),
    ),
    (
        ndarray.NDArray(
            [1, 2, 0, 3, 4, 0],
            indexer.Indexer(shape=(2, 2), offset=0, strides=(3, 1))
        ),
        ndarray.NDArray(
            [1, 0, 3, 2, 0, 4, 0, 0],
            indexer.Indexer(shape=(2, 2), offset=0, strides=(2, 3))
        ),
    ),
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer.make_basic(shape=(3, 2))
        ),
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer.make_basic(shape=(2, 3))
        ),
    ),
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 0],
            indexer.Indexer.make_basic(shape=(2, 3))
        ),
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer.make_basic(shape=(2, 3))
        ),
    ),
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer.make_basic(shape=(2, 2))
        ),
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer(shape=(2, 2), offset=1, strides=(2, 1))
        ),
    ),
    (
        ndarray.NDArray([1, 2, 3, 4], indexer.Indexer.make_basic(shape=(4,))),
        ndarray.NDArray([1, 2, 3, 4], indexer.Indexer.make_basic(shape=(2, 2))),
    ),
    (
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer(shape=(2, 2), offset=0, strides=(3, 1))
        ),
        ndarray.NDArray(
            [1, 2, 3, 4, 5, 6],
            indexer.Indexer.make_basic(shape=(2, 3))
        ),
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


@pytest.mark.parametrize('array, index, expt_result', [
    (ndarray.NDArray.from_values(1), (), 1),

    (
        ndarray.NDArray.from_values([1, 2, 3]), (),
        ndarray.NDArray.from_values([1, 2, 3])
    ),
    (
        ndarray.NDArray.from_values([1, 2, 3]), slice(1, None),
        ndarray.NDArray.from_values([2, 3])
    ),
    (ndarray.NDArray.from_values([1, 2, 3]), 1, 2),

    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]), (),
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]), 0,
        ndarray.NDArray.from_values([1, 2, 3])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]), (0, Ellipsis),
        ndarray.NDArray.from_values([1, 2, 3])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]), slice(1, None),
        ndarray.NDArray.from_values([[4, 5, 6]])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]), (slice(None), 1),
        ndarray.NDArray.from_values([2, 5])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]), (Ellipsis, 1),
        ndarray.NDArray.from_values([2, 5])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]),
        (slice(None), slice(None, 2)),
        ndarray.NDArray.from_values([[1, 2], [4, 5]])
    ),
])
def test_ndarray_getitem(array, index, expt_result):
    assert array[index] == expt_result


@pytest.mark.parametrize('array, index, value, expt_result', [
    (
        ndarray.NDArray.from_values(1), (), 2,
        ndarray.NDArray.from_values(2)
    ),
    (
        ndarray.NDArray.from_values([1, 2, 3]), (), [1, 2, 3],
        ndarray.NDArray.from_values([1, 2, 3])
    ),
    (
        ndarray.NDArray.from_values([1, 2, 3]), slice(1, None), [4, 5],
        ndarray.NDArray.from_values([1, 4, 5])
    ),
    (
        ndarray.NDArray.from_values([1, 2, 3]), 1, 4,
        ndarray.NDArray.from_values([1, 4, 3])
    ),
    (
        ndarray.NDArray.from_values([1, 2, 3]), slice(1, None), 4,
        ndarray.NDArray.from_values([1, 4, 4])
    ),


    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]),
        (), [[6, 7, 8], [9, 10, 11]],
        ndarray.NDArray.from_values([[6, 7, 8], [9, 10, 11]])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]),
        0, [7, 8, 9],
        ndarray.NDArray.from_values([[7, 8, 9], [4, 5, 6]])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]),
        (0, Ellipsis), [7, 8, 9],
        ndarray.NDArray.from_values([[7, 8, 9], [4, 5, 6]])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]),
        slice(1, None), [[7, 8, 9]],
        ndarray.NDArray.from_values([[1, 2, 3], [7, 8, 9]])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]),
        (slice(None), 1), [7, 8],
        ndarray.NDArray.from_values([[1, 7, 3], [4, 8, 6]])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]),
        (Ellipsis, 1), [7, 8],
        ndarray.NDArray.from_values([[1, 7, 3], [4, 8, 6]])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]),
        (slice(None), slice(None, 2)), [[7, 8], [9, 10]],
        ndarray.NDArray.from_values([[7, 8, 3], [9, 10, 6]])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]),
        (slice(None), slice(None, 2)), 7,
        ndarray.NDArray.from_values([[7, 7, 3], [7, 7, 6]])
    ),
    (
        ndarray.NDArray.from_values([[1, 2, 3], [4, 5, 6]]),
        (slice(None), slice(None, 2)), [7, 8],
        ndarray.NDArray.from_values([[7, 8, 3], [7, 8, 6]])
    ),
])
def test_ndarray_setitem(array, index, value, expt_result):
    array[index] = value
    assert array == expt_result


def test_ndarray_applied_elementwise():
    op = sp.Function("f")
    array1 = [sp.symbols("x0:3"), sp.symbols("x3:6")]
    array2 = [sp.symbols("y0:3"), sp.symbols("y3:6")]
    expt_result = ndarray.NDArray.from_values([
        [
            op(array1[0][0], array2[0][0]),
            op(array1[0][1], array2[0][1]),
            op(array1[0][2], array2[0][2]),
        ],
        [
            op(array1[1][0], array2[1][0]),
            op(array1[1][1], array2[1][1]),
            op(array1[1][2], array2[1][2]),
        ],
    ])

    result = ndarray.NDArray._applied_elementwise(op, array1, array2)
    assert result == expt_result


@pytest.mark.parametrize('method, operator', [
    (ndarray.NDArray.__lt__, operator.lt),
    (ndarray.NDArray.__le__, operator.le),
    (ndarray.NDArray.eq, operator.eq),
    (ndarray.NDArray.__ne__, operator.ne),
    (ndarray.NDArray.__ge__, operator.ge),
    (ndarray.NDArray.__gt__, operator.gt),
    (ndarray.NDArray.__add__, operator.add),
    (ndarray.NDArray.__sub__, operator.sub),
    (ndarray.NDArray.__mul__, operator.mul),
    (ndarray.NDArray.__truediv__, operator.truediv),
    (ndarray.NDArray.__floordiv__, operator.floordiv),
    (ndarray.NDArray.__mod__, operator.mod),
    (ndarray.NDArray.__pow__, operator.pow),
    (ndarray.NDArray.__lshift__, operator.lshift),
    (ndarray.NDArray.__rshift__, operator.rshift),
    (ndarray.NDArray.__and__, operator.and_),
    (ndarray.NDArray.__xor__, operator.xor),
    (ndarray.NDArray.__or__, operator.or_),
])
def test_ndarray_operators(method, operator):
    obj = mock.Mock(spec=ndarray.NDArray)
    obj._applied_elementwise = mock.Mock()

    method(obj, mock.sentinel.rhs)

    obj._applied_elementwise.assert_called_once_with(
        operator, obj, mock.sentinel.rhs,
    )

@pytest.mark.parametrize('method, operator', [
    (ndarray.NDArray.__radd__, operator.add),
    (ndarray.NDArray.__rsub__, operator.sub),
    (ndarray.NDArray.__rmul__, operator.mul),
    (ndarray.NDArray.__rtruediv__, operator.truediv),
    (ndarray.NDArray.__rfloordiv__, operator.floordiv),
    (ndarray.NDArray.__rmod__, operator.mod),
    (ndarray.NDArray.__rpow__, operator.pow),
])
def test_ndarray_rev_operators(method, operator):
    obj = mock.Mock(spec=ndarray.NDArray)
    obj._applied_elementwise = mock.Mock()

    method(obj, mock.sentinel.rhs)

    obj._applied_elementwise.assert_called_once_with(
        operator, mock.sentinel.rhs, obj,
    )


def test_ndarray_modify_elementwise():
    op = sp.Function("f")
    array1 = ndarray.NDArray.from_values(
        [sp.symbols("x0:3"), sp.symbols("x3:6")]
    )
    array2 = ndarray.NDArray.from_values(
        [sp.symbols("y0:3"), sp.symbols("y3:6")]
    )
    expt_result = ndarray.NDArray.from_values([
        [
            op(array1[0][0], array2[0][0]),
            op(array1[0][1], array2[0][1]),
            op(array1[0][2], array2[0][2]),
        ],
        [
            op(array1[1][0], array2[1][0]),
            op(array1[1][1], array2[1][1]),
            op(array1[1][2], array2[1][2]),
        ],
    ])

    array1._modify_elementwise(op, array2)
    assert array1 == expt_result


@pytest.mark.parametrize('method, operator', [
    (ndarray.NDArray.__iadd__, operator.iadd),
    (ndarray.NDArray.__isub__, operator.isub),
    (ndarray.NDArray.__imul__, operator.imul),
    (ndarray.NDArray.__itruediv__, operator.itruediv),
    (ndarray.NDArray.__ifloordiv__, operator.ifloordiv),
    (ndarray.NDArray.__imod__, operator.imod),
    (ndarray.NDArray.__ipow__, operator.ipow),
])
def test_ndarray_assignment_operators(method, operator):
    obj = mock.Mock(spec=ndarray.NDArray)
    obj._modify_elementwise = mock.Mock()

    method(obj, mock.sentinel.rhs)

    obj._modify_elementwise.assert_called_once_with(
        operator, mock.sentinel.rhs,
    )
