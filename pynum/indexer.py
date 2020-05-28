import functools
import itertools
import numbers
import operator


def _nd_indices(shape):
    """Generate sequence of dimension indices for a given shape."""
    return itertools.product(*(range(i) for i in shape))


def _nd_getitem(array, nd_index):
    """
    Get an item from an nd-array-like object.

    Meant for use on, e.g., nested lists. Does not accept slices.
    """
    if not isinstance(nd_index, tuple):
        raise ValueError
    if not all(isinstance(idx, numbers.Integral) for idx in nd_index):
        raise ValueError
    return functools.reduce(operator.getitem, nd_index, array)


def _nd_shape(array):
    """Calculate the dimensionality of the given nested list."""
    shape = []

    while True:
        dim_len = None  # None-len denotes scalars
        for i, nd_index in enumerate(_nd_indices(shape)):
            subarray = _nd_getitem(array, nd_index)

            try:
                subarray_len = len(subarray)
            except TypeError:
                subarray_len = None

            if dim_len != subarray_len:
                if i > 0:
                    raise ValueError("non-uniform dimension lengths")
                dim_len = subarray_len

        if dim_len is None:
            break
        shape.append(dim_len)

    return tuple(shape)


class Indexer:
    """A generic indexer object."""

    __slots__ = ('_shape', '_offset', '_strides')

    def __init__(self, shape, offset, strides):
        """Construct an instance."""
        if not isinstance(shape, tuple):
            shape = tuple(shape)
        if not isinstance(strides, tuple):
            strides = tuple(strides)

        if not all(isinstance(i, numbers.Integral) for i in shape):
            raise TypeError("shape dimensions must all be integral values")
        if not isinstance(offset, numbers.Integral):
            raise TypeError("index offset must be an integral value")
        if not all(isinstance(i, numbers.Integral) for i in strides):
            raise TypeError("strides must all be integral values")

        if not all(i >= 0 for i in shape):
            raise ValueError(
                "shape dimensions must all be non-negative values"
            )

        if len(shape) != len(strides):
            raise ValueError(
                "numbers of shape dimensions and strides must be equal"
            )

        self._shape = shape
        self._offset = offset
        self._strides = strides

    def __repr__(self):
        """Generate text representation of instance."""
        return (
            f"<{self.__class__.__name__}:"
            f" shape={self._shape},"
            f" offset={self._offset},"
            f" strides={self._strides}>"
        )

    @classmethod
    def make_basic(cls, shape):
        """Construct a basic, contiguous indexer object."""
        return cls(
            shape=shape,
            offset=0,
            strides=shape and tuple(
                itertools.accumulate((1,) + shape[:0:-1], operator.mul)
            )[::-1],
        )

    def __eq__(self, other):
        """
        Test for equality.

        Takes into account state redundancies; i.e., any dimension of size 1 or
        0 is equivalent, regardless of stride.
        """
        return (
            isinstance(other, self.__class__)
            and self._shape == other._shape
            and self._offset == other._offset
            and all(
                i == j or k in (0, 1)
                for i, j, k in zip(self._strides, other._strides, self._shape)
            )
        )

    def __iter__(self):
        """
        Generate individual flat-array indices from an nd-index.

        Iterates through dimensions starting with the last. E.g.,
        i0 -> array[0, 0, ..., 0, 0]
        i1 -> array[0, 0, ..., 0, 1]
        ...
        in-1 -> array[0, 0, ..., 0, -1]
        in -> array[0, 0, ..., 1, 0]
        """
        for nd_index in _nd_indices(self._shape):
            yield self._offset + sum(
                i*stride for i, stride in zip(nd_index, self._strides)
            )

    def __len__(self):
        """Calculate number of indices."""
        return functools.reduce(operator.mul, self._shape, 1)

    @property
    def min(self):
        if not self:
            raise ValueError
        return self._offset + sum(
            (i-1) * j
            for i, j in zip(self._shape, self._strides)
            if j < 0
        )

    @property
    def max(self):
        if not self:
            raise ValueError
        return self._offset + sum(
            (i-1) * j
            for i, j in zip(self._shape, self._strides)
            if j > 0
        )

    def added_dim(self, new_dim):
        """Create a copy of the indexer, adding an extra dimension."""
        if not -len(self._shape) <= new_dim <= len(self._shape):
            raise IndexError

        return self.__class__(
            shape=self._shape[:new_dim] + (1,) + self._shape[new_dim:],
            offset=self._offset,
            strides=self._strides[:new_dim] + (1,) + self._strides[new_dim:]
        )

    def sliced(self, index):
        """Make a new indexer for a slice of the data."""
        # Normalize single indices or index sequences into tuples
        if isinstance(index, (numbers.Integral, slice)):
            index = (index,)
        elif not isinstance(index, tuple):
            raise ValueError

        def slice_fill(index):
            """Create the slice(None) filling for an nd-index."""
            num = len(self._shape) - sum(1 for i in index if i is not Ellipsis)
            return (slice(None),) * num

        # Strip out ellipses
        try:
            i_mid = index.index(Ellipsis)
        except ValueError:
            pass
        else:
            if Ellipsis in index[i_mid+1:]:
                raise ValueError
            index = index[:i_mid] + slice_fill(index) + index[i_mid+1:]
            assert len(index) == len(self._shape)

        # Fill empty dims with full slices
        if len(index) < len(self._shape):
            index = index + slice_fill(index)
            assert len(index) == len(self._shape)
        elif len(index) > len(self._shape):
            raise ValueError

        # Map slices/single indices to literal coordinates
        index = tuple(
            range(dim)[idx]
            for idx, dim in zip(index, self._shape)
        )

        # Calculate parameters
        result = self.__class__(
            shape=tuple(
                len(idx)
                for idx in index
                if not isinstance(idx, numbers.Integral)
            ),
            offset=self._offset + sum(
                (i if isinstance(i, numbers.Integral) else i.start) * stride
                for i, stride in zip(index, self._strides)
            ),
            strides=tuple(
                idx.step * stride
                for idx, stride in zip(index, self._strides)
                if not isinstance(idx, numbers.Integral)
            ),
        )

        return result

    @classmethod
    def mutually_broadcasted(cls, indexer1, indexer2):
        """Make two indexer objects broadcasted to the same shape."""
        if not isinstance(indexer1, cls) or not isinstance(indexer2, cls):
            raise TypeError

        shape1 = list(indexer1._shape)
        strides1 = list(indexer1._strides)

        shape2 = list(indexer2._shape)
        strides2 = list(indexer2._strides)

        ndim_diff = len(shape2) - len(shape1)
        if ndim_diff >= 0:
            shape1 = ndim_diff*[1] + shape1
            strides1 = ndim_diff*[1] + strides1
        else:
            shape2 = (-ndim_diff)*[1] + shape2
            strides2 = (-ndim_diff)*[1] + strides2
        assert len(shape1) == len(shape2) == len(strides1) == len(strides2)

        for i in range(len(shape1))[::-1]:
            dim1 = shape1[i]
            dim2 = shape2[i]
            if dim1 == dim2:
                continue
            if dim1 == 1:
                shape1[i] = dim2
                strides1[i] = 0
            elif dim2 == 1:
                shape2[i] = dim1
                strides2[i] = 0
            else:
                raise ValueError

        result1 = cls(
            shape=shape1,
            offset=indexer1._offset,
            strides=strides1,
        )
        result2 = cls(
            shape=shape2,
            offset=indexer2._offset,
            strides=strides2,
        )

        return result1, result2

    def broadcasted_to(self, new_shape):
        """Copy this indexer object broadcasted to the given shape."""
        if not isinstance(new_shape, tuple):
            raise TypeError

        shape_self = list(self._shape)
        strides_self = list(self._strides)

        ndim_diff = len(new_shape) - len(shape_self)
        if ndim_diff < 0:
            raise ValueError
        shape_self = ndim_diff*[1] + shape_self
        strides_self = ndim_diff*[1] + strides_self
        assert len(shape_self) == len(strides_self) == len(new_shape)

        for i in range(len(shape_self))[::-1]:
            dim_self, dim_other = shape_self[i], new_shape[i]
            if dim_self == dim_other:
                continue
            if dim_self != 1:
                raise ValueError

            shape_self[i] = dim_other
            strides_self[i] = 0

        result = self.__class__(
            shape=shape_self,
            offset=self._offset,
            strides=strides_self,
        )

        return result
