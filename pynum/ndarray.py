import functools
import itertools
import numbers
import operator


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

        if (
            all(i != 0 for i in shape)
            and offset + sum(
                (i-1) * j for i, j in zip(shape, strides) if j < 0
            ) < 0
        ):
            raise ValueError(
                "resulting indices must all be positive values"
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
        if not self._shape:
            return
        for dim_indices in itertools.product(*(range(i) for i in self._shape)):
            yield self._offset + sum(
                i*stride for i, stride in zip(dim_indices, self._strides)
            )

    def __len__(self):
        """Calculate number of indices."""
        if not self._shape:
            return 0
        return functools.reduce(operator.mul, self._shape)

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
            offset=sum(
                (i if isinstance(i, numbers.Integral) else i.start) * stride
                for i, stride in zip(index, self._strides)
            ),
            shape=tuple(
                len(idx)
                for idx in index
                if not isinstance(idx, numbers.Integral)
            ),
            strides=tuple(
                idx.step * stride
                for idx, stride in zip(index, self._strides)
                if not isinstance(idx, numbers.Integral)
            ),
        )

        return result


class NDArray:
    """A pure-python version of the numpy ndarray."""

    __slots__ = ('_flat_array', '_indexer')

    def __init__(self, flat_array, indexer):
        """Construct an instance."""
        self._flat_array = flat_array
        self._indexer = indexer

    def __getitem__(self, index):
        """Index/slice the data."""
        return NDArray(self._flat_array, self._indexer.sliced(index))

    def __setitem__(self, index, value):
        """Set values to an index/slice of the data."""
        for i, j in zip(self._indexer.sliced(index), value._indexer):
            self._flat_array[i] = value._flat_array[j]
