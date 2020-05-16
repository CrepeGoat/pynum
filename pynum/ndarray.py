import itertools
import numbers
import operator


class Indexer:
    """A generic indexer object."""

    __slots__ = ('_offset', '_shape', '_strides')

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
                i * j for i, j in zip(shape, strides) if j < 0
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
        pass

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
        pass

    def sliced(self, index):
        """Make a new indexer for a slice of the data."""
        pass


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
