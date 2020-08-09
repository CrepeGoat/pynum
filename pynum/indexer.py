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


class Range:
    """
    A `range`-like class constructed with a length instead of a `stop` value.

    Because the length of this iterable is specified on construction and is not
    calculated from the `start`, `stop` and `step` values, it allows for
    `step=0`. Since doing so makes the four values redundant, the `stop`
    parameter is not required.
    """
    __slots__ = ('start', 'step', 'length')

    def __init__(self, *values):
        if len(values) > 3 or len(values) == 0:
            raise TypeError
        if not all(isinstance(i, numbers.Integral) for i in values):
            raise TypeError

        if len(values) == 2:
            values = (*values, 1)
        if len(values) == 1:
            values = (0, *values, 1)

        self.start, self.length, self.step = values

        if not self.length >= 0:
            raise ValueError

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, numbers.Integral):
            if not 0 <= index < self.length:
                raise IndexError
            return self.start + self.step*index

        if isinstance(index, slice):
            i_start, i_stop, i_step = slice.indices(self.length)
            return self.__class__(
                self.start + self.step*i_start,
                max(0, (i_stop - i_start) // i_step),
                self.step*i_step,
            )

        raise TypeError

    def __eq__(self, rhs):
        if isinstance(rhs, (self.__class__, range)):
            return (
                len(rhs) == self.length
                and (
                    rhs._start == self.start
                    or self.length == 0
                ) and (
                    rhs._step == self.step
                    or self.length <= 1
                )
            )

        return False

    @classmethod
    def from_range(cls, r):
        return cls(r.start, len(r), r.step)

    @property
    def min(self):
        if self.length == 0:
            raise ValueError
        return self.start + min(0, self.step) * (self.length - 1)

    @property
    def max(self):
        if self.length == 0:
            raise ValueError
        return self.start + max(0, self.step) * (self.length - 1)


class Indexer:
    """A generic indexer object."""

    __slots__ = ('_offset', '_dim_offsets')

    def __init__(self, offset, dim_offsets):
        """Construct an instance."""
        if not isinstance(dim_offsets, tuple):
            dim_offsets = tuple(dim_offsets)

        if not isinstance(offset, numbers.Integral):
            raise TypeError("index offset must be an integral value")
        if not all(
            isinstance(indices, Range)
            or all(isinstance(i, numbers.Integral) for i in indices)
            for indices in dim_offsets
        ):
            raise TypeError("dimension offsets must all be integral values")

        self._offset = offset
        self._dim_offsets = dim_offsets

    def __repr__(self):
        """Generate text representation of instance."""
        return (
            f"<{self.__class__.__name__}:"
            f" offset={self._offset},"
            f" dim offsets={self._dim_offsets}>"
        )

    @classmethod
    def make_basic(cls, shape):
        """Construct a basic, contiguous indexer object."""
        return cls(
            offset=0,
            dim_offsets=tuple(Range(0, length, step) for length, step in zip(
                shape,
                (
                    (1,) + itertools.accumulate(shape[:0:-1], operator.mul)
                )[:-len(shape):-1]
            ))
        )

    def __eq__(self, other):
        """
        Test for equality.

        Takes into account state redundancies; i.e., any dimension of size 1 or
        0 is equivalent, regardless of stride.
        """
        return (
            isinstance(other, self.__class__)
            and self._offset == other._offset
            and all(
                dim1 == dim2
                or all(i1 == i2 for i1, i2 in zip(dim1, dim2))
                for dim1, dim2 in zip(self._dim_offsets, other._dim_offsets)
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
        return (
            sum(indices)
            for indices in itertools.product(self._dim_offsets)
        )

    def __len__(self):
        """Calculate number of indices."""
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def ndim(self):
        return len(self._dim_offsets)

    @property
    def shape(self):
        return (len(dim) for dim in self._dim_offsets)

    @property
    def min(self):
        if not self:
            raise ValueError
        return self._offset + sum(dim.min for dim in self._dim_offsets)

    @property
    def max(self):
        if not self:
            raise ValueError
        return self._offset + sum(dim.max for dim in self._dim_offsets)

    def added_dim(self, new_dim):
        """Create a copy of the indexer, adding an extra dimension."""
        if not -len(self._dim_offsets) <= new_dim <= len(self._dim_offsets):
            raise IndexError

        return self.__class__(
            offset=self._offset,
            dim_offsets=(
                self._dim_offsets[:new_dim]
                + Range(1)
                + self._dim_offsets[new_dim:]
            )
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
            num = self.ndim - sum(1 for i in index if i is not Ellipsis)
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
            assert len(index) == self.ndim

        # Fill empty dims with full slices
        if len(index) < self.ndim:
            index = index + slice_fill(index)
            assert len(index) == self.ndim
        elif len(index) > self.ndim:
            raise ValueError

        # Map slices/single indices to literal coordinates
        index = tuple(dim[idx] for idx, dim in zip(index, self._dim_offsets))

        # Calculate parameters
        result = self.__class__(
            offset=self._offset + sum(
                (i if isinstance(i, numbers.Integral) else i.start) * dim.step
                for i, dim in zip(index, self._dim_offsets)
            ),
            dim_offsets=tuple(
                idx for idx in index
                if not isinstance(idx, numbers.Integral)
            ),
        )

        return result

    @classmethod
    def mutually_broadcasted(cls, indexer1, indexer2):
        """Make two indexer objects broadcasted to the same shape."""
        if not isinstance(indexer1, cls) or not isinstance(indexer2, cls):
            raise TypeError

        dim_offsets1 = list(indexer1._dim_offsets)
        dim_offsets2 = list(indexer2._dim_offsets)

        ndim_diff = len(dim_offsets1) - len(dim_offsets1)
        if ndim_diff >= 0:
            dim_offsets1 = ndim_diff*[Range(1)] + dim_offsets1
        else:
            dim_offsets2 = (-ndim_diff)*[Range(1)] + dim_offsets2
        assert len(dim_offsets1) == len(dim_offsets2)

        for i in range(len(dim_offsets1))[::-1]:
            dim1 = dim_offsets1[i]
            dim2 = dim_offsets2[i]
            if dim1.length == dim2.length:
                continue
            if dim1.length == 1:
                dim_offsets1[i] = dim2
            elif dim2.length == 1:
                dim_offsets2[i] = dim1
            else:
                raise ValueError

        result1 = cls(
            offset=indexer1._offset,
            dim_offsets=dim_offsets1,
        )
        result2 = cls(
            offset=indexer2._offset,
            dim_offsets=dim_offsets2,
        )

        return result1, result2

    def broadcasted_to(self, new_shape):
        """Copy this indexer object broadcasted to the given shape."""
        if not isinstance(new_shape, tuple):
            raise TypeError

        dim_offsets_self = list(self._dim_offsets)

        ndim_diff = len(new_shape) - len(dim_offsets_self)
        if ndim_diff < 0:
            raise ValueError
        dim_offsets_self = ndim_diff*[Range(1)] + dim_offsets_self
        assert len(dim_offsets_self) == len(new_shape)

        for i in range(len(dim_offsets_self))[::-1]:
            dim_self, dim_other = dim_offsets_self[i].length, new_shape[i]
            if dim_self == dim_other:
                continue
            if dim_self != 1:
                raise ValueError

            dim_offsets_self[i] = Range(0, dim_other, 0)

        result = self.__class__(
            offset=self._offset,
            dim_offsets=dim_offsets_self,
        )

        return result
