from pynum.indexer import _nd_indices, _nd_getitem, _nd_shape, Indexer


class NDArray:
    """A pure-python version of the numpy ndarray."""

    __slots__ = ('_flat_array', '_indexer')

    def __init__(self, flat_array, indexer):
        """Construct an instance."""
        if indexer.min < 0 or indexer.max >= len(flat_array):
            raise ValueError

        self._flat_array = flat_array
        self._indexer = indexer

    @classmethod
    def from_values(cls, values, immutable=False):
        """Create a new ndarray from a nested iterable."""
        shape = _nd_shape(values)

        FlatType = (tuple if immutable else list)
        if shape:
            flat_array = FlatType(
                _nd_getitem(values, nd_index)
                for nd_index in _nd_indices(shape)
            )
        else:
            flat_array = FlatType([values])

        return cls(flat_array, indexer=Indexer.make_basic(shape=shape))

    @classmethod
    def as_array(cls, values):
        if isinstance(values, cls):
            return values
        return cls.from_values(values)

    def to_list(self):
        """Convert array into equivalent nested lists."""
        if not self.shape:
            return self._flat_array[self._indexer._offset]
        return [self.slice[i].to_list() for i in range(len(self))]

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.to_list())})'

    def __eq__(self, other):
        """Test for equality."""
        try:
            return (
                self.shape == _nd_shape(other)
            ) and all(s_item == o_item for s_item, o_item in zip(
                (self._flat_array[i] for i in self._indexer),
                (_nd_getitem(other, idx) for idx in _nd_indices(self.shape)),
            ))
        except TypeError:
            return False

    @property
    def slice(self):
        """Index/slice the data."""
        class Slice:
            """
            Alternative indexer for ndarrays.

            Keeps the result as an NDArray, even if it's a 0D-array.
            """

            def __getitem__(_, index):
                """Index/slice the data."""
                return NDArray(self._flat_array, self._indexer.sliced(index))

        return Slice()

    def __getitem__(self, index):
        """Index/slice the data."""
        result = NDArray(self._flat_array, self._indexer.sliced(index))
        if not result.shape:
            return result._flat_array[result._indexer._offset]
        return result

    def __setitem__(self, index, value):
        """Set values to an index/slice of the data."""
        new_indexer = self._indexer.sliced(index)
        value_array = self.__class__.as_array(value)
        value_array._indexer = value_array._indexer.broadcasted_to(
            new_indexer._shape
        )

        for i, j in zip(new_indexer, value_array._indexer):
            self._flat_array[i] = value_array._flat_array[j]

    def __len__(self):
        """Calculate number of subarrays in first dimension."""
        if not self.shape:
            raise TypeError
        return self.shape[0]

    @property
    def shape(self):
        """Get array dimensionality."""
        return self._indexer._shape
