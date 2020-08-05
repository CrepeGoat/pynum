import operator

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

    # -------------------------------------------------------------------------

    @classmethod
    def _applied_elementwise(cls, op, values1, values2):
        """Apply operations elementwise between two arrays."""
        array1 = cls.as_array(values1)
        array2 = cls.as_array(values2)

        idxr1, idxr2 = Indexer.mutually_broadcasted(
            array1._indexer, array2._indexer
        )

        return cls(
            flat_array=[
                op(array1._flat_array[i1], array2._flat_array[i2])
                for i1, i2 in zip(idxr1, idxr2)
            ],
            indexer=Indexer.make_basic(idxr1._shape),
        )

    def _modified_elementwise(self, op, values):
        """Apply assignment operations elementwise on this array."""
        value_array = self.__class__.as_array(values)
        value_array._indexer = value_array._indexer.broadcasted_to(
            self._indexer._shape
        )

        for i, j in zip(self._indexer, value_array._indexer):
            self._flat_array[i] = op(
                self._flat_array[i],
                value_array._flat_array[j]
            )

    def __lt__(self, rhs):
        return self._applied_elementwise(operator.lt, self, rhs)

    def __le__(self, rhs):
        return self._applied_elementwise(operator.le, self, rhs)

    def eq(self, rhs):
        return self._applied_elementwise(operator.eq, self, rhs)

    def __ne__(self, rhs):
        return self._applied_elementwise(operator.ne, self, rhs)

    def __ge__(self, rhs):
        return self._applied_elementwise(operator.ge, self, rhs)

    def __gt__(self, rhs):
        return self._applied_elementwise(operator.gt, self, rhs)

    def __add__(self, rhs):
        return self._applied_elementwise(operator.add, self, rhs)

    def __radd__(self, rhs):
        return self._applied_elementwise(operator.add, rhs, self)

    def __iadd__(self, rhs):
        return self._modified_elementwise(operator.iadd, rhs)

    def __sub__(self, rhs):
        return self._applied_elementwise(operator.sub, self, rhs)

    def __rsub__(self, rhs):
        return self._applied_elementwise(operator.sub, rhs, self)

    def __isub__(self, rhs):
        return self._modified_elementwise(operator.isub, rhs)

    def __mul__(self, rhs):
        return self._applied_elementwise(operator.mul, self, rhs)

    def __rmul__(self, rhs):
        return self._applied_elementwise(operator.mul, rhs, self)

    def __imul__(self, rhs):
        return self._modified_elementwise(operator.imul, rhs)

    def __truediv__(self, rhs):
        return self._applied_elementwise(operator.truediv, self, rhs)

    def __rtruediv__(self, rhs):
        return self._applied_elementwise(operator.truediv, rhs, self)

    def __itruediv__(self, rhs):
        return self._modified_elementwise(operator.itruediv, rhs)

    def __floordiv__(self, rhs):
        return self._applied_elementwise(operator.floordiv, self, rhs)

    def __rfloordiv__(self, rhs):
        return self._applied_elementwise(operator.floordiv, rhs, self)

    def __ifloordiv__(self, rhs):
        return self._modified_elementwise(operator.ifloordiv, rhs)

    def __mod__(self, rhs):
        return self._applied_elementwise(operator.mod, self, rhs)

    def __rmod__(self, rhs):
        return self._applied_elementwise(operator.mod, rhs, self)

    def __imod__(self, rhs):
        return self._modified_elementwise(operator.imod, rhs)

    def __pow__(self, rhs):
        return self._applied_elementwise(operator.pow, self, rhs)

    def __rpow__(self, rhs):
        return self._applied_elementwise(operator.pow, rhs, self)

    def __ipow__(self, rhs):
        return self._modified_elementwise(operator.ipow, rhs)

    def __lshift__(self, rhs):
        return self._applied_elementwise(operator.lshift, self, rhs)

    def __rlshift__(self, rhs):
        return self._applied_elementwise(operator.lshift, rhs, self)

    def __ilshift__(self, rhs):
        return self._modified_elementwise(operator.ilshift, rhs)

    def __rshift__(self, rhs):
        return self._applied_elementwise(operator.rshift, self, rhs)

    def __rrshift__(self, rhs):
        return self._applied_elementwise(operator.rshift, rhs, self)

    def __irshift__(self, rhs):
        return self._modified_elementwise(operator.irshift, rhs)

    def __and__(self, rhs):
        return self._applied_elementwise(operator.and_, self, rhs)

    def __rand__(self, rhs):
        return self._applied_elementwise(operator.and_, rhs, self)

    def __iand__(self, rhs):
        return self._modified_elementwise(operator.iand, rhs)

    def __xor__(self, rhs):
        return self._applied_elementwise(operator.xor, self, rhs)

    def __rxor__(self, rhs):
        return self._applied_elementwise(operator.xor, rhs, self)

    def __ixor__(self, rhs):
        return self._modified_elementwise(operator.ixor, rhs)

    def __or__(self, rhs):
        return self._applied_elementwise(operator.or_, self, rhs)

    def __ror__(self, rhs):
        return self._applied_elementwise(operator.or_, rhs, self)

    def __ior__(self, rhs):
        return self._modified_elementwise(operator.ior, rhs)
