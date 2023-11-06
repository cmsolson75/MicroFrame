import numpy as np
from typing import TypeVar, Generic, Type


class StructuredArrayIndexer:
    """
    A class for indexing into numpy structured arrays using both row and column indices.

    :param values: The numpy structured array to be indexed.
    :type values: numpy.ndarray
    :param columns: Column names corresponding to the data.
    :type columns: list
    """

    def __init__(self, values, columns):
        """
        Initializes the StructuredArrayIndexer with the structured array and its column names.
        """
        self.values = values
        self.columns = columns

    def __getitem__(self, idx):
        """
        Retrieve an item or a row from the structured array.

        :param idx: Either a single index to retrieve a row, or a tuple of row and column indices to retrieve an item.
        :type idx: int or tuple
        :return: The retrieved row or item.
        :rtype: numpy.void or data type of the column
        """
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
        else:
            row_idx = idx
            col_idx = None
        if col_idx is None:
            return self.values[row_idx]
        column_name = self.columns[col_idx]
        return self.values[row_idx][column_name]

    def __setitem__(self, idx, value):
        """
        Set a value in the structured array at the specified index.

        :param idx: A tuple of row and column indices to identify the location for assignment.
        :type idx: tuple
        :param value: The value to be set at the specified index.
        :type value: compatible with the column data type
        :raises ValueError: If only a single index is provided instead of a tuple.
        """
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
        else:
            raise ValueError("Both row and column indices are required for assignment")

        column_name = self.columns[col_idx]
        self.values[row_idx][column_name] = value


T = TypeVar('T')


class IlocIndexer(Generic[T], StructuredArrayIndexer):
    def __init__(self, values, columns, return_type: Type[T]):
        super().__init__(values, columns)  # Initialize the base class
        self.return_type = return_type

    def __getitem__(self, idx):
        """
        Retrieve a subset of the data as the specified return type.

        If a single row is retrieved, it's wrapped in a NumPy array to maintain
        the structured array format. If a single column is retrieved, it's converted
        into a structured array with the appropriate field name and dtype.

        :param idx: Index or indices to retrieve data.
        :type idx: int, tuple, or slice
        :return: A subset of the data as the specified return type.
        :rtype: T
        """
        subset = super().__getitem__(idx)
        if isinstance(subset, np.void):  # Single row
            subset = np.array([subset], dtype=subset.dtype)
        elif isinstance(subset, np.ndarray) and subset.ndim == 1:  # Single column
            if isinstance(idx, tuple) and isinstance(idx[1], (int, np.integer)):
                # Create a structured array with a single named field
                dtype = [(self.columns[idx[1]], subset.dtype)]
                subset = np.array([tuple([val]) for val in subset], dtype=dtype)
        return self.return_type(subset)
