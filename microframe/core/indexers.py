import numpy as np
from typing import TypeVar, Generic, Type, Union, Any, List


class StructuredArrayIndexer:
    """
    A class for indexing into numpy structured arrays using both row and column indices.

    :param values: The numpy structured array to be indexed.
    :type values: numpy.ndarray
    :param columns: Column names corresponding to the data.
    :type columns: numpy.ndarray
    """

    def __init__(self, values: np.ndarray, columns: np.ndarray):
        """
        Initializes the StructuredArrayIndexer with the structured array and its column names.
        """
        self.values = values
        self.columns = columns

    def __getitem__(self, idx: Union[int, tuple]) -> Any:
        """
        Retrieve an item or a row from the structured array.

        The idx parameter supports multiple indexing modes:
            - Single integer for a row: `data[5]`
            - Tuple with row and column index: `data[5, 3]`
            - Tuple with row index and column slice: `data[5, 1:4]`
            - Slice for multiple rows: `data[1:5]`

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

    def __setitem__(self, idx: tuple, value: Any) -> None:
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
    """
    Provides integer-location based indexing for selection by position.

    This indexer is a generic class that returns a subset of the data in a
    structured array format. It inherits from `StructuredArrayIndexer` and allows
    for selection of data by integer-location, similar to `.iloc` in pandas.

    Parameters
    ----------
    values : numpy.ndarray
        The numpy structured array to be indexed.
    columns : numpy.ndarray
        Column names corresponding to the data in the structured array.
    return_type : Type[T]
        The type of the object that will be returned by the indexer. Typically, this will
        be a `MicroFrame` or similar class that can be initialized from a structured array.
    """

    def __init__(self, values: np.ndarray, columns: np.ndarray, return_type: Type[T]):
        """
        Initializes the indexer with structured array values, column names, and the return type.
        """
        super().__init__(values, columns)  # Initialize the base class
        self.return_type = return_type

    def __getitem__(self, idx: Union[int, tuple]) -> Type[T]:
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
        return self.return_type.from_structured_array(subset)
