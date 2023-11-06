import numpy as np
from typing import List, Any, Optional, Union, overload
from .printers import StructuredDataPrinter
from .manipulators import StructuredArrayManipulator
from .indexers import IlocIndexer


class MicroFrame:
    """
    A lightweight and efficient data structure for handling tabular data.

    The MicroFrame class provides a simple yet powerful interface for manipulating and displaying structured data. It
    is designed to be intuitive for users familiar with pandas DataFrame but aims to be more memory-efficient and
    faster for smaller datasets.

    Parameters
    ----------
    data : List[List[Any]]
        A list of lists representing the rows of data.
    dtypes : List[str]
        A list of data types for each column.
    columns : Optional[List[str]], optional
        A list of column names. If None, default column names will be generated.

    Raises
    ------
    TypeError
        If the provided `data`, `dtypes`, or `columns` are not lists, or if the `data` is not a list of lists.
    ValueError
        If there's a mismatch between the data rows and the provided columns or dtypes.

    Attributes
    ----------
    columns : np.ndarray
        An array of column names.
    values : np.ndarray
        A structured numpy array representing the data.

    Methods
    -------
    head(max_width=80, num_cols=None, num_rows=5)
        Prints the first `num_rows` rows of the MicroFrame.
    tail(max_width=80, num_cols=None, num_rows=5)
        Prints the last `num_rows` rows of the MicroFrame.
    rename(new_columns)
        Renames the columns of the MicroFrame.
    change_dtypes(dtypes_dict)
        Changes the data types of specified columns.
    """

    @overload
    def __init__(
            self,
            data: List[List[Any]],
            dtypes: List[str],
            columns: Optional[List[str]] = None,
    ):
        ...

    @overload
    def __init__(
            self,
            data: np.ndarray,
            columns: Optional[List[str]] = None,
    ):
        ...

    def __init__(
            self,
            data: Union[List[List[Any]], np.ndarray],
            dtypes: Optional[List[str]] = None,
            columns: Optional[List[str]] = None,
    ):
        if isinstance(data, list):
            # Original initialization with list of lists
            if not all(isinstance(row, list) for row in data):
                raise TypeError("Data must be a list of lists.")
            if dtypes is None or not isinstance(dtypes, list):
                raise TypeError("Dtypes must be a list.")

            self.columns = self._initialize_columns(data, columns)
            self.values = self._initialize_values(data, dtypes, self.columns)
        elif isinstance(data, np.ndarray):
            # Initialization with NumPy structured array
            if not data.dtype.names:
                raise TypeError("Data must be a structured numpy array with named fields.")
            if columns is not None and not isinstance(columns, list):
                raise TypeError("Columns must be a list.")

            self.columns = self._initialize_columns_from_structured_array(data, columns)
            self.values = data
        else:
            raise TypeError("Data must be a list of lists or a structured numpy array.")

    @staticmethod
    def _initialize_columns_from_structured_array(
            data: np.ndarray, columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Initialize the columns for the MicroFrame based on the provided structured array and columns.

        :param data: A structured numpy array representing the data rows.
        :param columns: A list of column names or None. If None, existing column names from structured array will be used.
        :return: A numpy array of initialized column names.
        :raises ValueError: If there's a mismatch between the data fields and the provided columns.
        """
        existing_columns = list(data.dtype.names)

        if columns is None:
            return np.array(existing_columns)
        else:
            if not all(isinstance(col, str) for col in columns):
                raise ValueError("All column names must be of type str.")
            if len(columns) != len(existing_columns):
                raise ValueError("Columns and the data fields must be of equal length.")
            return np.array(columns)

    @staticmethod
    def _initialize_columns(
            data: List[List[Any]], columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Initialize the columns for the MicroFrame based on the provided data and columns.

        :param data: A list of lists representing the data rows.
        :param columns: A list of column names or None. If None, default column names will be generated.
        :return: A numpy array of initialized column names.
        :raises ValueError: If there's a mismatch between the data rows and the provided columns.
        """
        if not data:
            raise ValueError("Data cannot be empty.")

        num_columns = len(data[0])
        if columns is None:
            columns = [f"Unnamed: {i}" for i in range(num_columns)]
        else:
            if not all(isinstance(col, str) for col in columns):
                raise ValueError("All column names must be of type str.")

            if len(columns) != num_columns:
                if len(columns) == num_columns + 1 and columns[-1] == "":
                    columns = columns[:-1]
                else:
                    raise ValueError(
                        "Columns and the data rows must be of equal length."
                    )

            columns = [
                f"Unnamed: {i}" if not name else name for i, name in enumerate(columns)
            ]

        return np.array(columns)

    @staticmethod
    def _initialize_values(
            data: List[List[Any]], dtypes: List[str], columns: np.ndarray
    ) -> np.ndarray:
        """
        Initialize the values for the MicroFrame based on the provided data, dtypes, and columns.

        :param data: A list of lists representing the data rows.
        :param dtypes: A list of data types corresponding to each column.
        :param columns: A list of column names.
        :return: A structured numpy array representing the initialized data.
        :raises ValueError: If there's a mismatch between the data and the provided dtypes or columns.
        """
        if not data or not dtypes or columns.size == 0:
            raise ValueError("Data, dtypes, and columns cannot be empty.")

        if len(data[0]) != len(dtypes) or len(dtypes) != len(columns):
            raise ValueError("The length of data rows, dtypes, and columns must match.")

        try:
            np_dtypes = np.dtype(list(zip(columns, dtypes)))
            return np.array([tuple(row) for row in data], dtype=np_dtypes)
        except TypeError as e:
            raise ValueError(f"Invalid dtypes provided: {e}")

    def __getitem__(self, column_header):
        """
        Allows column access using square bracket notation.

        This magic method is a part of Python's container protocol. It allows users
        to access columns of the MicroFrame as if it were a dictionary, using the
        column names as keys.

        :param column_header: The header (name) of the column to be accessed.
        :type column_header: str
        :return: The column data.
        :rtype: numpy.ndarray
        """
        return self.values[column_header]

    def __len__(self):
        """
        Returns the number of rows in the MicroFrame.

        This magic method allows the use of the built-in len() function on the
        MicroFrame object, conforming to the Python protocol for sequences.

        :return: The number of rows in the MicroFrame.
        :rtype: int
        """
        return len(self.values)

    def __repr__(self):
        """
        Provides a string representation of the MicroFrame for the console.

        This magic method controls how the MicroFrame is displayed when it is
        returned in the Python interpreter or printed to the console. By default,
        it displays the top rows of the MicroFrame, similar to the .head() method.

        :return: An empty string, as the actual printing is handled by the .head() method.
        :rtype: str
        """
        self.head()
        return ""

    def head(self, max_width: int = 80, num_cols: int = None, num_rows: int = 5):
        """
        Displays the first few rows of the MicroFrame.

        This method uses the StructuredDataPrinter class to provide a tabular representation
        of the first few rows of the MicroFrame, similar to the `.head()` method in pandas.

        :param max_width: Maximum width of the printed table in characters.
        :param num_cols: Number of columns to display. If None, all columns are displayed.
        :param num_rows: Number of rows to display.
        """
        printer = StructuredDataPrinter(self.values, self.columns)
        printer.structured_print(max_width, num_cols, num_rows)

    def tail(self, max_width=80, num_cols=None, num_rows=5):
        """
        Displays the last few rows of the MicroFrame.

        This method uses the StructuredDataPrinter class to provide a tabular representation
        of the last few rows of the MicroFrame, similar to the `.tail()` method in pandas.

        :param max_width: Maximum width of the printed table in characters.
        :param num_cols: Number of columns to display. If None, all columns are displayed.
        :param num_rows: Number of rows to display.
        """
        printer = StructuredDataPrinter(self.values, self.columns)
        printer.structured_print(max_width, num_cols, num_rows, tail=True)

    def rename(self, new_columns):
        """
        Renames the columns of the MicroFrame.

        This method uses the StructuredArrayManipulator class to rename the columns of
        the MicroFrame based on the provided mapping.

        :param new_columns: A dictionary mapping old column names to new column names.
        """
        manipulator = StructuredArrayManipulator(self.values, self.columns)
        manipulator.rename(new_columns)
        self.columns = manipulator.columns
        self.values = manipulator.values

    def change_dtypes(self, dtypes_dict: dict):
        """
        Changes the data types of the columns of the MicroFrame.

        This method uses the StructuredArrayManipulator class to change the data types
        of the columns of the MicroFrame based on the provided mapping.

        :param dtypes_dict: A dictionary mapping column names to their new data types.
        """
        manipulator = StructuredArrayManipulator(self.values, self.columns)
        manipulator.change_dtypes(dtypes_dict)
        self.values = manipulator.values

    @property
    def types(self):
        """
        Returns the data types of the columns in the MicroFrame.

        This property provides a convenient way to access the data types (dtypes)
        of the underlying structured numpy array. Each column's data type is returned
        in a numpy dtype object.

        :return: The data types of the columns.
        :rtype: numpy.dtype
        """
        return self.values.dtype

    @property
    def count(self):
        """
        Returns the number of rows in the MicroFrame.

        This property is useful for quickly determining the size of the dataset
        without having to inspect the underlying numpy array directly.

        :return: The number of rows in the MicroFrame.
        :rtype: int
        """
        return self.values.shape[0]

    @property
    def shape(self):
        """
        Returns the shape of the MicroFrame as a tuple.

        This property mimics the `.shape` attribute of numpy arrays and pandas
        DataFrames, providing a familiar interface for users. The shape is returned
        as a tuple where the first element is the number of rows and the second
        element is the number of columns.

        :return: The shape of the MicroFrame.
        :rtype: tuple
        """
        return self.values.shape[0], self.columns.shape[0]

    @property
    def iloc(self):
        """
        Provides integer-location based indexing for selection by position.

        This property returns an instance of _MicroFrameIndexer, a specialized indexer that
        extends StructuredArrayIndexer. It allows for indexing into the structured array using
        integer positions. When accessed, it ensures that the subset of data is returned as a
        MicroFrame instance, maintaining the structure and functionalities of the original MicroFrame.
        It is designed to mimic the `.iloc` property in pandas, providing a familiar interface for users.

        :return: A MicroFrame instance representing a subset of the original data based on integer-location indexing.
        :rtype: MicroFrame
        """
        return IlocIndexer(self.values, self.columns, MicroFrame)
