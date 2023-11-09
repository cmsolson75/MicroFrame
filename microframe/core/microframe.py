import numpy as np
from typing import List, Any, Optional
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



    Examples
    --------

    >>> # Simple Initialization
    >>> data = [[1, 'Alice'], [2, 'Bob']] # Initialize Data
    >>> dtypes = ['int32', 'U10'] # Initialize Data Types
    >>> columns = ['id', 'name'] # Initialize Column Names
    >>> mframe = MicroFrame(data, dtypes, columns)
    >>> mframe.head() # Display first 5 rows
    id  name
    ---------
    1   Alice
    2   Bob
    2 rows x 2 columns

    >>> # How to extract a subsection of the data and convert it to
    >>> # numpy for training
    >>> mframe_slice = mframe.iloc[:, 0] # returns all rows, but just col 0
    >>> numpy_array = mframe_slice.to_numpy() # returns mframe_slice as a numpy array
    >>> numpy_array
    array([[1],
           [2]], dtype=int32)



    Methods
    -------
    """

    def __init__(self, data: List[List[Any]], dtypes: List[str], columns: Optional[List[str]] = None):
        """
        Initializes a new instance of the MicroFrame class with a list of lists.

        :param data: Data to initialize the MicroFrame. Each inner list represents a row.
        :param dtypes: A list of data types for each column.
        :param columns: A list of column names. If None, default column names will be generated.
        """
        if not all(isinstance(row, list) for row in data):
            raise TypeError("Data must be a list of lists.")
        if dtypes is None or not isinstance(dtypes, list):
            raise TypeError("Dtypes must be a list.")

        self.columns = self._initialize_columns(data, columns)
        self.values = self._initialize_values(data, dtypes, self.columns)

    @classmethod
    def from_structured_array(cls, data: np.ndarray, columns: Optional[List[str]] = None):
        """
        Factory method to create a MicroFrame instance from a structured NumPy array.

        :param data: A structured NumPy array with named fields.
        :param columns: A list of column names. If None, default column names will be generated.
        :return: An instance of MicroFrame.
        """
        if not data.dtype.names:
            raise TypeError("Data must be a structured numpy array with named fields.")
        if columns is not None and not isinstance(columns, list):
            raise TypeError("Columns must be a list.")

        instance = cls.__new__(cls)
        instance.columns = cls._initialize_columns_from_structured_array(data, columns)
        instance.values = data
        return instance

    @staticmethod
    def _initialize_columns_from_structured_array(
            data: np.ndarray, columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Initialize the columns for the MicroFrame based on the provided structured array and columns.

        :param data: A structured numpy array representing the data rows.
        :param columns: A list of column names or None. If None, existing column names from
        structured array will be used.
        :return: A numpy array of initialized column names.
        :raises ValueError: If there's a mismatch between the data fields and the provided
        columns.
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

        Example::

        >>> mframe.head()  # Show first 5 rows
        >>> mframe.head(num_rows=10)  # Show first 10 rows

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

        Example::

            >>> mframe.tail()
            >>> mframe.tail(num_rows=10)

        """
        printer = StructuredDataPrinter(self.values, self.columns)
        printer.structured_print(max_width, num_cols, num_rows, tail=True)

    def rename(self, new_columns):
        """
        Renames the columns of the MicroFrame.

        This method uses the StructuredArrayManipulator class to rename the columns of
        the MicroFrame based on the provided mapping.

        :param new_columns: A dictionary mapping old column names to new column names.

        Example::

            >>> mframe.rename({'old_name1': 'new_name1', 'old_name2': 'new_name2'})
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

        Example::

            >>> mframe.change_dtypes({'column1': 'float64', 'column2': 'int32'})

        """
        manipulator = StructuredArrayManipulator(self.values, self.columns)
        manipulator.change_dtypes(dtypes_dict)
        self.values = manipulator.values

    def to_numpy(self):
        """
        Converts the MicroFrame to a regular 2D NumPy array (matrix).

        This conversion will result in a 2D NumPy array with each column corresponding to a field in the MicroFrame.
        All fields must be of a type that can be cast to a common dtype.

        :return: A 2D NumPy array representation of the MicroFrame data.
        :rtype: numpy.ndarray

        Example::

            >>> numpy_array = mframe.to_numpy()

        """
        manipulator = StructuredArrayManipulator(self.values, self.columns)
        return manipulator.to_numpy()

    def describe(self):
        """
        Generates descriptive statistics summarizing the central tendency,
        dispersion, and shape of the dataset's distribution, excluding NaN values.

        This method targets numeric data and provides an overview of statistical
        characteristics of numeric columns, including count, mean, standard deviation,
        minimum, and maximum values.

        NaN values are excluded from the calculations. The results are printed in a
        tabular format to the console.

        **Statistics computed:**

        - *count*: The number of non-NaN values.
        - *mean*: The mean of the values.
        - *std*: The sample standard deviation of the values.
        - *min*: The minimum value.
        - *max*: The maximum value.

        The method prints the summary to the console and does not return a value.

        :raises TypeError: If columns contain types that cannot be converted to float.
        :raises ValueError: If computations encounter issues like an empty column.

        Example::

            >>> mframe.describe()

        """
        # Identify numeric columns and their data types
        numeric_columns = [name for (name, dtype) in self.values.dtype.fields.items() if
                           np.issubdtype(dtype[0], np.number)]

        # Initialize statistics dictionary
        stats = {
            "count": [],
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
        }
        for key in stats.keys():
            stats[key].append(key)

        # Compute statistics for each numeric column, excluding NaN values
        for col in numeric_columns:
            column_data = self.values[col].astype(float)  # Convert to float for calculations
            valid_data = column_data[~np.isnan(column_data)]  # Exclude NaN values
            stats["count"].append(np.count_nonzero(~np.isnan(column_data)))
            stats["mean"].append(np.round(np.nanmean(column_data), 3))
            stats["std"].append(
                np.round(np.nanstd(column_data, ddof=1), 3))  # Sample standard deviation excluding NaN
            stats["min"].append(np.round(np.nanmin(column_data), 3))
            stats["max"].append(np.round(np.nanmax(column_data), 3))

        # Prepare the data for printing
        data = [value for value in stats.values()]
        headers = ['stats'] + numeric_columns

        # Print the statistics using StructuredDataPrinter
        summary_printer = StructuredDataPrinter(data, columns=headers, max_value_length=20)
        summary_printer.structured_print(max_width=80, num_cols=None, num_rows=10)

    @property
    def dtypes(self):
        """
        Returns the data types of the columns in the MicroFrame.

        This property provides a convenient way to access the data types (dtypes)
        of the underlying structured numpy array. Each column's data type is returned
        in a numpy dtype object.

        :return: The data types of the columns.
        :rtype: numpy.dtype

        Example::

            >>> mframe.dtypes

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

        Example::

            >>> mframe.count

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

        Example::

            >>> mframe.shape

        """
        return self.values.shape[0], self.columns.shape[0]

    @property
    def iloc(self):
        """
        Provides integer-location based indexing for selection by position.

        This property returns an instance of IlocIndexer, which is specialized for integer-location
        based indexing. It enables the selection of subsets of the MicroFrame's data by integer
        position, similar to the `.iloc` property in pandas DataFrames.

        When accessed, this property ensures that the subset of data is returned as a MicroFrame
        instance, maintaining the structure and functionalities of the original MicroFrame.

        :return: An instance of IlocIndexer for integer-location based indexing.
        :rtype: IlocIndexer

        Example::

            >>> first_row = mframe.iloc[0]  # First row of the MicroFrame
            >>> last_row = mframe.iloc[-1] # Last row of the MicroFrame
        """
        return IlocIndexer(self.values, self.columns, MicroFrame)
