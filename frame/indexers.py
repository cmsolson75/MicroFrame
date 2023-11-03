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
