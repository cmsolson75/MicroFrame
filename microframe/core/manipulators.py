import numpy as np


class ArrayManipulationError(Exception):
    """Raised when there's an error during Structured Array manipulation"""

    pass


class StructuredArrayManipulator:
    """
    A class for manipulating numpy structured arrays.

    :param values: The numpy structured array to be manipulated.
    :type values: numpy.ndarray
    :param columns: Column names corresponding to the data.
    :type columns: numpy.ndarray
    """

    def __init__(self, values: np.ndarray, columns: np.ndarray):
        """
        Initializes the StructuredArrayManipulator with the structured array and its column names.
        """
        self.values = values
        self.columns = columns

    def rename(self, new_columns: dict) -> None:
        """
        Renames columns in the structured array based on a provided mapping.

        :param new_columns: A dictionary mapping old column names to new column names.
        :type new_columns: dict
        :raises ArrayManipulationError: If the old column name doesn't exist or the new column name already exists.
        """
        for old_name, new_name in new_columns.items():
            if old_name not in self.columns:
                raise ArrayManipulationError(
                    f"Column '{old_name}' does not exist and cannot be renamed."
                )
            if new_name in self.columns and new_name != old_name:
                raise ArrayManipulationError(
                    f"Column '{new_name}' already exists. Duplicate names are not allowed."
                )

        new_dtypes = [
            (new_columns.get(old_name, old_name), self.values.dtype[old_name])
            for old_name in self.columns
        ]
        self.values = self.values.astype(new_dtypes)

        # Create a new columns array with the updated names
        new_columns_array = np.array(
            [new_columns.get(old_name, old_name) for old_name in self.columns]
        )
        self.columns = new_columns_array

    def change_dtypes(self, dtypes_dict: dict) -> None:
        """
        Changes the data types of specified columns in the structured array.

        :param dtypes_dict: A dictionary mapping column names to their new data types.
        :type dtypes_dict: dict
        :raises ArrayManipulationError: If the column doesn't exist or the type conversion is invalid.
        """
        try:
            for column_name, data_type in dtypes_dict.items():
                if column_name not in self.columns:
                    raise ArrayManipulationError(
                        f"Column '{column_name}' does not exist and cannot have its data "
                        f"type changed."
                    )

            # Create a list of tuples for new dtypes
            new_dtypes = [
                (name, dtypes_dict.get(name, self.values.dtype.fields[name][0]))
                for name in self.values.dtype.names
            ]
            new_values = np.zeros(self.values.shape, dtype=new_dtypes)

            for name in self.values.dtype.names:
                dtypes = dtypes_dict.get(name, self.values.dtype.fields[name][0])
                new_values[name] = self.values[name].astype(dtypes)

            self.values = new_values
        except ValueError as e:
            raise ArrayManipulationError(f"TypeError: {e}")

    def to_numpy(self):
        """
        Converts the structured array to a regular 2D NumPy array (matrix).

        This conversion will result in a 2D NumPy array with each column corresponding to a field in the structured array.
        All fields must be of a type that can be cast to a common dtype.

        :return: A 2D NumPy array representation of the structured array.
        :rtype: numpy.ndarray
        :raises ArrayManipulationError: If the conversion is not possible due to incompatible data types.
        """
        try:
            # Extract each column and stack them horizontally to form a 2D array
            columns = [self.values[field] for field in self.values.dtype.names]
            return np.column_stack(columns)
        except ValueError as e:
            raise ArrayManipulationError(f"Error in converting to a regular 2D NumPy array: {e}")