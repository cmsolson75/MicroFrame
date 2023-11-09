from .utils.csv_utils import open_csv, infer_column_dtypes
from ..core.microframe import MicroFrame


def read_csv(file_path: str) -> MicroFrame:
    """
    Reads a CSV file and constructs a `MicroFrame` object from it.

    The function reads the CSV file specified by `file_path`, infers the data types of its columns, and returns a
    `MicroFrame` object containing the data and inferred data types.

    :param file_path: The path to the CSV file to be read.
    :type file_path: str
    :return: A `MicroFrame` object containing the data from the CSV file.
    :rtype: MicroFrame
    :raises FileNotFoundError: If the specified file does not exist.
    :raises csv.Error: If an error occurs during CSV reading.
    :raises TypeError: If the contents of the CSV file are not in the expected format.
    :raises ValueError: If the CSV file is empty or the data types cannot be inferred.

    Example:
        >>> from microframe.readers.readers import read_csv
        >>> microframe = read_csv('path/to/your.csv')
        >>> print(microframe)
    """
    csv_content = open_csv(file_path)
    if not csv_content or not csv_content[0]:
        raise ValueError("The CSV file is empty or does not contain headers.")

    columns = csv_content[0]
    data = csv_content[1:]

    if not data:
        raise ValueError("The CSV file does not contain data rows.")

    dtypes = infer_column_dtypes(data)
    return MicroFrame(data, dtypes, columns)
