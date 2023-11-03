import csv


def open_csv(file_path: str) -> list:
    """
    Reads a CSV file and returns its contents as a list of lists.

    :param file_path: The path to the CSV file.
    :type file_path: str
    :return: A list of lists where each inner list represents a row in the CSV.
    :rtype: list
    :raises TypeError: If the provided file_path is not a string.
    :raises FileNotFoundError: If no file exists at the given file_path.
    :raises csv.Error: If there's an error reading the CSV file.
    """

    if not isinstance(file_path, str):
        raise TypeError("The file_path must be a string.")

    try:
        with open(file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=",")
            csv_contents = list(reader)
            return csv_contents
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path {file_path} does not exist.")
    except csv.Error as e:
        raise csv.Error(f"An error occurred while reading the CSV file: {str(e)}")


def is_float(string: str) -> bool:
    """
    Checks if a given string can be converted to a float.

    :param string: The string to check.
    :type string: str
    :return: True if the string can be converted to a float, False otherwise.
    :rtype: bool
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def infer_column_dtypes(data: list) -> list:
    """
    Infers the data types of columns based on the first row of the data.

    :param data: A 2D list where each inner list represents a data row.
    :type data: list
    :return: A list of inferred data types for each column.
    :rtype: list
    :raises TypeError: If the provided data is not a list or if the first row is not a list.
    :raises ValueError: If the data is an empty list or if the data is not a 2D list.
    """

    if not isinstance(data, list):
        raise TypeError("Expected list got {}".format(type(data).__name__))

    if len(data) == 0:
        raise ValueError("data wrong shape needs to be 2d list")

    if not isinstance(data[0], list):
        raise ValueError("data wrong shape needs to be 2d list")

    # Default Types
    numeric_default_type = "float32"
    string_default_type = "U100"

    return [
        numeric_default_type if is_float(item) else string_default_type
        for item in data[0]
    ]
