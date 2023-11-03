from unittest.mock import patch, mock_open
import pytest
import csv

from file_readers.utils import csv_utils


def test_open_csv_file_expected_data():
    mock_data = "col1,col2,col3\nval1,val2,val3\n"
    m = mock_open(read_data=mock_data)
    with patch("builtins.open", m), patch("csv.reader",
                                          return_value=iter([["col1", "col2", "col3"], ["val1", "val2", "val3"]])):
        result = csv_utils.open_csv("any_path")
    m.assert_called_once_with("any_path", mode='r', newline='', encoding='utf-8')
    assert result == [["col1", "col2", "col3"], ["val1", "val2", "val3"]]


@pytest.mark.parametrize("input_data,expected_output", [
    ("", []),
    ("col1\nval1\n", [["col1"], ["val1"]]),
    ("col1,col2\nval1,", [["col1", "col2"], ["val1", ""]]),
])
def test_open_csv_file_edge_cases(input_data, expected_output):
    with patch("builtins.open", mock_open(read_data=input_data)), patch("csv.reader", return_value=iter(
            [line.split(",") for line in input_data.split("\n") if line])):
        result = csv_utils.open_csv("any_path")
    assert result == expected_output


@pytest.mark.parametrize("file_input, expected_exception", [
    (None, TypeError),
    (123, TypeError),
    ([], TypeError)
])
def test_open_csv_file_exceptions(file_input, expected_exception):
    with pytest.raises(expected_exception):
        csv_utils.open_csv(file_input)


def test_open_csv_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        csv_utils.open_csv("non_existent_file.csv")


def test_open_csv_csv_error():
    # Simulate a CSV error by having the mock raise an error when used
    m = mock_open()
    m.side_effect = csv.Error("Mock CSV error")
    with patch("builtins.open", m), pytest.raises(csv.Error) as excinfo:
        csv_utils.open_csv("any_path.csv")
    assert "Mock CSV error" in str(excinfo.value)


@pytest.mark.parametrize("input_data, expected_output", [
    ("1.4", True),
    ("Hi", False),
    ("2", True)
])
def test_is_float_expected(input_data, expected_output):
    assert csv_utils.is_float(input_data) == expected_output


@pytest.mark.parametrize("input_data, expected_output", [
    ([["1", "2.1", "String here"]], ["float32", "float32", "U100"])
])
def test_infer_column_dtypes_expected(input_data, expected_output):
    result = csv_utils.infer_column_dtypes(input_data)
    assert result == expected_output


@pytest.mark.parametrize("input_data, expected_exception", [
    (None, TypeError),
    (123, TypeError),
    ([], ValueError),
    (["1", "2.1", "String here"], ValueError)
])
def test_infer_column_dtypes_exceptions(input_data, expected_exception):
    with pytest.raises(expected_exception):
        csv_utils.infer_column_dtypes(input_data)
