import pytest
import numpy as np
from unittest.mock import mock_open, patch
from microframe.readers.readers import read_csv
from microframe.core.microframe import MicroFrame


def test_read_valid_csv():
    mock_data = "col1,col2,col3\nval1,val2,val3\n"
    m = mock_open(read_data=mock_data)
    with patch("builtins.open", m), patch(
        "csv.reader",
        return_value=iter([["col1", "col2", "col3"], ["val1", "val2", "val3"]]),
    ):
        microframe = read_csv("any_path")
        assert isinstance(microframe, MicroFrame)

        assert len(microframe) > 0
    m.assert_called_once_with("any_path", mode="r", newline="", encoding="utf-8")


def test_invalid_csv_format(tmpdir):
    file_path = tmpdir.join("invalid.csv")
    file_path.write("Column1|Column2\n1|Hello,2|World")
    with pytest.raises(ValueError):
        read_csv(str(file_path))


def test_empty_csv(tmpdir):
    file_path = tmpdir.join("empty.csv")
    file_path.write("")
    with pytest.raises(ValueError):  # Ensure read_csv checks for empty content
        read_csv(str(file_path))


def test_infer_data_types(tmpdir):
    file_path = tmpdir.join("datatypes.csv")
    file_path.write("Integers,Floats,Strings\n1,3.14,Hello\n2,2.71,World")
    microframe = read_csv(str(file_path))
    assert microframe.types["Floats"] == np.float32
    assert microframe.types["Strings"] == "U100"


def test_incorrect_csv_content_format(tmpdir):
    file_path = tmpdir.join("incorrect_format.csv")
    file_path.write("This is not a CSV format")
    with pytest.raises(ValueError):  # Change to expect ValueError
        read_csv(str(file_path))
