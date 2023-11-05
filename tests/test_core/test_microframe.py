import pytest
import numpy as np
from microframe.core.microframe import MicroFrame
from microframe.core.indexers import StructuredArrayIndexer


@pytest.fixture
def default_microframe():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    dtypes = ["float32", "U100"]
    columns = ["num", "char"]
    return MicroFrame(data, dtypes, columns)


def test_microframe_initialization(default_microframe):
    """
    Test initialization of microframe class
    """
    microframe = default_microframe
    expected_values = np.array(
        [(1, "a"), (2, "b"), (3, "c")], dtype=[("num", "<f4"), ("char", "<U100")]
    )

    assert isinstance(microframe, MicroFrame)

    assert all(microframe.columns == ["num", "char"])
    assert np.array_equal(microframe.values, expected_values)
    assert np.array_equal(microframe.values.dtype, expected_values.dtype)

    assert isinstance(microframe.columns, np.ndarray)
    assert isinstance(microframe.values, np.ndarray)

    # Test types property
    assert np.array_equal(
        microframe.types, np.dtype([("num", "<f4"), ("char", "<U100")])
    )

    # Test shape property
    assert microframe.shape == (3, 2)


def test_microframe_initialization_with_structured_array():
    """
    Test initialization of MicroFrame class with a NumPy structured array
    """
    structured_data = np.array(
        [(1, 'Alice', 35.5), (2, 'Bob', 45.0), (3, 'Charlie', 25.5)],
        dtype=[('ID', 'i4'), ('Name', 'U10'), ('Age', 'f4')]
    )

    microframe = MicroFrame(data=structured_data)

    assert isinstance(microframe, MicroFrame)
    assert all(microframe.columns == ['ID', 'Name', 'Age'])
    assert np.array_equal(microframe.values, structured_data)
    assert np.array_equal(microframe.values.dtype, structured_data.dtype)


@pytest.mark.parametrize(
    "data, dtypes, columns, exception",
    [
        (1, ["float32", "U100"], ["num", "char"], TypeError),
        ("hi", ["float32", "U100"], ["num", "char"], TypeError),
        ([["1", "a"], ["2", "b"], ["3", "c"]], 1, ["num", "char"], TypeError),
        ([["1", "a"], ["2", "b"], ["3", "c"]], ["float32", "U100"], 1, TypeError),
        ([], ["float32", "U100"], ["num", "char"], ValueError),
        ([["1", "a"], ["2", "b"], ["3", "c"]], [], ["num", "char"], ValueError),
        ([["1", "a"], ["2", "b"], ["3", "c"]], ["float32", "U100"], [], ValueError),
    ],
)
def test_microframe_initialization_exceptions(data, dtypes, columns, exception):
    with pytest.raises(exception):
        MicroFrame(data, dtypes, columns)


def test_initialize_columns_with_columns_specified():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    columns = ["num", "char"]

    result = MicroFrame._initialize_columns(data, columns)

    assert np.array_equal(result, np.array(columns))


def test_initialize_columns_with_no_columns_specified():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    columns = None

    result = MicroFrame._initialize_columns(data, columns)

    expected_columns = np.array(["Unnamed: 0", "Unnamed: 1"])
    assert np.array_equal(result, expected_columns)


def test_initialize_columns_with_columns_specified_and_trailing_empty_column():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    columns = ["num", "char", ""]

    result = MicroFrame._initialize_columns(data, columns)

    expected_columns = np.array(["num", "char"])
    assert np.array_equal(result, expected_columns)


def test_initialize_columns_with_mismatched_data_and_columns():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    columns = ["num", "char", "extra"]

    with pytest.raises(ValueError):
        MicroFrame._initialize_columns(data, columns)


def test_initialize_columns_with_wrong_data_type_in_columns():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    columns = ["num", "char", 1]

    with pytest.raises(ValueError):
        MicroFrame._initialize_columns(data, columns)


def test_initialize_values_with_valid_data_and_dtypes():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    dtypes = ["int32", "U1"]
    columns = np.array(["num", "char"])

    result = MicroFrame._initialize_values(data, dtypes, columns)

    expected_values = np.array(
        [(1, "a"), (2, "b"), (3, "c")], dtype=[("num", "i4"), ("char", "U1")]
    )
    assert np.array_equal(result, expected_values)


def test_initialize_values_with_invalid_dtype():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    dtypes = ["int32", "this_is_not_a_valid_dtype"]
    columns = np.array(["num", "char"])

    with pytest.raises(ValueError) as excinfo:
        MicroFrame._initialize_values(data, dtypes, columns)
    assert "Invalid dtypes provided" in str(excinfo.value)


def test_initialize_values_with_mismatched_data_and_dtypes():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    dtypes = ["int32", "U1", "float64"]
    columns = np.array(["num", "char"])

    with pytest.raises(ValueError):
        MicroFrame._initialize_values(data, dtypes, columns)


def test_initialize_values_with_mismatched_data_and_columns():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    dtypes = ["int32", "U1"]
    columns = np.array(["num", "char", "extra"])

    with pytest.raises(ValueError):
        MicroFrame._initialize_values(data, dtypes, columns)


def test_properties_microframe(default_microframe):
    mf = default_microframe
    # Assertions
    assert mf.types == mf.values.dtype
    assert mf.count == len(mf.values)
    assert mf.shape == (len(mf.values), len(mf.columns))


@pytest.mark.parametrize(
    "rename_dict, expected_columns",
    [
        ({"num": "number", "char": "character"}, ["number", "character"]),
        ({"char": "symbol"}, ["num", "symbol"]),
    ],
)
def test_rename_columns_microframe(default_microframe, rename_dict, expected_columns):
    # Rename columns
    microframe = default_microframe
    microframe.rename(rename_dict)

    # Assertions
    assert list(microframe.columns) == expected_columns
    assert all(
        new_name in microframe.values.dtype.names for new_name in expected_columns
    )


@pytest.mark.parametrize(
    "new_dtypes, expected_dtypes",
    [
        (
                {"num": "f8", "char": "U10"},
                {"num": np.dtype("f8"), "char": np.dtype("U10")},
        ),
        ({"char": "U5"}, {"num": np.dtype("f4"), "char": np.dtype("U5")}),
    ],
)
def test_change_dtypes_microframe(default_microframe, new_dtypes, expected_dtypes):
    # Change dtypes
    microframe = default_microframe
    microframe.change_dtypes(new_dtypes)

    # Assertions
    for column, expected_dtype in expected_dtypes.items():
        assert microframe.values.dtype[column] == expected_dtype


def test_iloc_microframe(default_microframe):
    mf = default_microframe

    # Assertions
    # Check if iloc[0] returns a MicroFrame instance
    assert isinstance(mf.iloc[0], MicroFrame)

    # Check the first row values
    first_row = mf.iloc[0]
    assert first_row.values[0][0] == 1.0
    assert first_row.values[0][1] == "a"

    # Check slicing for multiple rows
    sliced_mf = mf.iloc[:2]  # Get first two rows
    assert isinstance(sliced_mf, MicroFrame)
    assert sliced_mf.shape == (2, 2)  # Shape of the sliced MicroFrame
    assert np.array_equal(sliced_mf.values['num'], [1.0, 2.0])
    assert np.array_equal(sliced_mf.values['char'], ['a', 'b'])

    # Check setting a value
    mf.iloc[2, 1] = "Test"
    assert mf.values[2][1] == "Test"


def test_repr_microframe(default_microframe, capsys):
    mf = default_microframe

    # Invoke __repr__
    repr(mf)

    # Capture the output and verify it
    captured = capsys.readouterr()
    assert "num" in captured.out
    assert "char" in captured.out
    assert "a" in captured.out
    assert "b" in captured.out


def test_head_microframe(default_microframe, capsys):
    mf = default_microframe
    # Setup MicroFrame instance

    # Invoke tail
    mf.head(num_rows=2)
    captured = capsys.readouterr()
    assert "a" in captured.out
    assert "b" in captured.out


def test_tail_microframe(default_microframe, capsys):
    mf = default_microframe
    # Setup MicroFrame instance

    # Invoke tail
    mf.tail(num_rows=2)

    # Capture the output and verify it
    captured = capsys.readouterr()
    assert "b" in captured.out
    assert "c" in captured.out


def test_getitem_microframe(default_microframe):
    mf = default_microframe

    assert list(mf["num"]) == [1.0, 2.0, 3.0]
