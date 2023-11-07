import pytest
import numpy as np
from microframe.core.manipulators import StructuredArrayManipulator, ArrayManipulationError


@pytest.fixture
def default_manipulator():
    values = np.array(
        [(1, "a"), (2, "b"), (3, "c")], dtype=[("num", "i4"), ("char", "U1")]
    )
    columns = np.array(["num", "char"])
    manipulator = StructuredArrayManipulator(values, columns)
    return manipulator


def test_structured_array_manipulator_initialization():
    local_values = np.array(
        [(1, "a"), (2, "b"), (3, "c")], dtype=[("num", "i4"), ("char", "U1")]
    )
    local_columns = np.array(["num", "char"])

    local_manipulator = StructuredArrayManipulator(local_values, local_columns)
    assert isinstance(local_manipulator, StructuredArrayManipulator)
    assert local_manipulator.values is local_values
    assert local_manipulator.columns is local_columns


@pytest.mark.parametrize(
    "rename_dict, expected_columns",
    [
        ({"num": "Num"}, ["Num", "char"]),
        ({"num": "Number"}, ["Number", "char"]),
        ({"num": "Number", "char": "Character"}, ["Number", "Character"]),
    ],
)
def test_rename_columns_success(default_manipulator, rename_dict, expected_columns):
    manipulator = default_manipulator
    manipulator.rename(rename_dict)

    # # Check if columns have been renamed correctly
    for old, new in rename_dict.items():
        assert old not in manipulator.columns
        assert new in manipulator.columns

    # Further verify that the order and total columns match expectations
    assert list(manipulator.columns) == expected_columns


def test_rename_nonexistent_column(default_manipulator):
    manipulator = default_manipulator
    original_columns = list(manipulator.columns)
    with pytest.raises(ArrayManipulationError):
        manipulator.rename({"nonexistent_column": "new_column"})
    assert list(manipulator.columns) == original_columns


def test_rename_with_duplicate_column_name(default_manipulator):
    manipulator = default_manipulator
    with pytest.raises(ArrayManipulationError):
        manipulator.rename({"num": "char"})  # "char" already exists


def test_change_multiple_dtypes_success(default_manipulator):
    manipulator = default_manipulator
    new_dtypes = {
        "num": "f8",
        "char": "U10",
    }  # Changing 'num' to float and 'char' to a longer string type

    # Store old values for comparison
    old_num_values = manipulator.values["num"].copy()
    old_char_values = manipulator.values["char"].copy()

    manipulator.change_dtypes(new_dtypes)

    # Check if dtypes have been changed successfully
    assert manipulator.values.dtype["num"] == np.dtype("f8")
    assert manipulator.values.dtype["char"] == np.dtype("U10")

    # Check if values have been preserved after dtypes change
    np.testing.assert_array_almost_equal(manipulator.values["num"], old_num_values)
    assert isinstance(old_char_values, np.ndarray)
    assert all(manipulator.values["char"] == old_char_values)


def test_change_dtypes_invalid_conversion(default_manipulator):
    manipulator = default_manipulator
    invalid_dtypes = {
        "char": "i4"
    }  # Trying to convert a string column to an integer type

    # Expect ArrayManipulationError to be raised due to invalid conversion
    with pytest.raises(ArrayManipulationError):
        manipulator.change_dtypes(invalid_dtypes)


def test_change_dtypes_for_nonexistent_column(default_manipulator):
    manipulator = default_manipulator
    dtypes_for_nonexistent_column = {
        "nonexistent_column": "f8"
    }  # Attempting to change dtype for a column that doesn't exist

    # Expect ArrayManipulationError to be raised due to non-existent column
    with pytest.raises(ArrayManipulationError):
        manipulator.change_dtypes(dtypes_for_nonexistent_column)


def test_to_numpy_success(default_manipulator):
    result = default_manipulator.to_numpy()

    # Expected result is a 2D NumPy array with the same values
    expected = np.array([[1, "a"], [2, "b"], [3, "c"]])

    # Since we are dealing with heterogeneous types, we need to compare the string representation
    np.testing.assert_array_equal(result.astype(str), expected.astype(str))


def test_to_numpy_empty_array():
    empty_values = np.array([], dtype=[("num", "i4"), ("char", "U1")])
    empty_manipulator = StructuredArrayManipulator(empty_values, np.array(["num", "char"]))
    result = empty_manipulator.to_numpy()

    expected = np.empty((0, 2))
    np.testing.assert_array_equal(result, expected)

