import pytest
import numpy as np
from microframe.core.indexers import StructuredArrayIndexer, IlocIndexer
from microframe.core.microframe import MicroFrame

values = np.array([(1, "a"), (2, "b"), (3, "c")], dtype=[("num", "i4"), ("char", "U1")])
columns = np.array(["num", "char"])
structured_array_indexer = StructuredArrayIndexer(values, columns)


@pytest.fixture
def default_microframe():
    data = [["1", "a"], ["2", "b"], ["3", "c"]]
    dtypes = ["float32", "U100"]
    columns = ["num", "char"]
    return MicroFrame(data, dtypes, columns)


def test_structured_array_indexer_instantiation():
    local_values = np.array(
        [(1, "a"), (2, "b"), (3, "c")], dtype=[("num", "i4"), ("char", "U1")]
    )
    local_columns = np.array(["num", "char"])
    local_structured_array_indexer = StructuredArrayIndexer(local_values, local_columns)

    assert isinstance(local_structured_array_indexer, StructuredArrayIndexer)
    assert local_structured_array_indexer.values is local_values
    assert local_structured_array_indexer.columns is local_columns


@pytest.mark.parametrize(
    "index, expected",
    [
        (1, (2, "b")),
        (2, (3, "c")),
    ],
)
def test_structured_array_indexer_getitem_with_row_index(index, expected):
    result = structured_array_indexer[index]
    assert tuple(result) == expected


@pytest.mark.parametrize(
    "row_index, col_index, expected",
    [
        (1, 1, "b"),
        (2, 0, 3),
    ],
)
def test_structured_array_indexer_getitem_with_row_and_column_indices(row_index, col_index, expected):
    result = structured_array_indexer[row_index, col_index]
    assert result == expected


@pytest.mark.parametrize(
    "row_index, col_index",
    [
        (10, 1),
        (1, 10),
    ],
)
def test_structured_array_indexer_getitem_invalid_indices(row_index, col_index):
    with pytest.raises(IndexError):
        _ = structured_array_indexer[row_index, col_index]


@pytest.mark.parametrize(
    "row_index, col_index, value",
    [
        (1, 1, "z"),
        (0, 0, 10),
    ],
)
def test_structured_array_indexer_setitem(row_index, col_index, value):
    structured_array_indexer[row_index, col_index] = value
    column_name = columns[col_index]
    assert structured_array_indexer.values[row_index][column_name] == value


@pytest.mark.parametrize(
    "index",
    [
        1,
        2,
    ],
)
def test_structured_array_indexer_setitem_without_column_index(index):
    with pytest.raises(ValueError):
        structured_array_indexer[index] = "z"


@pytest.mark.parametrize(
    "row_index, col_index, value",
    [
        (10, 1, "z"),
        (1, 10, 10),
    ],
)
def test_structured_array_indexer_setitem_invalid_indices(row_index, col_index, value):
    with pytest.raises(IndexError):
        structured_array_indexer[row_index, col_index] = value


def test_iloc_indexer_instantiation_microframe(default_microframe):
    # Test the instantiation of IlocIndexer with MicroFrame
    iloc_indexer = IlocIndexer(default_microframe.values, default_microframe.columns, return_type=MicroFrame)
    assert isinstance(iloc_indexer, IlocIndexer)
    assert iloc_indexer.return_type is MicroFrame


def test_iloc_indexer_getitem_row_microframe(default_microframe):
    # Test the __getitem__ method for a single row
    iloc_indexer = IlocIndexer(default_microframe.values, default_microframe.columns, return_type=MicroFrame)
    result = iloc_indexer[1]  # Get the second row
    assert isinstance(result, MicroFrame)
    assert np.array_equal(result.values, np.array([(2, "b")], dtype=[("num", "<f4"), ("char", "<U100")]))


def test_iloc_indexer_getitem_slice_microframe(default_microframe):
    # Test the __getitem__ method for a slice
    iloc_indexer = IlocIndexer(default_microframe.values, default_microframe.columns, return_type=MicroFrame)
    result = iloc_indexer[1:3]  # Get the second and third rows
    assert isinstance(result, MicroFrame)
    assert result.shape == (2, 2)
    assert np.array_equal(result.values['num'], [2.0, 3.0])
    assert np.array_equal(result.values['char'], ['b', 'c'])


def test_iloc_indexer_getitem_column_microframe(default_microframe):
    # Test the __getitem__ method for a single column
    iloc_indexer = IlocIndexer(default_microframe.values, default_microframe.columns, return_type=MicroFrame)
    result = iloc_indexer[:, 1]  # Get the second column
    assert isinstance(result, MicroFrame)
    assert np.array_equal(result.values, np.array([("a",), ("b",), ("c",)], dtype=[("char", "<U100")]))


def test_iloc_indexer_setitem_microframe(default_microframe):
    # Test the __setitem__ method
    iloc_indexer = IlocIndexer(default_microframe.values, default_microframe.columns, return_type=MicroFrame)
    iloc_indexer[2, 1] = "Test"  # Set a value in the third row, second column
    assert default_microframe.values[2][1] == "Test"

