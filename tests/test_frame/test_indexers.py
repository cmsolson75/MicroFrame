import pytest
import numpy as np
from frame.indexers import StructuredArrayIndexer

values = np.array([(1, 'a'), (2, 'b'), (3, 'c')],
                  dtype=[('num', 'i4'), ('char', 'U1')])
columns = ['num', 'char']
structured_array_indexer = StructuredArrayIndexer(values, columns)


def test_structured_array_indexer_instantiation():
    local_values = np.array([(1, 'a'), (2, 'b'), (3, 'c')],
                            dtype=[('num', 'i4'), ('char', 'U1')])
    local_columns = ['num', 'char']
    local_structured_array_indexer = StructuredArrayIndexer(local_values, local_columns)

    assert isinstance(local_structured_array_indexer, StructuredArrayIndexer)
    assert local_structured_array_indexer.values is local_values
    assert local_structured_array_indexer.columns == local_columns


@pytest.mark.parametrize("index, expected", [
    (1, (2, 'b')),
    (2, (3, 'c')),
])
def test_structured_array_indexer_getitem_with_row_index(index, expected):
    result = structured_array_indexer[index]
    assert tuple(result) == expected


@pytest.mark.parametrize("row_index, col_index, expected", [
    (1, 1, 'b'),
    (2, 0, 3),
])
def test_iloc_getitem_with_row_and_column_indices(row_index, col_index, expected):
    result = structured_array_indexer[row_index, col_index]
    assert result == expected


@pytest.mark.parametrize("row_index, col_index", [
    (10, 1),
    (1, 10),
])
def test_iloc_getitem_invalid_indices(row_index, col_index):
    with pytest.raises(IndexError):
        _ = structured_array_indexer[row_index, col_index]


@pytest.mark.parametrize("row_index, col_index, value", [
    (1, 1, 'z'),
    (0, 0, 10),
])
def test_iloc_setitem(row_index, col_index, value):
    structured_array_indexer[row_index, col_index] = value
    column_name = columns[col_index]
    assert structured_array_indexer.values[row_index][column_name] == value


@pytest.mark.parametrize("index", [1, 2, ])
def test_iloc_setitem_without_column_index(index):
    with pytest.raises(ValueError):
        structured_array_indexer[index] = 'z'


@pytest.mark.parametrize("row_index, col_index, value", [
    (10, 1, 'z'),
    (1, 10, 10),
])
def test_iloc_setitem_invalid_indices(row_index, col_index, value):
    with pytest.raises(IndexError):
        structured_array_indexer[row_index, col_index] = value
