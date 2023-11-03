import pytest
import numpy as np
from frame.printers import StructuredDataPrinter


@pytest.fixture
def default_printer():
    values = np.array([(1, 'a'), (2, 'b'), (3, 'c')],
                      dtype=[('num', 'i4'), ('char', 'U1')])
    columns = ['num', 'char']
    printer = StructuredDataPrinter(values, columns)

    # Confirm that max_value_length is set to the default value (20)
    assert printer.max_value_length == 20

    return printer


@pytest.fixture
def printer_with_custom_max_length():
    values = []
    columns = []
    printer = StructuredDataPrinter(values, columns)
    printer.max_value_length = 30  # Set a larger max length
    return printer


@pytest.fixture
def varied_printer():
    # Creating a dataset with varying column widths
    values = np.array([(1, 'apple', 1000), (2, 'banana', 2000), (3, 'cherry', 3000)],
                      dtype=[('id', 'i4'), ('fruit', 'U10'), ('quantity', 'i4')])
    columns = ['id', 'fruit', 'quantity']
    return StructuredDataPrinter(values, columns)


@pytest.fixture
def printer_with_many_columns():
    values = np.array([(1, 'a', 'long_value1_value', 'extra1'),
                       (2, 'b', 'long_value2_value', 'extra2'),
                       (3, 'c', 'long_value3_value', 'extra3'),
                       (4, 'd', 'long_value4_value', 'extra4')],
                      dtype=[('num', 'i4'), ('char', 'U1'), ('long', 'U100'), ('extra', 'U100')])
    columns = np.array(['num', 'char', 'long', 'extra'])
    return StructuredDataPrinter(values, columns)


def test_printer_initialization(default_printer):
    printer = default_printer

    assert isinstance(printer, StructuredDataPrinter)
    assert printer.values.dtype.names == ('num', 'char')
    assert printer.columns == ['num', 'char']
    assert printer.max_value_length == 20  # Assuming the default max_value_length
    assert printer.column_widths is None
    assert printer.header_widths is None
    assert printer.max_widths is None
    assert printer.subset_rows is None


@pytest.mark.parametrize("printer, value, expected", [
    (pytest.lazy_fixture('default_printer'), "Short Value", "Short Value"),
    (pytest.lazy_fixture('default_printer'), "This is a long value that needs to be truncated", "This is a long va..."),
    (pytest.lazy_fixture('default_printer'), "1234567890", "1234567890"),
    (pytest.lazy_fixture('default_printer'), "", ""),
    (pytest.lazy_fixture('default_printer'), "A" * 20, "A" * 20),
    (pytest.lazy_fixture('printer_with_custom_max_length'), "Short Value", "Short Value"),
])
def test_truncate_value(printer, value, expected):
    result = printer._truncate_value(value)
    assert result == expected


@pytest.mark.parametrize("num_rows, expected_column_widths, expected_header_widths, expected_max_widths", [
    (3, [1, 1], [3, 4], [3, 4]),  # Adjusted to match the data
    (2, [1, 1], [3, 4], [3, 4]),  # Adjusted to match the data
    (1, [1, 1], [3, 4], [3, 4]),  # Adjusted to match the data
])
def test_compute_widths(default_printer, num_rows, expected_column_widths, expected_header_widths, expected_max_widths):
    printer = default_printer

    # Call the method to compute widths (modify if the actual method is different)
    printer._compute_widths(num_rows=num_rows)

    # Verify column widths
    assert printer.column_widths == expected_column_widths

    # Verify header widths
    assert printer.header_widths == expected_header_widths

    # Verify max widths (the greatest of column widths and header widths)
    assert printer.max_widths == expected_max_widths

    # Verify subset rows
    expected_subset_rows = printer.values[:num_rows]  # Subset based on num_rows
    np.testing.assert_array_equal(printer.subset_rows, expected_subset_rows)


@pytest.mark.parametrize("max_width, num_cols, expected_left_cols, expected_right_cols", [
    (80, None, 2, 1),  # Wide enough to fit all columns
    (10, None, 1, 0),  # Only wide enough for the first column
    (20, None, 1, 1),  # Wide enough for the first two columns
    (80, 2, 1, 1),  # Limiting the number of columns to 2
])
def test_determine_columns_to_show(varied_printer, max_width, num_cols, expected_left_cols, expected_right_cols):
    printer = varied_printer
    printer._compute_widths(num_rows=len(printer.values))  # Ensure the widths are computed

    # Call the method to determine columns to show
    left_cols, right_cols = printer._determine_columns_to_show(max_width, num_cols)

    # Verify the number of left and right columns
    assert left_cols == expected_left_cols
    assert right_cols == expected_right_cols


def test_print_all_columns(default_printer, capsys):
    # Set up the printer with a width that can fit all columns
    default_printer._compute_widths(num_rows=3)
    total_width = sum(default_printer.max_widths) + len(default_printer.columns) - 1
    assert total_width <= 80  # Assuming 80 is the maximum width

    # Invoke the method to print all columns
    default_printer._print_all_columns()

    # Capture the output and verify it
    captured = capsys.readouterr()
    expected_output = (
        "num  char\n"
        "---------\n"
        "1    a   \n"
        "2    b   \n"
        "3    c   \n"
    )
    assert captured.out == expected_output


def test_print_truncated_columns(default_printer, capsys):
    # Set up the printer with a width that can fit only a subset of columns
    default_printer._compute_widths(num_rows=3)
    max_width = 20  # Set a small max_width to truncate columns
    left_cols, right_cols = default_printer._determine_columns_to_show(max_width)

    # Invoke the method to print truncated columns
    default_printer._print_truncated_columns(left_cols, max_width)

    # Capture the output and verify it
    captured = capsys.readouterr()
    expected_output = (
        " ... char\n"
        "---------\n"
        " ... a   \n"
        " ... b   \n"
        " ... c   \n"
    )
    assert captured.out == expected_output


def test_structured_print_default(default_printer, capsys):
    # Set up the printer with a width that can fit all columns
    default_printer._compute_widths(num_rows=3)
    max_width = 80

    # Invoke the structured_print method
    default_printer.structured_print(max_width=max_width)

    # Capture the output and verify it
    captured = capsys.readouterr()
    expected_output = (
        "num  char\n"
        "---------\n"
        "1    a   \n"
        "2    b   \n"
        "3    c   \n"
        "3 rows x 2 columns\n"
    )
    assert captured.out == expected_output


def test_structured_print_truncated(printer_with_many_columns, capsys):
    # Invoke the structured_print method with num_cols set to 2 to trigger truncation
    printer_with_many_columns.structured_print(num_cols=2)

    # Capture the output and verify it
    captured = capsys.readouterr()
    expected_output = (
        "num ... long             \n"
        "-------------------------\n"
        "1   ... extra1           \n"
        "2   ... extra2           \n"
        "3   ... extra3           \n"
        "4   ... extra4           \n"
        "4 rows x 4 columns\n"
    )
    assert captured.out == expected_output


def test_structured_print_tail(default_printer, capsys):
    # Set up the printer to print the last 2 rows
    num_rows = 2

    # Invoke the structured_print method with tail=True
    default_printer.structured_print(num_rows=num_rows, tail=True)

    # Capture the output and verify it
    captured = capsys.readouterr()
    expected_output = (
        "num  char\n"
        "---------\n"
        "2    b   \n"
        "3    c   \n"
        "...\n"
        "3 rows x 2 columns\n"
    )
    assert captured.out == expected_output


def test_structured_print_truncated_tail(printer_with_many_columns, capsys):
    printer_with_many_columns.structured_print(num_cols=2, num_rows=2, tail=True)

    captured = capsys.readouterr()
    expected_output = (
        "num ... long             \n"
        "-------------------------\n"
        "3   ... extra3           \n"
        "4   ... extra4           \n"
        "...\n"
        "4 rows x 4 columns\n"
    )
    assert captured.out == expected_output
