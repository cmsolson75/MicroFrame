class StructuredDataPrinter:
    """
    A class for displaying numpy structured arrays or lists of tuples in a tabular format.

    :param values: The data to be printed.
    :type values: numpy.ndarray or list
    :param columns: Column names corresponding to the data.
    :type columns: list
    :param max_value_length: Maximum display length for cell values, defaults to 20.
    :type max_value_length: int, optional

    :ivar values: The data to be printed.
    :ivar columns: Column names corresponding to the data.
    :ivar max_value_length: Maximum display length for cell values.

    Example:
        >>> import numpy as np
        >>> # Initialize a numpy structured array for data
        >>> data = np.array([(0, 10. , 'Item 1'), (1, 20.5, 'Item 2'), (2, 30.2, 'Item 3'), (3, 40.8, 'Item 4'),
        >>>                 (4, 50.1, 'Item 5')], dtype=[('id', '<i4'), ('value', '<f4'), ('description', '<U25')])
        >>> # Initialize normal numpy array for columns
        >>> columns = np.array(['id', 'value', 'description'])
        >>> printer = StructuredDataPrinter(data, columns)
    """

    def __init__(self, values, columns, max_value_length=20):
        """
        Initializes the StructuredDataPrinter with data, columns, and an optional maximum value length.
        """
        self.values = values
        self.columns = columns
        self.column_widths = None
        self.header_widths = None
        self.max_widths = None
        self.max_value_length = max_value_length
        self.subset_rows = None

    def _truncate_value(self, value):
        """
        Truncate a value if it exceeds the maximum length set for the column.

        :param value: The value to be truncated.
        :type value: str
        :return: The truncated value with an ellipsis if it exceeds the maximum length.
        :rtype: str
        """
        str_value = str(value)
        if len(str_value) > self.max_value_length:
            return str_value[: self.max_value_length - 3] + "..."
        return str_value

    def _print_all_columns(self):
        """
        Print all columns of the structured data without truncating any columns.
        This method formats the header and the rows and then prints them to the console.
        """
        h = self.columns
        truncated_headers = [self._truncate_value(header) for header in h]
        header_str = "  ".join(
            [truncated_headers[i].ljust(self.max_widths[i]) for i in range(len(h))]
        )
        print(header_str)
        print("-" * len(header_str))
        for row in self.subset_rows:
            print(
                "  ".join(
                    [
                        self._truncate_value(row[i]).ljust(self.max_widths[i])
                        for i in range(len(h))
                    ]
                )
            )

    def _print_truncated_columns(self, num_cols, max_width):
        """
        Print a truncated version of the columns based on the maximum width and specified number of columns.

        :param num_cols: The number of columns to display.
        :type num_cols: int
        :param max_width: The maximum width of the display.
        :type max_width: int
        """
        left_cols, right_cols = self._determine_columns_to_show(max_width, num_cols)
        headers = self.columns
        left_headers = headers[:left_cols]
        right_headers = headers[-left_cols - right_cols :]
        left_widths = self.max_widths[:left_cols]
        right_widths = self.max_widths[-left_cols - right_cols :]

        header_str = "  ".join(
            [
                self._truncate_value(left_headers[i]).ljust(left_widths[i])
                for i in range(left_cols)
            ]
        )
        if right_cols > 0:
            header_str += " ... " + "  ".join(
                [
                    self._truncate_value(right_headers[i]).ljust(right_widths[i])
                    for i in range(right_cols)
                ]
            )

        print(header_str)
        print("-" * len(header_str))
        for row in self.subset_rows:
            left_items = [self._truncate_value(row[i]) for i in range(left_cols)]
            right_items = [
                self._truncate_value(row[i])
                for i in range(len(headers) - right_cols, len(headers))
            ]
            row_str = "  ".join(
                [left_items[i].ljust(left_widths[i]) for i in range(left_cols)]
            )
            if right_cols > 0:
                row_str += " ... " + "  ".join(
                    [right_items[i].ljust(right_widths[i]) for i in range(right_cols)]
                )
            print(row_str)

    def _compute_widths(self, num_rows, tail=False):
        """
        Compute the widths for the columns based on a subset of rows from the data.

        :param num_rows: The number of rows to consider for computing widths.
        :type num_rows: int
        :param tail: Whether to consider the last rows instead of the first rows, defaults to False.
        :type tail: bool, optional
        """
        headers = self.columns
        if tail:
            self.subset_rows = self.values[-num_rows:]
        else:
            self.subset_rows = self.values[:num_rows]
        truncated_headers = [self._truncate_value(header) for header in headers]
        self.column_widths = [
            max(len(self._truncate_value(row[i])) for row in self.subset_rows)
            for i in range(len(headers))
        ]
        self.header_widths = [len(header) for header in truncated_headers]
        self.max_widths = [
            max(self.column_widths[i], self.header_widths[i])
            for i in range(len(headers))
        ]

    def _determine_columns_to_show(self, max_width, num_cols=None):
        """
        Determine the number of columns to show based on the maximum width and optionally the number of columns.

        :param max_width: The maximum width of the display.
        :type max_width: int
        :param num_cols: The number of columns to display, defaults to None.
        :type num_cols: int, optional
        :return: A tuple containing the number of columns to show on the left and right.
        :rtype: (int, int)
        """
        left_cols, right_cols = 0, 0
        curr_width = 0
        headers = self.columns

        if num_cols is not None:
            left_cols = num_cols // 2
            right_cols = num_cols - left_cols
            return left_cols, right_cols

        for i in range(len(headers)):
            curr_width += self.max_widths[i] + 2
            if curr_width > max_width:
                break
            if i % 2 == 0:
                left_cols += 1
            else:
                right_cols += 1

        return left_cols, right_cols

    def structured_print(self, max_width=80, num_cols=None, num_rows=10, tail=False):
        """
        Prints the data in a tabular format with configurable display options.

        :param max_width: Maximum width of the table, defaults to 80.
        :type max_width: int, optional
        :param num_cols: Number of columns to display, defaults to None.
        :type num_cols: int, optional
        :param num_rows: Number of rows to display, defaults to 10.
        :type num_rows: int, optional
        :param tail: Whether to display the last rows instead of the first, defaults to False.
        :type tail: bool, optional

        Example:

            >>> printer.structured_print() # Print the data in the default tabular format.
            >>> printer.structured_print(max_width=50) # Print the data with a constrained width
            >>> printer.structured_print(num_cols=3) # Print 3 columns of data.
            >>> printer.structured_print(num_rows=15) # Print the first 15 rows of data
            >>> printer.structured_print(num_rows=5, tail=True) # If you want to see the end of a large dataset, use the `tail` parameter to print the last rows.
        """

        self._compute_widths(num_rows, tail)
        total_width = sum(self.max_widths) + len(self.columns) - 1
        if num_cols is not None or total_width > max_width:
            self._print_truncated_columns(num_cols, max_width)
        else:
            self._print_all_columns()

        total_num_rows = len(self.values)
        total_num_cols = len(self.columns)
        if num_rows + 1 <= total_num_rows:
            print("...")
        print(f"{total_num_rows} rows x {total_num_cols} columns")
