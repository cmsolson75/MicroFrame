# MicroFrame


MicroFrame is a lightweight, efficient data manipulation library in Python, designed for handling structured data with ease. Inspired by the functionality of pandas, MicroFrame offers a simplified approach to data manipulation, making it ideal for small to medium-sized datasets.

## Features

- **CSV Reading**: Read CSV files efficiently and construct MicroFrame objects with ease.
- **Data Type Inference**: Automatically infer the data types of columns based on the first row of the data.
- **MicroFrame Objects**: Work with data encapsulated in MicroFrame objects that offer flexibility and powerful analytical capabilities.
- **Tabular Printout**: Enjoy clear and organized tabular printouts of your data for easy analysis and presentation.
- **Data Manipulation**: Leverage robust data manipulation tools to transform and analyze your datasets effectively.

## Installation

To install MicroFrame from the GitHub repository, follow these steps:

```bash
git clone https://github.com/cmsolson75/MicroFrame.git
cd MicroFrame
pip install -r requirements.txt
```

Ensure you have Python 3.x installed on your system to use MicroFrame.

## Usage

### Reading a CSV File

```python
from microframe.frame import read_csv

# Read a CSV file and construct a MicroFrame object
microframe = read_csv('path/to/your/file.csv')

# Now you can work with microframe as needed
# For example, print the head of the table
microframe.head()
```

### Instantiating a MicroFrame Object

You can create a MicroFrame object manually by providing data, data types, and column names:

```python
from microframe.frame.microframe import MicroFrame

# Sample data to be encapsulated in the MicroFrame object
data = [
    ["1", "Alice", "2023-01-01"],
    ["2", "Bob", "2023-01-02"],
    ["3", "Charlie", "2023-01-03"]
]
dtypes = ["int32", "U100", "U100"]
columns = ["ID", "Name", "Date"]

# Creating a MicroFrame object
microframe = MicroFrame(data, dtypes, columns)
```

### Using the Printer Class

To print out your data in a tabular format, you can use the Printer class:

```python
import numpy as np
from microframe.frame.printers import StructuredDataPrinter

# Sample structured data
values = np.array(
    [(1, "a"), (2, "b"), (3, "c")],
    dtype=[("num", "i4"), ("char", "U1")]
)
columns = ["num", "char"]

# Creating a Printer instance
printer = StructuredDataPrinter(values, columns)

# Print a tabular view of your data
printer.print_table()
```

For more detailed examples and usage, please refer to the [Documentation](https://cmsolson75.github.io/MicroFrame/).

## Documentation

For full documentation, visit our [MicroFrame Documentation](https://cmsolson75.github.io/MicroFrame/). Here, you will find detailed information on all the functionalities that MicroFrame offers.

## Contributing

We welcome contributions to MicroFrame! If you have ideas for improvements or want to help out with development, please read our contributing guidelines and submit your pull requests.

## License

MicroFrame is released under the [MIT License](#). Feel free to use it in your projects, and we'd love to hear about what you build!

