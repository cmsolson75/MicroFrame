
# MicroFrame

MicroFrame is a lightweight educational data manipulation library designed to provide a pandas-like interface for students learning to work with real-world data. It is optimized for toy datasets and aims to introduce users to data analysis concepts without the overhead of pandas.

## Features

- **Efficient CSV Reading**: Quickly and easily read CSV files to create MicroFrame objects, optimized for educational purposes and smaller datasets.
- **Data Type Handling**: Advanced data type inference and explicit type setting offer both convenience and control over the structure of your data.
- **Flexible MicroFrame Objects**: Utilize MicroFrame objects that mimic pandas DataFrame for intuitive data manipulation and analysis.
- **Clear Tabular Display**: Use MicroFrame’s printing capabilities to generate well-formatted tabular representations of your data, making it easier to interpret and present.
- **Robust Data Manipulation**: Perform a variety of data manipulation tasks with methods similar to pandas, such as filtering, column dtype modification and summarizing data.
- **Advanced Indexing**: Access data efficiently with advanced indexing options, using the `iloc` method similar to pandas iloc method.
- **Data Conversion Tools**: Seamlessly convert your MicroFrame objects to other formats, including NumPy arrays, with the `to_numpy` method for further numerical computation.
- **User-Friendly API**: Experience a user-friendly API that mirrors pandas to facilitate the transition from educational projects to real-world data analysis.


## Installation

Install MicroFrame using pip:

```bash
pip install microframe
```

## Quick Start

Import MicroFrame and load a dataset:

```python
import microframe as mf

# Read a CSV file into a MicroFrame object
mframe = mf.read_csv("path_to_your_csv_file.csv")

# Alternatively, create a MicroFrame object manually
data = [[1, "a"], [2, "b"], [3, "c"]]
columns = ["num", "char"]
dtypes = ["int32", "U1"]

mframe = mf.MicroFrame(data, columns, dtypes)
```

## Examples

### Basic Usage

MicroFrame simplifies the process of data analysis. Here are some basic operations:

#### Reading from a CSV

```python
import microframe as mf

mframe = mf.read_csv("path_to_your_csv_file.csv")
```

#### Creating a MicroFrame Object

```python
data = [[1, "a"], [2, "b"], [3, "c"]]
columns = ["num", "char"]
dtypes = ["int32", "U1"]

mframe = mf.MicroFrame(data, columns, dtypes)
```

### Data Manipulation

MicroFrame provides several methods to manipulate your data:

#### Renaming Columns

```python
mframe.rename({"num": "number", "char": "character"})
```

#### Changing Data Types

```python
mframe.change_dtypes({"number": "float64", "character": "U10"})
```

#### Accessing Column Data with Boolean Indexing

```python
data = [[1, "a"], [2, "b"], [3, "c"]]
columns = ["num", "char"]
dtypes = ["int32", "U1"]

mframe = mf.MicroFrame(data, columns, dtypes)
first_col = mframe["num"] # Access just num column
```
#### Accessing Row Data with `iloc`

```python
first_row = mframe.iloc[0]
```

### Advanced Indexing with `iloc`

The `iloc` indexer allows for integer-location based indexing:

#### Accessing a Single Row

```python
first_row = mframe.iloc[0]
```

#### Accessing Multiple Rows

```python
first_two_rows = mframe.iloc[:2]
```

#### Accessing a Single Cell

```python
cell_value = mframe.iloc[2, 1]
```

#### Setting a Value in a Cell

```python
mframe.iloc[2, 1] = "Test"
```

#### Slicing Rows and Columns

```python
subset = mframe.iloc[:2, :2]
```


### Displaying Data

Similar to pandas, you can display parts of your dataset:

#### Print the First Few Rows

```python
mframe.head(2)
```

#### Print the Last Few Rows

```python
mframe.tail(2)
```

### Converting to NumPy Array

For times when you need to work with a NumPy array, MicroFrame provides the `to_numpy` method:

```python
# Convert the MicroFrame to a 2D NumPy array
numpy_array = mframe.to_numpy()
```

This method will convert the structured data within the MicroFrame to a regular 2D NumPy array.

### Chaining `iloc` with `to_numpy`

For scenarios where you need to perform NumPy operations on a subset of your data, you can chain the `iloc` indexer with the `to_numpy` method:

```python
# Select the first two rows using iloc and convert them to a NumPy array
numpy_subset = mframe.iloc[:, 1:5].to_numpy()
```

## Documentation

For full documentation, visit our [MicroFrame Documentation](https://cmsolson75.github.io/MicroFrame/). Here, you will find detailed information on all the functionalities that MicroFrame offers.


## License

MicroFrame is released under the [MIT License](#). Feel free to use it in your projects, and we'd love to hear about what you build!
