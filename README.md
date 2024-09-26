# funcnodes-pandas

`funcnodes-pandas` is an extension for the [Funcnodes](https://github.com/linkdlab/funcnodes) framework that allows you to manipulate [Pandas](https://pandas.pydata.org/) DataFrames and Series using **FuncNodes**' visual node-based system. It provides a collection of nodes for performing typical operations on Pandas data structures, such as conversions, data manipulations, and calculations.

This library enables **no-code** and **low-code** workflows for **Pandas** by providing drag-and-drop functionality in a visual interface. It also supports Python-based scripting to handle more complex operations.

## Features

- **DataFrame Conversion**:
  - Convert DataFrames to dictionaries and vice versa.
  - Handle CSV and Excel files easily using DataFrame nodes.
- **Data Manipulation**:

  - Add, drop, and manipulate rows and columns of a DataFrame.
  - Handle missing data with nodes for `fillna`, `dropna`, `ffill`, and `bfill`.
  - Perform merges and joins with intuitive nodes for `merge`, `concatenate`, and `join`.

- **Math & Statistical Operations**:

  - Perform descriptive statistics like `mean`, `sum`, `std`, `var`, and `corr`.
  - Evaluate custom expressions directly on DataFrames using the `eval` node.

- **Masking & Filtering**:

  - Apply masks to filter DataFrame data.
  - Use conditions to filter rows and columns dynamically.

- **Grouping & Aggregation**:

  - Group data using `groupby` and aggregate it with `sum`, `mean`, `count`, etc.
  - Easily convert groups into lists of DataFrames.

- **Series Support**:
  - Nodes for converting Series to lists and dictionaries.
  - Access individual elements using `iloc` and `loc`.
  - Perform string operations on Series.

## Installation

Install the package with:

```bash
pip install funcnodes-pandas
```

Ensure that you have **Pandas** and **FuncNodes** installed.

## Getting Started

Here's an overview of the basic programmatically usage of **funcnodes-pandas**.

1. **Convert a DataFrame to a Dictionary**

```python
import pandas as pd
import funcnodes_pandas as fnpd

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
node = fnpd.to_dict()
node.inputs['df'].value = df
await node
print(node.outputs['dict'].value)
```

This code converts a DataFrame to a dictionary using the `to_dict` node.

2. **Filling Missing Data**

```python
node = fnpd.fillna()
node.inputs["df"].value = df
node.inputs["value"].value = 0
await node
print(node.outputs["out"].value)
```

The `fillna` node fills missing data in a DataFrame.

3. **Group By Operations**

```python
node = fnpd.group_by()
node.inputs["df"].value = df
node.inputs["by"].value = "A"
await node
print(node.outputs["grouped"].value)
```

This groups data based on column `A` in the DataFrame.

## Testing

The repository contains a suite of tests to ensure that the various functionalities of `funcnodes-pandas` work as expected. The tests are based on **unittest** and **IsolatedAsyncioTestCase**. You can run the tests using:

```bash
python -m unittest discover
```

Test cases for operations such as `groupby`, `add_column`, `dropna`, etc., are included.

## Contribution

Feel free to contribute to this project by submitting pull requests. You can help by adding new nodes, fixing bugs, or enhancing documentation.

## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please open an issue on the GitHub repository.
