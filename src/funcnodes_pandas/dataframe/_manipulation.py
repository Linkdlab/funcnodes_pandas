import pandas as pd
import funcnodes as fn
from typing import Optional, Literal, Union, Any, List
import numpy as np
# region nan


@fn.NodeDecorator(
    node_id="pd.df_reset_index",
    name="Reset Index",
    description="Resets the index of a DataFrame.",
)
def df_reset_index(
    df: pd.DataFrame,
    drop: bool = False,
) -> pd.DataFrame:
    """
    Resets the index of a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - drop: bool, if True, the old index is not added as a column

    Returns:
    - A new DataFrame with the index reset or modifies the original DataFrame in place
    """
    return df.reset_index(drop=drop)


@fn.NodeDecorator(
    node_id="pd.dropna",
    name="Drop NA",
    description="Drops rows or columns with NA values.",
)
def dropna(
    df: pd.DataFrame,
    axis: Literal["index", "columns"] = "index",
    how: Literal["any", "all"] = "any",
    subset: Optional[str] = None,
) -> pd.DataFrame:
    if subset is not None:
        subset = [s.strip() for s in subset.split(",")]

    return df.dropna(axis=axis, how=how, subset=subset)


@fn.NodeDecorator(
    node_id="pd.fillna",
    name="Fill NA",
    description="Fills NA values with a specified value.",
)
def fillna(
    df: pd.DataFrame,
    value: Union[int, float, str] = 0,
) -> pd.DataFrame:
    return df.fillna(value)


@fn.NodeDecorator(
    node_id="pd.bfill",
    name="Backfill",
    description="Backfills NA values.",
)
def bfill(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return df.bfill()


@fn.NodeDecorator(
    node_id="pd.ffill",
    name="Forwardfill",
    description="Forwardfills NA values.",
)
def ffill(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return df.ffill()


# endregion nan


# region duplicates


@fn.NodeDecorator(
    node_id="pd.drop_duplicates",
    name="Drop Duplicates",
    description="Drops duplicate rows.",
)
def drop_duplicates(
    df: pd.DataFrame,
    subset: Optional[str] = None,
) -> pd.DataFrame:
    if subset is not None:
        subset = [s.strip() for s in subset.split(",")]
    return df.drop_duplicates(subset=subset)


# endregion duplicates


# region filter


@fn.NodeDecorator(
    node_id="pd.numeric_only",
    name="Numeric Only",
)
def numeric_only(df: pd.DataFrame, label_encode: bool = False) -> pd.DataFrame:
    """
    Converts a DataFrame to only hold numeric values.
    Optionally, non-numeric values can be converted to numeric labels.

    Parameters:
    - df: pandas DataFrame
    - label_encode: bool, if True, convert non-numeric values to numeric labels

    Returns:
    - A new DataFrame containing only numeric values
    """

    df = df.copy()
    for column in df.select_dtypes(exclude=[np.number]):
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            pass

    if label_encode:
        for column in df.select_dtypes(include=["object", "category"]):
            df[column] = df[column].astype("category").cat.codes

    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df


# endregion filter

# region drop


@fn.NodeDecorator(
    node_id="pd.drop_column",
    name="Drop Column",
    description="Drops a column from a DataFrame.",
    default_io_options={
        "df": {
            "on": {
                "after_set_value": fn.decorator.update_other_io_options(
                    "column",
                    lambda x: list(x.columns),
                )
            }
        },
    },
)
def DropColumnNode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.drop(column, axis=1)


@fn.NodeDecorator(
    node_id="pd.drop_row",
    name="Drop Row",
    description="Drops a row from a DataFrame.",
    default_io_options={
        "df": {
            "on": {
                "after_set_value": fn.decorator.update_other_io_options(
                    "row",
                    lambda x: list(x.index),
                )
            }
        },
    },
)
def DropRowNode(df: pd.DataFrame, row: str) -> pd.DataFrame:
    return df.drop(df.index.to_list()[0].__class__(row), axis=0)


@fn.NodeDecorator(
    node_id="pd.drop_columns",
    name="Drop Columns",
    description="Drops columns from a DataFrame.",
)
def drop_columns(
    df: pd.DataFrame,
    columns: str,
) -> pd.DataFrame:
    columns = [s.strip() for s in columns.split(",")]
    return df.drop(columns, axis=1)


@fn.NodeDecorator(
    node_id="pd.drop_rows",
    name="Drop Rows",
    description="Drops rows from a DataFrame.",
)
def drop_rows(
    df: pd.DataFrame,
    rows: str,
) -> pd.DataFrame:
    rows = [s.strip() for s in rows.split(",")]

    if len(df.index) == 0:
        return df
    cls = df.index.to_list()[0].__class__
    rows = [cls(row) for row in rows]

    return df.drop(rows, axis=0)


@fn.NodeDecorator(
    node_id="pd.reduce_df",
    name="Reduce DataFrame",
    description="Reduces the DataFrame by keeping rows where the specified columns change significantly.",
    outputs=[{"name": "reduced df", "type": pd.DataFrame}],
)
def reduce_df(
    df: pd.DataFrame,
    on: Union[str, List[str]],
    threshold: Optional[Union[float, List[float]]] = None,
    percentage_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Reduces the DataFrame by keeping rows where the specified columns change significantly.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    on (Union[str, List[str]]): Column(s) to monitor for significant changes.
                                If a single string is provided, it will be split by commas and whitespace stripped.
    threshold (Optional[Union[float, List[float]]]): Threshold for significant change.
                                Can be a single value or a list of values corresponding to `on`.
                                If None, the threshold is set to percentage_threshold of the min-max
                                range of each column.
    percentage_threshold (float): Percentage of the min-max range of each column to use as the threshold.
                                  Ignored if `threshold` is provided.

    Returns:
    pd.DataFrame: The reduced DataFrame.
    """
    # Handle case where `on` is a comma-separated string
    if isinstance(on, str):
        on = [c.strip() for c in on.split(",")]

    # Ensure `on` is a list of strings
    if not isinstance(on, list):
        on = [on]

    # Check for empty DataFrame or insufficient rows
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if len(df) == 1:
        return df

    # Ensure the columns specified in `on` are present and numeric
    for col in on:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"Column '{col}' must contain numeric data.")

    # Calculate default thresholds if not provided
    if threshold is None:
        ranges = np.array([df[col].max() - df[col].min() for col in on])
        # Prevent thresholds from being zero by using a small epsilon value
        threshold = percentage_threshold * np.where(
            ranges == 0, np.finfo(float).eps, ranges
        )

    # Ensure threshold is the correct length
    if not isinstance(threshold, (list, np.ndarray)):
        threshold = [threshold] * len(on)
    threshold = np.array(threshold)

    if len(threshold) != len(on):
        raise ValueError(
            "Threshold must be a single value or a list of values equal in length to 'on'."
        )

    # Vectorize difference calculation for faster processing

    values = df[on].values

    current_row = 0
    keep = [current_row]
    while True:
        # use values[0] since values will be reassign to reduce the aize each iteration
        maxval = values[0] + threshold
        minval = values[0] - threshold

        row_out_ranges = (values > maxval).any(axis=1) | (values < minval).any(axis=1)
        if row_out_ranges.any():
            # first out of range index
            first_out_row = row_out_ranges.argmax()

            if first_out_row > 1:
                # if it is larger than 1, we keep the previous row to capture the change e.g. if it is a step
                first_out_row -= 1
            elif first_out_row == 0:
                # if the first row out is 0 the first elemnte is true,
                # which should not happen since we compare against it
                raise ValueError(
                    "The first row is out of range, this should not happen"
                )

            # set the current row to the first out of range row
            current_row += first_out_row
            # reduce the values to the remaining rows
            values = values[first_out_row:]
            # add the current row to the keep list
            keep.append(current_row)
        else:
            # if no row is out of range, we can break
            break

    # Ensure the last row is included
    if len(df) - 1 not in keep:
        keep.append(len(df) - 1)

    return df.iloc[keep].reset_index(drop=True)


# endregion drop

# region add


@fn.NodeDecorator(
    node_id="pd.add_column",
    name="Add Column",
    description="Adds a column from a DataFrame.",
)
def add_column(
    column: str,
    data: Any,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if df is not None:
        df = df.copy()
        df[column] = data
    else:
        df = pd.DataFrame({column: data})

    return df


@fn.NodeDecorator(
    node_id="pd.add_row",
    name="Add Row",
    description="Adds a row to a DataFrame.",
)
def add_row(
    row: Union[dict, list, pd.Series],
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if isinstance(row, pd.Series):
        if df is None:
            return pd.DataFrame([row])
        elif len(row) != len(df.columns):
            raise ValueError(
                "Row must have the same number of columns as the DataFrame"
            )
        df = pd.concat([df, row.to_frame().T])
        return df
    if not isinstance(row, dict):
        try:
            row = {c: row[c] for c in df.columns}
        except Exception:
            pass
        if len(row) != len(df.columns):
            raise ValueError(
                "Row must have the same number of columns as the DataFrame"
            )
        row = {c: [v] for c, v in zip(df.columns, row)}
    if df is None:
        df = pd.DataFrame(row)
    else:
        df = pd.concat([df, pd.DataFrame(row)])
    return df


# endregion add


# region merge
@fn.NodeDecorator(
    node_id="pd.concat",
    name="Concatenate",
    description="Concatenates two DataFrames.",
)
def df_concatenate(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df1, df2])


@fn.NodeDecorator(
    node_id="pd.merge",
    name="Merge",
    description="Merges two DataFrames.",
    outputs=[{"name": "df", "type": pd.DataFrame}],
    default_io_options={
        "df_left": {
            "on": {
                "after_set_value": fn.decorator.update_other_io_options(
                    "left_on",
                    lambda x: list(x.columns),
                )
            }
        },
        "df_right": {
            "on": {
                "after_set_value": fn.decorator.update_other_io_options(
                    "right_on",
                    lambda x: list(x.columns),
                )
            }
        },
    },
)
def df_merge(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    how: Literal["inner", "outer", "left", "right"] = "inner",
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
) -> pd.DataFrame:
    return pd.merge(df_left, df_right, how=how, left_on=left_on, right_on=right_on)


@fn.NodeDecorator(
    node_id="pd.join",
    name="Join",
    description="Joins two DataFrames.",
    outputs=[{"name": "df", "type": pd.DataFrame}],
)
def df_join(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    how: Literal["inner", "outer", "left", "right"] = "left",
    on: Optional[str] = None,
    lsuffix: str = "",
    rsuffix: str = "",
) -> pd.DataFrame:
    return df_left.join(df_right, how=how, on=on, lsuffix=lsuffix, rsuffix=rsuffix)


# endregion merge


MANIPULATE_SHELF = fn.Shelf(
    nodes=[
        df_reset_index,
        dropna,
        fillna,
        bfill,
        ffill,
        drop_duplicates,
        numeric_only,
        DropColumnNode,
        DropRowNode,
        drop_columns,
        drop_rows,
        reduce_df,
        add_column,
        add_row,
        df_concatenate,
        df_merge,
        df_join,
    ],
    name="Manipulation",
    description="DataFrame manipulations",
    subshelves=[],
)
