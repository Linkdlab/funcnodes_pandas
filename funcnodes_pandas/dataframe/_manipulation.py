import pandas as pd
import funcnodes as fn
from typing import Optional, Literal, Union, Any
import numpy as np
# region nan


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
    value: Union[str, int, float] = 0,
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
                "after_set_value": fn.decorator.update_other_io(
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
                "after_set_value": fn.decorator.update_other_io(
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


# endregion drop

# region add


@fn.NodeDecorator(
    node_id="pd.add_column",
    name="Add Column",
    description="Adds a column from a DataFrame.",
)
def add_column(
    df: pd.DataFrame,
    column: str,
    data: Any,
) -> pd.DataFrame:
    df = df.copy()
    df[column] = data
    return df


@fn.NodeDecorator(
    node_id="pd.add_row",
    name="Add Row",
    description="Adds a row to a DataFrame.",
)
def add_row(
    df: pd.DataFrame,
    row: Union[dict, list],
) -> pd.DataFrame:
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
                "after_set_value": fn.decorator.update_other_io(
                    "left_on",
                    lambda x: list(x.columns),
                )
            }
        },
        "df_right": {
            "on": {
                "after_set_value": fn.decorator.update_other_io(
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
