import pandas as pd
import funcnodes as fn
from typing import Any, List
from ..utils import to_valid_identifier

# region cols


@fn.NodeDecorator(
    node_id="pd.get_column",
    name="Get Column",
    description="Gets a column from a DataFrame.",
    outputs=[{"name": "series", "type": pd.Series}],
    default_io_options={
        "df": {
            "on": {
                "after_set_value": fn.decorator.update_other_io_options(
                    "column",
                    lambda x: list(iter(x)),
                )
            }
        },
    },
)
def GetColumnNode(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column]


@fn.NodeDecorator(
    node_id="pd.set_column",
    name="Set Column",
    description="Sets a column in a DataFrame.",
    default_io_options={
        "df": {
            "on": {
                "after_set_value": fn.decorator.update_other_io_options(
                    "column",
                    lambda x: list(iter(x)),
                )
            }
        },
    },
)
def SetColumnNode(df: pd.DataFrame, column: str, data: Any) -> pd.DataFrame:
    df = df.copy()
    df[column] = data
    return df


@fn.NodeDecorator(
    node_id="pd.get_columns",
    name="Get Columns",
    description="Get the names of the columns in a DataFrame.",
    outputs=[{"name": "columns"}],
)
def get_column_names(
    df: pd.DataFrame,
) -> List[str]:
    """
    Gets multiple columns from a DataFrame.
    """
    return list(df.columns)


@fn.NodeDecorator(
    node_id="pd.get_columns_by_names",
    name="Get Columns by Names",
    description="Gets multiple columns from a DataFrame by names.",
    outputs=[{"name": "subdf", "type": pd.DataFrame}],
)
def get_columns_by_names(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Gets multiple columns from a DataFrame by names.
    """
    if isinstance(columns, str):
        columns = columns.split(",")
    columns = [col.strip() for col in columns]
    return df[columns]


@fn.NodeDecorator(
    node_id="pd.get_columns_by_index",
    name="Gets a Columns by its Index",
    description="Gets a column from a DataFrame by its index.",
    outputs=[{"name": "series"}],
    default_io_options={
        "df": {
            "on": {
                "after_set_value": fn.decorator.update_other_io_value_options(
                    "index",
                    lambda result: dict(min=0, max=len(result.columns) - 1, step=1),
                )
            }
        },
    },
)
def get_columns_by_index(
    df: pd.DataFrame,
    index: int = 0,
) -> pd.Series:
    """
    Gets a column from a DataFrame by its index.
    """
    if index < 0 or index >= len(df.columns):
        raise IndexError("Index out of bounds for DataFrame columns.")
    return df[df.columns[int(index)]].copy()


# endregion cols


# region rows


@fn.NodeDecorator(
    node_id="pd.get_row",
    name="Get Row",
    description="Gets a row from a DataFrame by label.",
    outputs=[{"name": "series", "type": pd.Series}],
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
def GetRowNode(df: pd.DataFrame, row: str) -> pd.Series:
    return df.loc[df.index.to_list()[0].__class__(row)]  # transform to the correct type


@fn.NodeDecorator(
    node_id="pd.get_rows",
    name="Get Rows",
    description="Gets rows from a DataFrame by label.",
)
def get_rows(
    df: pd.DataFrame,
    rows: List[Any],
) -> pd.DataFrame:
    rows = [df.index.to_list()[0].__class__(row) for row in rows]
    return df.loc[rows]


@fn.NodeDecorator(
    node_id="pd.set_row",
    name="Set Row",
    description="Sets a row in a DataFrame.",
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
def SetRowNode(df: pd.DataFrame, row: str, data: Any) -> pd.DataFrame:
    df = df.copy()
    df.loc[df.index.to_list()[0].__class__(row)] = data
    return df


@fn.NodeDecorator(
    node_id="pd.df_iloc",
    name="Get Row by Index",
    description="Gets a row from a DataFrame by index.",
    outputs=[{"name": "row", "type": pd.Series}],
    default_io_options={
        "df": {
            "on": {
                "after_set_value": fn.decorator.update_other_io_value_options(
                    "index", lambda result: dict(min=0, max=len(result) - 1, step=1)
                )
            }
        },
    },
)
def df_iloc(
    df: pd.DataFrame,
    index: int = 0,
) -> pd.Series:
    return df.iloc[int(index)]


@fn.NodeDecorator(
    node_id="pd.df_locs",
    name="Get Rows by Indices",
    description="Gets rows from a DataFrame by indices.",
    outputs=[{"name": "rows", "type": pd.DataFrame}],
)
def df_ilocs(
    df: pd.DataFrame,
    indices: List[int],
) -> pd.DataFrame:
    return df.iloc[[int(i) for i in indices]]


# endregion rows


@fn.NodeDecorator(
    node_id="pd.df_rename_col",
    name="Rename Column",
    description="Renames a column in a DataFrame.",
    default_io_options={
        "df": {
            "on": {
                "after_set_value": fn.decorator.update_other_io_options(
                    "old_name",
                    lambda x: x.columns.to_list(),
                )
            }
        },
    },
)
def df_rename_col(
    df: pd.DataFrame,
    old_name: str,
    new_name: str,
) -> pd.DataFrame:
    return df.rename(columns={old_name: new_name})


@fn.NodeDecorator(
    "pd.df_rename_cols_valid_identifier",
    name="Rename Columns to Valid Identifiers",
    description="Renames columns in a DataFrame to valid identifiers.",
)
def df_rename_cols_valid_identifier(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return df.rename(
        columns={
            col: to_valid_identifier(
                col,
            )
            for col in df.columns
        }
    )


@fn.NodeDecorator(
    node_id="pd.df_get_index",
    name="Get Index",
    description="Gets the index of a DataFrame as a Series.",
    outputs=[{"name": "index"}],
)
def df_get_index(df: pd.DataFrame) -> pd.Series:
    return df.index.to_series()


ROW_COLS_SHELF = fn.Shelf(
    nodes=[
        GetColumnNode,
        SetColumnNode,
        GetRowNode,
        SetRowNode,
        get_column_names,
        get_columns_by_names,
        get_columns_by_index,
        df_iloc,
        get_rows,
        df_ilocs,
        df_rename_col,
        df_rename_cols_valid_identifier,
        df_get_index,
    ],
    name="Rows and Columns",
    description="OPeration on rows and columns",
    subshelves=[],
)
