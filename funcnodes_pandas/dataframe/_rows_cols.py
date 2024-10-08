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
                "after_set_value": fn.decorator.update_other_io(
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
                "after_set_value": fn.decorator.update_other_io(
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
                "after_set_value": fn.decorator.update_other_io(
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
                "after_set_value": fn.decorator.update_other_io(
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
                "after_set_value": lambda src, result: src.node[
                    "index"
                ].update_value_options(min=0, max=len(result) - 1, step=1)
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
                "after_set_value": fn.decorator.update_other_io(
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


ROW_COLS_SHELF = fn.Shelf(
    nodes=[
        GetColumnNode,
        SetColumnNode,
        GetRowNode,
        SetRowNode,
        df_iloc,
        get_rows,
        df_ilocs,
        df_rename_col,
        df_rename_cols_valid_identifier,
    ],
    name="Rows and Columns",
    description="OPeration on rows and columns",
    subshelves=[],
)
