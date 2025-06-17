import funcnodes as fn


from ._convert import (  # noqa: F401
    to_dict,
    from_dict,
    from_csv_str,
    from_csv_auto,
    to_csv_str,
    to_orient_dict,
    from_orient_dict,
    df_from_array,
    DfFromExcelNode,
    df_to_xls,
    CONVERT_SHELF,
    pd,
)
from ._manipulation import (  # noqa: F401
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
    df_reset_index,
    MANIPULATE_SHELF,
)
from ._math import (  # noqa: F401
    df_corr,
    df_cov,
    df_mean,
    df_median,
    df_std,
    df_sum,
    df_var,
    df_quantile,
    df_describe,
    df_value_counts,
    df_eval,
    MATH_SHELF,
)

from ._rows_cols import (  # noqa: F401
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
    ROW_COLS_SHELF,
    df_get_index,
)
from ._masking import (  # noqa: F401
    filter,
    mask,
    MASK_SHELF,
)


@fn.NodeDecorator(
    node_id="pd.displaydf",
    name="Display",
    description="Helper node to display a DataFrame.",
    default_render_options={"data": {"src": "df", "type": "table"}},
)
def display_df(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected a DataFrame, got {type(df)}")


NODE_SHELF = fn.Shelf(
    nodes=[display_df],
    name="Datataframe",
    description="Pandas DataFrame nodes",
    subshelves=[
        CONVERT_SHELF,
        MANIPULATE_SHELF,
        ROW_COLS_SHELF,
        MASK_SHELF,
        MATH_SHELF,
    ],
)
