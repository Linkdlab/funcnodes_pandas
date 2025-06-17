import funcnodes as fn

# import funcnodes_numpy to register the types
import funcnodes_numpy as fnnp  # noqa: F401
import pandas as pd
from exposedfunctionality.function_parser.types import type_to_string
from .dataframe import (
    display_df,
    NODE_SHELF as DF_SHELF,
    to_dict,
    from_dict,
    from_csv_str,
    from_csv_auto,
    GetColumnNode as get_column,
    SetColumnNode as set_column,
    get_column_names,
    get_columns_by_names,
    get_columns_by_index,
    to_orient_dict,
    from_orient_dict,
    df_iloc,
    df_ilocs,
    get_rows,
    GetRowNode as df_loc,
    SetRowNode as set_row,
    to_csv_str,
    df_from_array,
    DfFromExcelNode,
    df_to_xls,
    dropna,
    ffill,
    bfill,
    fillna,
    drop_duplicates,
    numeric_only,
    drop_columns,
    drop_rows,
    reduce_df,
    DropColumnNode as drop_column,
    DropRowNode as drop_row,
    add_column,
    add_row,
    df_concatenate,
    df_merge,
    filter,
    mask,
    df_join,
    # rows and columns
    df_rename_col,
    df_rename_cols_valid_identifier,
    # end rows and columns
    # math
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
    df_get_index,
    df_reset_index,
    # end math
)

from .dataseries import (
    ser_to_dict,
    ser_values,
    ser_to_list,
    ser_loc,
    ser_iloc,
    ser_from_dict,
    ser_from_list,
    NODE_SHELF as SERIES_SHELF,
    ser_str_contains,
    ser_str_startswith,
    ser_str_endswith,
    ser_str_count,
    ser_str_len,
    ser_str_extract,
    ser_str_find,
    ser_str_findall,
    ser_str_get,
    ser_str_replace,
    ser_str_slice,
    ser_str_split,
    ser_str_strip,
    ser_str_zfill,
    ser_str_pad,
    ser_str_center,
    ser_str_ljust,
    ser_str_rjust,
    ser_str_wrap,
    ser_str_repeat,
    ser_str_upper,
    ser_str_lower,
    ser_str_title,
    ser_str_capitalize,
    ser_str_swapcase,
    ser_str_cat,
)

from .grouping import (
    GroupByColumnNode as group_by_column,
    group_by,
    gr_mean,
    gr_sum,
    gr_max,
    gr_min,
    gr_std,
    gr_var,
    gr_count,
    gr_describe,
    group_to_list,
    GetDFfromGroupNode as get_df_from_group,
    NODE_SHELF as GROUPING_SHELF,
)


def encode_pdDf(obj, preview=False):
    if isinstance(obj, pd.DataFrame):
        if preview:
            return fn.Encdata(
                obj.head().to_dict(orient="split"),
                handeled=True,
                continue_preview=False,
            )
        else:
            return fn.Encdata(
                obj.to_dict(orient="split"),
                handeled=True,
                continue_preview=False,
            )
    if isinstance(obj, pd.Series):
        return fn.Encdata(
            obj.values,
            handeled=True,
        )
    return fn.Encdata(obj, handeled=False)


fn.JSONEncoder.add_encoder(encode_pdDf, [pd.DataFrame, pd.Series])


NODE_SHELF = fn.Shelf(
    nodes=[],
    subshelves=[DF_SHELF, SERIES_SHELF, GROUPING_SHELF],
    name="Pandas",
    description="Pandas nodes",
)

FUNCNODES_RENDER_OPTIONS: fn.RenderOptions = {
    "typemap": {
        type_to_string(pd.DataFrame): "table",
        type_to_string(pd.Series): "list",
    },
}

__version__ = "0.4.0"

__all__ = [
    "display_df",
    "NODE_SHELF",
    "to_dict",
    "from_dict",
    "from_csv_str",
    "from_csv_auto",
    "get_column",
    "to_orient_dict",
    "from_orient_dict",
    "df_iloc",
    "df_loc",
    "ser_to_dict",
    "ser_values",
    "ser_to_list",
    "ser_loc",
    "ser_iloc",
    "ser_from_dict",
    "ser_from_list",
    "SERIES_SHELF",
    "DF_SHELF",
    "to_csv_str",
    "df_from_array",
    "DfFromExcelNode",
    "df_to_xls",
    "dropna",
    "ffill",
    "bfill",
    "fillna",
    "drop_duplicates",
    "df_corr",
    "numeric_only",
    "drop_columns",
    "df_concatenate",
    "drop_rows",
    "drop_column",
    "drop_row",
    "reduce_df",
    "add_column",
    "add_row",
    "df_describe",
    "df_merge",
    "df_reset_index",
    "df_get_index",
    "filter",
    "set_column",
    "set_row",
    "get_column_names",
    "get_columns_by_names",
    "get_columns_by_index",
    "mask",
    "df_cov",
    "df_quantile",
    "df_value_counts",
    "df_eval",
    "df_corr",
    "df_join",
    "df_ilocs",
    "get_rows",
    # df
    # rows and columns
    "df_rename_col",
    "df_rename_cols_valid_identifier",
    # end rows and columns
    # math
    "df_corr",
    "df_cov",
    "df_mean",
    "df_median",
    "df_std",
    "df_sum",
    "df_var",
    "df_quantile",
    "df_describe",
    "df_value_counts",
    "df_eval",
    # end math
    # end df
    # series
    # str
    "ser_str_contains",
    "ser_str_startswith",
    "ser_str_endswith",
    "ser_str_count",
    "ser_str_len",
    "ser_str_extract",
    "ser_str_find",
    "ser_str_findall",
    "ser_str_get",
    "ser_str_replace",
    "ser_str_slice",
    "ser_str_split",
    "ser_str_strip",
    "ser_str_zfill",
    "ser_str_pad",
    "ser_str_center",
    "ser_str_ljust",
    "ser_str_rjust",
    "ser_str_wrap",
    "ser_str_repeat",
    "ser_str_upper",
    "ser_str_lower",
    "ser_str_title",
    "ser_str_capitalize",
    "ser_str_swapcase",
    "ser_str_cat",
    # end str
    # end series
    # grouping
    "group_by",
    "gr_mean",
    "gr_sum",
    "gr_max",
    "gr_min",
    "gr_std",
    "gr_var",
    "gr_count",
    "gr_describe",
    "group_to_list",
    "group_by_column",
    "get_df_from_group",
    # end grouping
]
