import funcnodes as fn


from ._convert import (  # noqa: F401
    to_dict,
    from_dict,
    from_csv_str,
    to_csv_str,
    to_orient_dict,
    from_orient_dict,
    df_from_array,
    DfFromExcelNode,
    df_to_xls,
    CONVERT_SHELF,
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
    df_iloc,
    get_rows,
    df_ilocs,
    df_rename_col,
    df_rename_cols_valid_identifier,
    ROW_COLS_SHELF,
)
from ._masking import (  # noqa: F401
    filter,
    mask,
    MASK_SHELF,
)

NODE_SHELF = fn.Shelf(
    nodes=[],
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
