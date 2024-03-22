import funcnodes as fn

# import funcnodes_numpy to register the types
import funcnodes_numpy as fnnp  # noqa: F401
import pandas as pd
from exposedfunctionality.function_parser.types import type_to_string
from .dataframe import (
    NODE_SHELF as DF_SHELF,
    to_dict,
    from_dict,
    from_csv_str,
    GetColumnNode as get_column,
    to_orient_dict,
    from_orient_dict,
    df_iloc,
    df_loc,
    to_csv_str,
    df_from_array,
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
)


def encode_pdDf(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="split"), True
    if isinstance(obj, pd.Series):
        return obj.to_list(), True
    return obj, False


fn.JSONEncoder.add_encoder(encode_pdDf)


NODE_SHELF = fn.Shelf(
    nodes=[],
    subshelves=[DF_SHELF, SERIES_SHELF],
    name="Pandas",
    description="Pandas nodes",
)

FUNCNODES_RENDER_OPTIONS: fn.RenderOptions = {
    "typemap": {
        type_to_string(pd.DataFrame): "table",
        type_to_string(pd.Series): "list",
    },
}


__all__ = [
    "NODE_SHELF",
    "to_dict",
    "from_dict",
    "from_csv_str",
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
]
