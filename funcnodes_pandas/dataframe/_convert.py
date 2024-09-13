import pandas as pd
import funcnodes as fn
from typing import Optional, Literal, Union, List
from ._types import DataFrameDict, SepEnum, DecimalEnum
from io import StringIO, BytesIO
import numpy as np

# region dict


@fn.NodeDecorator(
    node_id="pd.df_to_dict",
    name="To Dictionary",
    description="Converts a DataFrame to a dictionary.",
    outputs=[{"name": "dict", "type": DataFrameDict}],
)
def to_dict(
    df: pd.DataFrame,
) -> dict:
    return df.to_dict(orient="split")


@fn.NodeDecorator(
    node_id="pd.df_to_orient_dict",
    name="To Dictionary with Orientation",
    description="Converts a DataFrame to a dictionary with a specific orientation.",
    outputs=[{"name": "dict", "type": dict}],
)
def to_orient_dict(
    df: pd.DataFrame,
    orient: Literal["dict", "list", "split", "tight", "records", "index"] = "split",
) -> dict:
    return df.to_dict(orient=orient)


@fn.NodeDecorator(
    node_id="pd.df_from_dict",
    name="From Dictionary",
    description="Converts a dictionary to a DataFrame.",
    outputs=[{"name": "df", "type": pd.DataFrame}],
)
def from_dict(
    data: dict,
) -> pd.DataFrame:
    # from "split" orientation or from "thight" orientation
    if "columns" in data and "index" in data and "data" in data:
        df = pd.DataFrame(
            data["data"],
            columns=data["columns"],
            index=data["index"],
        )
        idxnames = data.get("index_names")
        if idxnames is not None and len(idxnames) == len(df.index):
            df.index.names = idxnames
        colnames = data.get("column_names")
        if colnames is not None and len(colnames) == len(df.columns):
            df.columns.names = colnames
        return df

    # by default we cannot distringuise between "dict" and "index" orientation since both have the same structure of
    # {column: {index: value}} or {index: {column: value}}
    # a small heuristic is to check if the first key is a string or not to determine the orientation
    if isinstance(data, list):
        return pd.DataFrame(data)
    if len(data) == 0:
        return pd.DataFrame()
    if isinstance(next(iter(data)), str):
        return pd.DataFrame(data)
    else:
        return pd.DataFrame(data).T


@fn.NodeDecorator(
    node_id="pd.df_from_orient_dict",
    name="From Dictionary with Orientation",
    description="Converts a dictionary with a specific orientation to a DataFrame.",
    outputs=[{"name": "df", "type": pd.DataFrame}],
)
def from_orient_dict(
    data: dict,
    orient: Literal["dict", "list", "split", "tight", "records", "index"] = "split",
) -> pd.DataFrame:
    if orient == "split":
        return pd.DataFrame(
            data.get("data"), columns=data.get("columns"), index=data.get("index")
        )
    elif orient in ["dict", "list", "records"]:
        return pd.DataFrame(data)
    elif orient == "tight":
        df = pd.DataFrame(
            data.get("data"), columns=data.get("columns"), index=data.get("index")
        )
        df.columns.names = data.get("column_names")
        df.index.names = data.get("index_names")
        return df
    elif orient == "index":
        return pd.DataFrame(data).T
    return pd.DataFrame(data)


# endregion dict


# region csv


@fn.NodeDecorator(
    node_id="pd.df_from_csv_str",
    name="From CSV",
    description="Reads a CSV file into a DataFrame.",
    outputs=[{"name": "df", "type": pd.DataFrame}],
)
def from_csv_str(
    source: str,
    sep: SepEnum = ",",
    decimal: DecimalEnum = ".",
    thousands: Optional[DecimalEnum] = None,
) -> pd.DataFrame:
    sep = SepEnum.v(sep)
    decimal = DecimalEnum.v(decimal)
    thousands = DecimalEnum.v(thousands) if thousands is not None else None

    return pd.read_csv(StringIO(source), sep=sep, decimal=decimal, thousands=thousands)


@fn.NodeDecorator(
    node_id="pd.df_to_csv_str",
    name="To CSV",
    description="Writes a DataFrame to a CSV string.",
    outputs=[{"name": "csv"}],
)
def to_csv_str(
    df: pd.DataFrame,
    sep: SepEnum = ",",
    decimal: DecimalEnum = ".",
    thousands: Optional[DecimalEnum] = None,
    index: bool = False,
) -> str:
    sep = SepEnum.v(sep)
    decimal = DecimalEnum.v(decimal)
    thousands = DecimalEnum.v(thousands) if thousands is not None else None

    return df.to_csv(sep=sep, decimal=decimal, index=index)


# endregion csv

# region excel


@fn.NodeDecorator(
    node_id="pd.df_from_xlsx",
    name="From Excel",
    description="Reads an Excel file into a DataFrame.",
    outputs=[{"name": "df", "type": pd.DataFrame}],
    default_io_options={
        "data": {
            "on": {
                "after_set_value": fn.decorator.update_other_io(
                    "sheet",
                    lambda x: pd.ExcelFile(BytesIO(x)).sheet_names,
                )
            }
        }
    },
)
def DfFromExcelNode(data: bytes, sheet: Optional[str] = None, with_index: bool = False):
    # if sheed is not provided, we return the first sheet
    if sheet is None:
        sheet = 0
    return pd.read_excel(
        BytesIO(data), sheet_name=sheet, index_col=0 if with_index else None
    )


@fn.NodeDecorator(
    node_id="pd.df_to_xls",
    name="To Excel",
    description="Writes a DataFrame to an Excel file.",
    outputs=[{"name": "xls"}],
)
def df_to_xls(
    df: pd.DataFrame,
    sheet_name: str = "Sheet1",
    with_index: bool = False,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=with_index)
    return output.getvalue()


# endregion excel


# region array


@fn.NodeDecorator(
    node_id="pd.df_from_array",
    name="From Array",
    description="Creates a DataFrame from an array.",
    outputs=[{"name": "df", "type": pd.DataFrame}],
)
def df_from_array(
    data: Union[list[list[Union[str, int, float]]], np.ndarray],
    columns: List[str] = None,
    index: List[Union[str, int, float]] = None,
) -> pd.DataFrame:
    if columns is None:
        columns = [f"Col {i+1}" for i in range(len(data[0]))]
    return pd.DataFrame(data, columns=columns, index=index)


# endregion array


CONVERT_SHELF = fn.Shelf(
    nodes=[
        to_dict,
        from_dict,
        from_csv_str,
        to_csv_str,
        to_orient_dict,
        from_orient_dict,
        df_from_array,
        DfFromExcelNode,
        df_to_xls,
    ],
    name="Convert",
    description="Conversions for DataFrames",
    subshelves=[],
)
