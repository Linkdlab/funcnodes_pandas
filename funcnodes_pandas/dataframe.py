from typing import TypedDict, List, Union, Literal
import funcnodes as fn
from funcnodes.triggerstack import TriggerStack
import pandas
import exposedfunctionality.function_parser.types as exf_types
import enum
from io import StringIO
import numpy as np


class DataFrameDict(TypedDict):
    columns: list[str]
    index: List[Union[str, int, float]]
    data: List[List[Union[str, int, float]]]


exf_types.add_type("DataFrameDict", DataFrameDict)


@fn.NodeDecorator(
    node_id="pd.df_to_dict",
    name="To Dictionary",
    description="Converts a DataFrame to a dictionary.",
    outputs=[{"name": "dict", "type": DataFrameDict}],
)
def to_dict(
    df: pandas.DataFrame,
) -> dict:
    return df.to_dict(orient="split")


@fn.NodeDecorator(
    node_id="pd.df_to_orient_dict",
    name="To Dictionary with Orientation",
    description="Converts a DataFrame to a dictionary with a specific orientation.",
    outputs=[{"name": "dict", "type": DataFrameDict}],
)
def to_orient_dict(
    df: pandas.DataFrame,
    orient: Literal["dict", "list", "split", "tight", "records", "index"] = "split",
) -> dict:
    return df.to_dict(orient=orient)


@fn.NodeDecorator(
    node_id="pd.df_from_dict",
    name="From Dictionary",
    description="Converts a dictionary to a DataFrame.",
    outputs=[{"name": "df", "type": pandas.DataFrame}],
)
def from_dict(
    data: dict,
) -> pandas.DataFrame:
    # from "split" orientation or from "thight" orientation
    if "columns" in data and "index" in data and "data" in data:
        df = pandas.DataFrame(
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
        return pandas.DataFrame(data)
    if len(data) == 0:
        return pandas.DataFrame()
    if isinstance(next(iter(data)), str):
        return pandas.DataFrame(data)
    else:
        return pandas.DataFrame(data).T


@fn.NodeDecorator(
    node_id="pd.df_from_orient_dict",
    name="From Dictionary with Orientation",
    description="Converts a dictionary with a specific orientation to a DataFrame.",
    outputs=[{"name": "df", "type": pandas.DataFrame}],
)
def from_orient_dict(
    data: dict,
    orient: Literal["dict", "list", "split", "tight", "records", "index"] = "split",
) -> pandas.DataFrame:
    if orient == "split":
        return pandas.DataFrame(
            data.get("data"), columns=data.get("columns"), index=data.get("index")
        )
    elif orient in ["dict", "list", "records"]:
        return pandas.DataFrame(data)
    elif orient == "tight":
        df = pandas.DataFrame(
            data.get("data"), columns=data.get("columns"), index=data.get("index")
        )
        df.columns.names = data.get("column_names")
        df.index.names = data.get("index_names")
        return df
    elif orient == "index":
        return pandas.DataFrame(data).T
    return pandas.DataFrame(data)


class SepEnum(enum.Enum):
    COMMA = ","
    SEMICOLON = ";"
    TAB = "\t"
    SPACE = " "
    PIPE = "|"

    def __str__(self):
        return str(self.value)


class DecimalEnum(enum.Enum):
    COMMA = ","
    DOT = "."

    def __str__(self):
        return str(self.value)


exf_types.add_type("pd.SepEnum", SepEnum)
exf_types.add_type("pd.DecimalEnum", DecimalEnum)


@fn.NodeDecorator(
    node_id="pd.df_from_csv_str",
    name="From CSV",
    description="Reads a CSV file into a DataFrame.",
    outputs=[{"name": "df", "type": pandas.DataFrame}],
)
def from_csv_str(
    source: str,
    sep: SepEnum = ",",
    decimal: DecimalEnum = ".",
) -> pandas.DataFrame:
    if "SepEnum." in sep:
        sep = sep.replace("SepEnum.", "")

    if "DecimalEnum." in decimal:
        decimal = decimal.replace("DecimalEnum.", "")
    # Check if sep is a string that matches an enum member's name, then get its value

    if isinstance(sep, str) and sep in SepEnum.__members__:
        sep = SepEnum[sep].value
    elif isinstance(sep, SepEnum):  # Direct instance of SepEnum
        sep = sep.value

    # Similar check and conversion for decimal
    if isinstance(decimal, str) and decimal in DecimalEnum.__members__:
        decimal = DecimalEnum[decimal].value
    elif isinstance(decimal, DecimalEnum):  # Direct instance of DecimalEnum
        decimal = decimal.value

    return pandas.read_csv(StringIO(source), sep=sep, decimal=decimal)


@fn.NodeDecorator(
    node_id="pd.df_to_csv_str",
    name="To CSV",
    description="Writes a DataFrame to a CSV string.",
    outputs=[{"name": "csv", "type": str}],
)
def to_csv_str(
    df: pandas.DataFrame,
    sep: SepEnum = ",",
    decimal: DecimalEnum = ".",
    index: bool = False,
) -> str:
    if "SepEnum." in sep:
        sep = sep.replace("SepEnum.", "")

    if "DecimalEnum." in decimal:
        decimal = decimal.replace("DecimalEnum.", "")
    # Check if sep is a string that matches an enum member's name, then get its value

    if isinstance(sep, str) and sep in SepEnum.__members__:
        sep = SepEnum[sep].value
    elif isinstance(sep, SepEnum):
        sep = sep.value

    # Similar check and conversion for decimal
    if isinstance(decimal, str) and decimal in DecimalEnum.__members__:
        decimal = DecimalEnum[decimal].value
    elif isinstance(decimal, DecimalEnum):
        decimal = decimal.value

    return df.to_csv(sep=sep, decimal=decimal, index=index)


class GetColumnNode(fn.Node):
    node_id = "pd.get_column"
    node_name = "Get Column"
    df = fn.NodeInput(
        "DataFrame",
        type=pandas.DataFrame,
        uuid="df",
    )

    column = fn.NodeInput(
        "Column",
        type=str,
        uuid="column",
    )

    series = fn.NodeOutput(
        "Series",
        type=pandas.Series,
        uuid="series",
    )

    def __init__(self):
        super().__init__()
        self.get_input("df").on("after_set_value", self._update_columns)

    def _update_columns(self, **kwargs):
        try:
            self.get_input("column").value_options = {
                c: c for c in self.get_input("df").value.columns
            }
        except Exception:
            self.get_input("column").value_options = {}

    async def func(
        self,
        df: pandas.DataFrame,
        column: str,
    ) -> pandas.Series:
        self.get_output("series").value = df[column]
        return df[column]


@fn.NodeDecorator(
    node_id="pd.df_loc",
    name="Get Row",
    description="Gets a row from a DataFrame by label.",
    outputs=[{"name": "row", "type": pandas.Series}],
)
def df_loc(
    df: pandas.DataFrame,
    label: Union[str],
) -> pandas.Series:
    # taransform label to the correct type
    label = df.index.to_list()[0].__class__(label)
    return df.loc[label]


@fn.NodeDecorator(
    node_id="pd.df_iloc",
    name="Get Row by Index",
    description="Gets a row from a DataFrame by index.",
    outputs=[{"name": "row", "type": pandas.Series}],
)
def df_iloc(
    df: pandas.DataFrame,
    index: Union[int],
) -> pandas.Series:
    return df.iloc[index]


@fn.NodeDecorator(
    node_id="pd.df_from_array",
    name="From Array",
    description="Creates a DataFrame from an array.",
    outputs=[{"name": "df", "type": pandas.DataFrame}],
)
def df_from_array(
    data: Union[list[list[Union[str, int, float]]], np.ndarray],
    columns: List[str] = None,
    index: List[Union[str, int, float]] = None,
) -> pandas.DataFrame:
    if columns is None:
        columns = [f"Col {i+1}" for i in range(len(data[0]))]
    return pandas.DataFrame(data, columns=columns, index=index)


NODE_SHELF = fn.Shelf(
    nodes=[
        to_dict,
        from_dict,
        from_csv_str,
        to_csv_str,
        GetColumnNode,
        to_orient_dict,
        from_orient_dict,
        df_loc,
        df_iloc,
        df_from_array,
    ],
    name="Datataframe",
    description="Pandas DataFrame nodes",
    subshelves=[],
)
