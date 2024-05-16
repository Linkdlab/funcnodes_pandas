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


class DfFromExcelNode(fn.Node):
    node_id = "pd.df_from_xlsx"
    node_name = "From Excel"

    data = fn.NodeInput(
        id="data",
        type=bytes,
    )
    sheet = fn.NodeInput(
        id="sheet",
        type=str,
        default=None,
        required=False,
    )

    df = fn.NodeOutput(id="df", type=pandas.DataFrame)

    async def func(self, data: bytes, sheet: str = None):
        # get sheet names
        sheets = pandas.ExcelFile(data).sheet_names
        self.inputs["sheet"].value_options = {s: s for s in sheets}
        if sheet is None or sheet not in sheets:
            sheet = sheets[0]
        self.inputs["sheet"].set_value(sheet, does_trigger=False)
        self.outputs["df"].value = pandas.read_excel(data, sheet_name=sheet)


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
            df = self.get_input("df").value
            col = self.get_input("column")
        except KeyError:
            return
        try:
            col.update_value_options(options=list(df.columns))
        except Exception:
            col.update_value_options(options=[])

    async def func(
        self,
        df: pandas.DataFrame,
        column: str,
    ) -> pandas.Series:
        self.get_output("series").value = df[column]
        return df[column]


class GetRowNode(fn.Node):
    node_id = "pd.df_loc"
    node_name = "Get Row"
    description = "Gets a row from a DataFrame by label."
    df = fn.NodeInput(
        "DataFrame",
        type=pandas.DataFrame,
        uuid="df",
    )

    row = fn.NodeInput(
        "Row",
        type=str,
        uuid="row",
    )

    series = fn.NodeOutput(
        "Series",
        type=pandas.Series,
        uuid="series",
    )

    def __init__(self):
        super().__init__()
        self.get_input("df").on("after_set_value", self._update_rows)

    def _update_rows(self, **kwargs):
        try:
            df = self.get_input("df").value
            row = self.get_input("row")
        except KeyError:
            return
        try:
            row.update_value_options(options=list(df.index))
        except Exception:
            row.update_value_options(options=[])

    async def func(
        self,
        df: pandas.DataFrame,
        row: str,
    ) -> pandas.Series:
        if len(df.index) == 0:
            return pandas.Series(index=df.columns)
        label = df.index.to_list()[0].__class__(row)
        ser = df.loc[label]
        self.get_output("series").value = ser
        return ser


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


@fn.NodeDecorator(
    node_id="pd.dropna",
    name="Drop NA",
    description="Drops rows or columns with NA values.",
)
def dropna(
    df: pandas.DataFrame,
    axis: Literal["index", "columns"] = "index",
    how: Literal["any", "all"] = "any",
) -> pandas.DataFrame:
    return df.dropna(axis=axis, how=how)


@fn.NodeDecorator(
    node_id="pd.fillna",
    name="Fill NA",
    description="Fills NA values with a specified value.",
)
def fillna(
    df: pandas.DataFrame,
    value: Union[str, int, float] = 0,
) -> pandas.DataFrame:
    return df.fillna(value)


@fn.NodeDecorator(
    node_id="pd.bfill",
    name="Backfill",
    description="Backfills NA values.",
)
def bfill(
    df: pandas.DataFrame,
) -> pandas.DataFrame:
    return df.bfill()


@fn.NodeDecorator(
    node_id="pd.ffill",
    name="Forwardfill",
    description="Forwardfills NA values.",
)
def ffill(
    df: pandas.DataFrame,
) -> pandas.DataFrame:
    return df.ffill()


@fn.NodeDecorator(
    node_id="pd.drop_duplicates",
    name="Drop Duplicates",
    description="Drops duplicate rows.",
)
def drop_duplicates(
    df: pandas.DataFrame,
) -> pandas.DataFrame:
    return df.drop_duplicates()


@fn.NodeDecorator(
    node_id="pd.corr",
    name="Correlation",
    description="Calculates the correlation between columns.",
    outputs=[{"name": "correlation", "type": pandas.DataFrame}],
)
def corr(
    df: pandas.DataFrame,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    numeric_only: bool = False,
) -> pandas.DataFrame:
    return df.corr(method=method, numeric_only=numeric_only)


@fn.NodeDecorator(
    node_id="pd.numeric_only",
    name="Numeric Only",
)
def numeric_only(df: pandas.DataFrame, label_encode=False) -> pandas.DataFrame:
    """
    Converts a DataFrame to only hold numeric values.
    Optionally, non-numeric values can be converted to numeric labels.

    Parameters:
    - df: pandas DataFrame
    - label_encode: bool, if True, convert non-numeric values to numeric labels

    Returns:
    - A new DataFrame containing only numeric values
    """

    if label_encode:
        df = df.copy()
        for column in df.select_dtypes(exclude=[np.number]):
            try:
                df[column] = pandas.to_numeric(df[column])
            except ValueError:
                pass
        for column in df.select_dtypes(include=["object", "category"]):
            df[column] = df[column].astype("category").cat.codes

    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df


class DropColumnNode(fn.Node):
    node_id = "pd.drop_column"
    node_name = "Drop Column"
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

    out = fn.NodeOutput(
        "New DataFrame",
        type=pandas.DataFrame,
        uuid="out",
    )

    def __init__(self):
        super().__init__()
        self.get_input("df").on("after_set_value", self._update_columns)

    def _update_columns(self, **kwargs):
        try:
            df = self.get_input("df").value
            col = self.get_input("column")
        except KeyError:
            return
        try:
            col.update_value_options(options=list(df.columns))
        except Exception:
            col.update_value_options(options=[])

    async def func(
        self,
        df: pandas.DataFrame,
        column: str,
    ) -> pandas.DataFrame:
        df = df.drop(column, axis=1)
        self.get_output("out").value = df
        return df


class DropRowNode(fn.Node):
    node_id = "pd.drop_row"
    node_name = "Drop Row"
    df = fn.NodeInput(
        "DataFrame",
        type=pandas.DataFrame,
        uuid="df",
    )

    row = fn.NodeInput(
        "Row",
        type=str,
        uuid="row",
    )

    out = fn.NodeOutput(
        "New DataFrame",
        type=pandas.DataFrame,
        uuid="out",
    )

    def __init__(self):
        super().__init__()
        self.get_input("df").on("after_set_value", self._update_rows)

    def _update_rows(self, **kwargs):
        try:
            df = self.get_input("df").value
            row = self.get_input("row")
        except KeyError:
            return
        try:
            row.update_value_options(options=list(df.index))
        except Exception:
            row.update_value_options(options=[])

    async def func(
        self,
        df: pandas.DataFrame,
        row: str,
    ) -> pandas.DataFrame:
        df = df.drop(row, axis=0)
        self.get_output("out").value = df
        return df


@fn.NodeDecorator(
    node_id="pd.drop_columns",
    name="Drop Columns",
    description="Drops columns from a DataFrame.",
)
def drop_columns(
    df: pandas.DataFrame,
    columns: str,
) -> pandas.DataFrame:
    columns = [s.strip() for s in columns.split(",")]
    return df.drop(columns, axis=1)


@fn.NodeDecorator(
    node_id="pd.drop_rows",
    name="Drop Rows",
    description="Drops rows from a DataFrame.",
)
def drop_rows(
    df: pandas.DataFrame,
    rows: str,
) -> pandas.DataFrame:
    rows = [s.strip() for s in rows.split(",")]

    if len(df.index) == 0:
        return df
    cls = df.index.to_list()[0].__class__
    rows = [cls(row) for row in rows]

    return df.drop(rows, axis=0)


NODE_SHELF = fn.Shelf(
    nodes=[
        to_dict,
        from_dict,
        from_csv_str,
        to_csv_str,
        GetColumnNode,
        to_orient_dict,
        from_orient_dict,
        GetRowNode,
        df_iloc,
        df_from_array,
        DfFromExcelNode,
        dropna,
        fillna,
        bfill,
        ffill,
        drop_duplicates,
        corr,
        numeric_only,
        DropColumnNode,
        DropRowNode,
        drop_columns,
        drop_rows,
    ],
    name="Datataframe",
    description="Pandas DataFrame nodes",
    subshelves=[],
)
