from typing import List, Union, TypedDict
import exposedfunctionality.function_parser.types as exf_types
import funcnodes as fn


class DataFrameDict(TypedDict):
    columns: list[str]
    index: List[Union[str, int, float]]
    data: List[List[Union[str, int, float]]]


exf_types.add_type("DataFrameDict", DataFrameDict)


class SepEnum(fn.DataEnum):
    COMMA = ","
    SEMICOLON = ";"
    TAB = "\t"
    SPACE = " "
    PIPE = "|"

    def __str__(self):
        return str(self.value)


class DecimalEnum(fn.DataEnum):
    COMMA = ","
    DOT = "."

    def __str__(self):
        return str(self.value)


exf_types.add_type("pd.SepEnum", SepEnum)
exf_types.add_type("pd.DecimalEnum", DecimalEnum)
