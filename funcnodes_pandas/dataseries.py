from typing import TypedDict, List, Union, Literal, Any, Optional
import funcnodes as fn
from funcnodes.triggerstack import TriggerStack
import numpy as np
import pandas
import exposedfunctionality.function_parser.types as exf_types
import enum


@fn.NodeDecorator(
    node_id="pd.ser_to_dict",
    name="To Dictionary",
    description="Converts a Series to a dictionary.",
    outputs=[{"name": "dict", "type": dict}],
)
def ser_to_dict(
    ser: pandas.Series,
) -> dict:
    return ser.to_dict()


@fn.NodeDecorator(
    node_id="pd.ser_values",
    name="Get Values",
    description="Gets the values of a Series.",
    outputs=[{"name": "values", "type": np.ndarray}],
)
def ser_values(
    ser: pandas.Series,
) -> np.ndarray:
    return ser.to_numpy(copy=True)


@fn.NodeDecorator(
    node_id="pd.ser_to_list",
    name="To List",
    description="Converts a Series to a list.",
    outputs=[{"name": "list", "type": list}],
)
def ser_to_list(
    ser: pandas.Series,
) -> list:
    return ser.to_list()


@fn.NodeDecorator(
    node_id="pd.ser_loc",
    name="Get Value",
    description="Gets a value from a Series by label.",
    outputs=[{"name": "value", "type": Any}],
)
def ser_loc(
    ser: pandas.Series,
    label: Union[str],
) -> Union[str]:
    # taransform label to the correct type
    label = ser.index.to_list()[0].__class__(label)
    return ser.loc[label]


@fn.NodeDecorator(
    node_id="pd.ser_iloc",
    name="Get Value by Index",
    description="Gets a value from a Series by index.",
    outputs=[{"name": "value", "type": Any}],
)
def ser_iloc(
    ser: pandas.Series,
    index: int,
) -> Union[str]:
    return ser.iloc[index]


@fn.NodeDecorator(
    node_id="pd.ser_from_dict",
    name="From Dictionary",
    description="Creates a Series from a dictionary.",
    outputs=[{"name": "series", "type": pandas.Series}],
)
def ser_from_dict(
    data: dict,
    name: Optional[str] = None,
) -> pandas.Series:
    return pandas.Series(data, name=name)


@fn.NodeDecorator(
    node_id="pd.ser_from_list",
    name="From List",
    description="Creates a Series from a list.",
    outputs=[{"name": "series", "type": pandas.Series}],
)
def ser_from_list(
    data: list,
    name: Optional[str] = None,
) -> pandas.Series:
    return pandas.Series(data, name=name)


NODE_SHELF = fn.Shelf(
    name="Series",
    nodes=[
        ser_to_dict,
        ser_values,
        ser_to_list,
        ser_loc,
        ser_iloc,
        ser_from_dict,
        ser_from_list,
    ],
    description="Pandas Series nodes",
    subshelves=[],
)
