import pandas as pd
import funcnodes as fn
from typing import List


@fn.NodeDecorator(
    node_id="pd.filter",
    name="Filter",
    description="Filters a DataFrame based on a condition.",
    outputs=[{"name": "filtered", "type": pd.DataFrame}],
)
def filter(
    df: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    return df.query(condition)


@fn.NodeDecorator(
    node_id="pd.mask",
    name="Mask",
    description="Masks a DataFrame based on a condition.",
    outputs=[{"name": "masked", "type": pd.DataFrame}],
)
def mask(
    df: pd.DataFrame,
    mask: List[bool],
) -> pd.DataFrame:
    return df[mask]


MASK_SHELF = fn.Shelf(
    nodes=[
        filter,
        mask,
    ],
    name="Masking",
    description="Masking operations on DataFrames",
    subshelves=[],
)
