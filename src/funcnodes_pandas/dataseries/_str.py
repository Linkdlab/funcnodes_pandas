"""
string operations for pandas Series
"""

import pandas as pd
import funcnodes as fn
from typing import Any, List, Optional, Literal

# region filter


@fn.NodeDecorator(
    node_id="pd.str_contains",
    name="Contains",
    description="Checks if a string is contained in each element of a Series.",
)
def ser_str_contains(series: pd.Series, pat: str) -> pd.Series:
    return series.str.contains(pat)


@fn.NodeDecorator(
    node_id="pd.str_startswith",
    name="Starts With",
    description="Checks if each element of a Series starts with a string.",
)
def ser_str_startswith(series: pd.Series, pat: str) -> pd.Series:
    return series.str.startswith(pat)


@fn.NodeDecorator(
    node_id="pd.str_endswith",
    name="Ends With",
    description="Checks if each element of a Series ends with a string.",
)
def ser_str_endswith(series: pd.Series, pat: str) -> pd.Series:
    return series.str.endswith(pat)


# endregion filter

# region stats


@fn.NodeDecorator(
    node_id="pd.str_count",
    name="Count",
    description="Counts occurrences of a pattern in each element of a Series.",
)
def ser_str_count(series: pd.Series, pat: str) -> pd.Series:
    return series.str.count(pat)


@fn.NodeDecorator(
    node_id="pd.str_len",
    name="Length",
    description="Calculates the length of each element of a Series.",
)
def ser_str_len(series: pd.Series) -> pd.Series:
    return series.str.len()


# endregion stats


# region substring


@fn.NodeDecorator(
    node_id="pd.str_extract",
    name="Extract",
    description="Extracts a pattern from each element of a Series.",
)
def ser_str_extract(series: pd.Series, pat: str) -> pd.DataFrame:
    return series.str.extract(pat, expand=True)


@fn.NodeDecorator(
    node_id="pd.str_find",
    name="Find",
    description="Finds the first occurrence of a pattern in each element of a Series.",
)
def ser_str_find(series: pd.Series, pat: str) -> pd.Series:
    return series.str.find(pat)


@fn.NodeDecorator(
    node_id="pd.str_findall",
    name="Find All",
    description="Finds all occurrences of a pattern in each element of a Series.",
)
def ser_str_findall(series: pd.Series, pat: str) -> pd.Series:
    return series.str.findall(pat)


@fn.NodeDecorator(
    node_id="pd.str_get",
    name="Get",
    description="Gets the nth character of each element of a Series.",
)
def ser_str_get(series: pd.Series, index: int) -> pd.Series:
    return series.str.get(index)


# endregion substring

# region transform


@fn.NodeDecorator(
    node_id="pd.str_replace",
    name="Replace",
    description="Replaces a pattern in each element of a Series.",
)
def ser_str_replace(series: pd.Series, pat: str, repl: str) -> pd.Series:
    return series.str.replace(pat, repl)


@fn.NodeDecorator(
    node_id="pd.str_slice",
    name="Slice",
    description="Slices each element of a Series.",
)
def ser_str_slice(series: pd.Series, start: int, stop: int, step: int = 1) -> pd.Series:
    return series.str.slice(start, stop, step)


@fn.NodeDecorator(
    node_id="pd.str_split",
    name="Split",
    description="Splits each element of a Series.",
)
def ser_str_split(series: pd.Series, pat: str) -> pd.Series:
    return series.str.split(pat)


@fn.NodeDecorator(
    node_id="pd.str_strip",
    name="Strip",
    description="Strips whitespace from each element of a Series.",
)
def ser_str_strip(series: pd.Series) -> pd.Series:
    return series.str.strip()


@fn.NodeDecorator(
    node_id="pd.str_zfill",
    name="Zero Fill",
    description="Fills each element of a Series with zeros.",
)
def ser_str_zfill(series: pd.Series, width: int) -> pd.Series:
    return series.str.zfill(width)


@fn.NodeDecorator(
    node_id="pd.str_pad",
    name="Pad",
    description="Pads each element of a Series.",
)
def ser_str_pad(
    series: pd.Series,
    width: int,
    side: Literal["left", "right", "both"] = "left",
    fillchar: str = " ",
) -> pd.Series:
    return series.str.pad(width, side, fillchar)


@fn.NodeDecorator(
    node_id="pd.str_center",
    name="Center",
    description="Centers each element of a Series.",
)
def ser_str_center(series: pd.Series, width: int, fillchar: str = " ") -> pd.Series:
    return series.str.center(width, fillchar)


@fn.NodeDecorator(
    node_id="pd.str_ljust",
    name="Left Justify",
    description="Left justifies each element of a Series.",
)
def ser_str_ljust(series: pd.Series, width: int, fillchar: str = " ") -> pd.Series:
    return series.str.ljust(width, fillchar)


@fn.NodeDecorator(
    node_id="pd.str_rjust",
    name="Right Justify",
    description="Right justifies each element of a Series.",
)
def ser_str_rjust(series: pd.Series, width: int, fillchar: str = " ") -> pd.Series:
    return series.str.rjust(width, fillchar)


@fn.NodeDecorator(
    node_id="pd.str_wrap",
    name="Wrap",
    description="Wraps each element of a Series.",
)
def ser_str_wrap(series: pd.Series, width: int) -> pd.Series:
    return series.str.wrap(width)


@fn.NodeDecorator(
    node_id="pd.str_repeat",
    name="Repeat",
    description="Repeats each element of a Series.",
)
def ser_str_repeat(series: pd.Series, repeats: int) -> pd.Series:
    return series.str.repeat(repeats)


@fn.NodeDecorator(
    node_id="pd.str_upper",
    name="Upper",
    description="Converts each element of a Series to uppercase.",
)
def ser_str_upper(series: pd.Series) -> pd.Series:
    return series.str.upper()


@fn.NodeDecorator(
    node_id="pd.str_lower",
    name="Lower",
    description="Converts each element of a Series to lowercase.",
)
def ser_str_lower(series: pd.Series) -> pd.Series:
    return series.str.lower()


@fn.NodeDecorator(
    node_id="pd.str_title",
    name="Title",
    description="Converts each element of a Series to title case.",
)
def ser_str_title(series: pd.Series) -> pd.Series:
    return series.str.title()


@fn.NodeDecorator(
    node_id="pd.str_capitalize",
    name="Capitalize",
    description="Capitalizes each element of a Series.",
)
def ser_str_capitalize(series: pd.Series) -> pd.Series:
    return series.str.capitalize()


@fn.NodeDecorator(
    node_id="pd.str_swapcase",
    name="Swap Case",
    description="Swaps the case of each element of a Series.",
)
def ser_str_swapcase(series: pd.Series) -> pd.Series:
    return series.str.swapcase()


@fn.NodeDecorator(
    node_id="pd.str_cat",
    name="Concatenate",
    description="Concatenates strings in a Series.",
    default_io_options={
        "series": {
            "on": {
                "after_set_value": fn.decorator.update_other_io_options(
                    "others",
                    lambda x: list(iter(x)),
                )
            }
        },
    },
)
def ser_str_cat(
    series: pd.Series, sep: Optional[str] = None, others: Optional[List[Any]] = None
) -> str:
    return series.str.cat(
        others,
        sep=sep,
    )


# endregion transform


SERIES_STR_SHELF = fn.Shelf(
    name="String Operations",
    description="String operations for pandas Series.",
    nodes=[
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
    ],
    subshelves=[],
)
