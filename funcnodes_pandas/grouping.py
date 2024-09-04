import pandas as pd
import funcnodes as fn
from pandas.api.typing import DataFrameGroupBy
from typing import List


@fn.NodeDecorator(
    node_id="pd.groupby_column",
    name="Group By Column",
    description="Groups a DataFrame by a column.",
    outputs=[{"name": "group"}],
    default_io_options={
        "df": {
            "on": {
                "after_set_value": fn.decorator.update_other_io(
                    "column",
                    lambda x: list(x.columns),
                )
            }
        },
    },
)
def GroupByColumnNode(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    df = df.copy()
    return df.groupby(column)


@fn.NodeDecorator(
    node_id="pd.groupby",
    name="Group By",
    description="Groups a DataFrame by a column.",
    outputs=[{"name": "grouped", "type": pd.DataFrame}],
)
def group_by(
    df: pd.DataFrame,
    by: str,
) -> DataFrameGroupBy:
    sep = [s.strip() for s in by.split(",")]
    df = df.copy()
    return df.groupby(sep)


@fn.NodeDecorator(
    node_id="pd.mean",
    name="Mean",
    description="Calculates the mean of a DataFrameGroup.",
    outputs=[{"name": "mean", "type": pd.DataFrame}],
)
def mean(
    group: DataFrameGroupBy,
) -> pd.DataFrame:
    return group.mean()


@fn.NodeDecorator(
    node_id="pd.sum",
    name="Sum",
    description="Calculates the sum of a DataFrameGroup.",
    outputs=[{"name": "sum", "type": pd.DataFrame}],
)
def sum(
    group: DataFrameGroupBy,
) -> pd.DataFrame:
    return group.sum()


@fn.NodeDecorator(
    node_id="pd.max",
    name="Max",
    description="Calculates the max of a DataFrameGroup.",
    outputs=[{"name": "max", "type": pd.DataFrame}],
)
def max(
    group: DataFrameGroupBy,
) -> pd.DataFrame:
    return group.max()


@fn.NodeDecorator(
    node_id="pd.min",
    name="Min",
    description="Calculates the min of a DataFrameGroup.",
    outputs=[{"name": "min", "type": pd.DataFrame}],
)
def min(
    group: DataFrameGroupBy,
) -> pd.DataFrame:
    return group.min()


@fn.NodeDecorator(
    node_id="pd.std",
    name="Standard Deviation",
    description="Calculates the standard deviation of a DataFrameGroup.",
    outputs=[{"name": "std", "type": pd.DataFrame}],
)
def std(
    group: DataFrameGroupBy,
) -> pd.DataFrame:
    return group.std()


@fn.NodeDecorator(
    node_id="pd.var",
    name="Variance",
    description="Calculates the variance of a DataFrameGroup.",
    outputs=[{"name": "var", "type": pd.DataFrame}],
)
def var(
    group: DataFrameGroupBy,
) -> pd.DataFrame:
    return group.var()


@fn.NodeDecorator(
    node_id="pd.count",
    name="Count",
    description="Calculates the count of a DataFrameGroup.",
    outputs=[{"name": "count", "type": pd.DataFrame}],
)
def count(
    group: DataFrameGroupBy,
) -> pd.DataFrame:
    return group.count()


@fn.NodeDecorator(
    node_id="pd.describe",
    name="Describe",
    description="Describes a DataFrameGroup.",
    outputs=[
        {
            "name": "description",
        }
    ],
)
def describe(
    group: DataFrameGroupBy,
) -> pd.DataFrame:
    return group.describe()


@fn.NodeDecorator(
    node_id="pd.group_to_list",
    name="Group to List",
    description="Converts a DataFrameGroup to a list of DataFrames.",
    outputs=[{"name": "list", "type": list}],
)
def group_to_list(
    group: DataFrameGroupBy,
) -> List[pd.DataFrame]:
    return [group for _, group in group]


@fn.NodeDecorator(
    node_id="pd.get_df_from_group",
    name="Get DataFrame from Group",
    description="Gets a DataFrame from a DataFrameGroup.",
    outputs=[{"name": "df"}],
    default_io_options={
        "group": {
            "on": {
                "after_set_value": fn.decorator.update_other_io(
                    "name",
                    lambda x: list(x.groups.keys()),
                )
            }
        },
    },
)
def GetDFfromGroupNode(
    group: DataFrameGroupBy,
    name: str,
) -> pd.DataFrame:
    df = group.get_group(name).copy()
    return df


NODE_SHELF = fn.Shelf(
    nodes=[
        GroupByColumnNode,
        group_by,
        mean,
        sum,
        max,
        min,
        std,
        var,
        count,
        describe,
        group_to_list,
        GetDFfromGroupNode,
    ],
    name="Grouping",
    description="Pandas grouping nodes",
    subshelves=[],
)
