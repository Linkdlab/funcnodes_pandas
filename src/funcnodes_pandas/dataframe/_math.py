import pandas as pd
import funcnodes as fn
from typing import Literal, List, Optional, Union

from ..utils import to_valid_identifier


@fn.NodeDecorator(
    node_id="pd.corr",
    name="Correlation",
    description="Calculates the correlation between columns.",
    outputs=[{"name": "correlation", "type": pd.DataFrame}],
)
def df_corr(
    df: pd.DataFrame,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    numeric_only: bool = False,
) -> pd.DataFrame:
    return df.corr(method=method, numeric_only=numeric_only)


@fn.NodeDecorator(
    node_id="pd.cov",
    name="Covariance",
    description="Calculates the covariance between columns.",
    outputs=[{"name": "covariance", "type": pd.DataFrame}],
)
def df_cov(
    df: pd.DataFrame,
    min_periods: int = 1,
    ddof: int = 1,
    numeric_only: bool = False,
) -> pd.DataFrame:
    return df.cov(min_periods=min_periods, ddof=ddof, numeric_only=numeric_only)


@fn.NodeDecorator(
    node_id="pd.mean",
    name="Mean",
    description="Calculates the mean of entries.",
    outputs=[{"name": "mean", "type": pd.Series}],
)
def df_mean(
    df: pd.DataFrame,
    axis: Literal[None, 0, 1] = 0,
    numeric_only: bool = False,
) -> pd.Series:
    return df.mean(axis=axis, numeric_only=numeric_only)


@fn.NodeDecorator(
    node_id="pd.median",
    name="Median",
    description="Calculates the median of entries.",
    outputs=[{"name": "median", "type": pd.Series}],
)
def df_median(
    df: pd.DataFrame,
    axis: Literal[None, 0, 1] = 0,
    numeric_only: bool = False,
) -> pd.Series:
    return df.median(axis=axis, numeric_only=numeric_only)


@fn.NodeDecorator(
    node_id="pd.std",
    name="Standard Deviation",
    description="Calculates the standard deviation of entries.",
    outputs=[{"name": "std", "type": pd.Series}],
)
def df_std(
    df: pd.DataFrame,
    axis: Literal[None, 0, 1] = 0,
    numeric_only: bool = False,
) -> pd.Series:
    return df.std(axis=axis, numeric_only=numeric_only)


@fn.NodeDecorator(
    node_id="pd.sum",
    name="Sum",
    description="Calculates the sum of entries.",
    outputs=[{"name": "sum", "type": pd.Series}],
)
def df_sum(
    df: pd.DataFrame,
    axis: Literal[None, 0, 1] = 0,
    numeric_only: bool = False,
) -> pd.Series:
    return df.sum(axis=axis, numeric_only=numeric_only)


@fn.NodeDecorator(
    node_id="pd.var",
    name="Variance",
    description="Calculates the variance of entries.",
    outputs=[{"name": "var", "type": pd.Series}],
)
def df_var(
    df: pd.DataFrame,
    axis: Literal[None, 0, 1] = 0,
    numeric_only: bool = False,
) -> pd.Series:
    return df.var(axis=axis, numeric_only=numeric_only)


@fn.NodeDecorator(
    node_id="pd.quantile",
    name="Quantile",
    description="Calculates the quantile of entries.",
    outputs=[{"name": "quantile", "type": pd.Series}],
)
def df_quantile(
    df: pd.DataFrame,
    q: float = 0.5,
    axis: Literal[None, 0, 1] = 0,
    numeric_only: bool = False,
) -> pd.Series:
    return df.quantile(q=q, axis=axis, numeric_only=numeric_only)


@fn.NodeDecorator(
    node_id="pd.describe",
    name="Describe",
    description="Describes the DataFrame.",
    outputs=[{"name": "description", "type": pd.DataFrame}],
)
def df_describe(
    df: pd.DataFrame,
    percentiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    return df.describe(
        percentiles=percentiles,
    )


@fn.NodeDecorator(
    node_id="pd.value_counts",
    name="Value Counts",
    description="Counts the occurrences of unique values.",
    outputs=[{"name": "value_counts", "type": pd.Series}],
)
def df_value_counts(
    df: pd.DataFrame,
    subset: Optional[str] = None,
    normalize: bool = False,
    sort: bool = True,
    ascending: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    if subset is not None:
        subset = [s.strip() for s in subset.split(",")]
    multilabel_series = df.value_counts(
        subset=subset,
        normalize=normalize,
        sort=sort,
        ascending=ascending,
        dropna=dropna,
    )
    # convert to a dataframe with the indices as columns and the count column named "count"

    out_df = pd.DataFrame(multilabel_series).reset_index()
    out_df.columns = out_df.columns.astype(str)
    return out_df


@fn.NodeDecorator(
    node_id="pd.eval",
    name="Eval",
    description="Evaluates an expression in the context of the DataFrame.",
    outputs=[{"name": "result", "type": pd.Series}],
)
def df_eval(
    df: pd.DataFrame,
    expr: str,
) -> Union[pd.Series, pd.DataFrame]:
    locals = df.to_dict(orient="series")
    renamed_locals = df.rename(
        columns={col: to_valid_identifier(col) for col in df.columns}
    ).to_dict(orient="series")
    return pd.eval(
        expr,
        target=df,
        local_dict={**locals, **renamed_locals, "df": df},
        global_dict={},
    )


MATH_SHELF = fn.Shelf(
    nodes=[
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
    ],
    name="Math",
    description="Math on DataFrames",
    subshelves=[],
)
