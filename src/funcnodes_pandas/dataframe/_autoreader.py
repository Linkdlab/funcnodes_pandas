from typing import List, Union, Dict, Any, Tuple
import chardet
import pandas as pd
from io import StringIO
import numpy as np


def detect_encoding(bytes):
    """Detect file encoding using chardet."""
    result = chardet.detect(bytes)
    return result["encoding"]


def get_lines(file: str):
    with open(file, "rb") as rawdata:
        data = rawdata.read()

    encoding = detect_encoding(data)

    with open(file, "r", encoding=encoding, errors="replace") as f:
        lines = f.readlines()

    return lines, encoding


def guess_table_info(
    line,
    possible_delimiters,
    possible_decimal_separators,
    possible_thousands_separators,
):
    lineinfo = {
        "length": len(line),
    }

    for delim in possible_delimiters:
        for dec_sep in possible_decimal_separators:
            if delim == dec_sep:
                continue
            for thou_sep in possible_thousands_separators:
                if delim == thou_sep:
                    continue
                if dec_sep == thou_sep:
                    continue

                # try to read the line to Series
                try:
                    series = pd.read_csv(
                        StringIO(line),
                        sep=delim,
                        decimal=dec_sep,
                        thousands=thou_sep,
                        header=None,
                        index_col=False,
                        engine="python",
                    )
                    lineinfo[(delim, dec_sep, thou_sep)] = series
                except Exception:
                    continue
    return lineinfo


def guess_best_table_params(
    tableinfos: List[dict], checklines: List[str], cutoff_ratio: float = 0.5
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tableinfo_summary = {}
    number_of_lines = len(checklines)

    # fill tableinfo_summary with None values where there is no data
    for tableinfo in tableinfos:
        for parsedata, series in tableinfo.items():
            if not isinstance(series, pd.DataFrame):
                continue
            if parsedata not in tableinfo_summary:
                tableinfo_summary[parsedata] = [None] * number_of_lines

    for i, tableinfo in enumerate(tableinfos):
        for parsedata, series in tableinfo.items():
            if not isinstance(series, pd.DataFrame):
                continue
            tableinfo_summary[parsedata][i] = series

    # first we count the number of valid data reads for set of reding parameters
    parameter_lengths = {
        k: len([x for x in v if x is not None]) for k, v in tableinfo_summary.items()
    }
    max_length = max(parameter_lengths.values())
    cutoff = int(np.ceil(max_length * cutoff_ratio))
    # we drop all that are below the cutoff
    for paml, param_len in parameter_lengths.items():
        if param_len < cutoff:
            tableinfo_summary.pop(paml)

    # next we filter by the mean data length, mean to make sure not a single long line is weighted to much,
    # but readings that result in the simple case of a single entry should also removed compared to many longer ones
    series_lengths = {
        k: float(np.mean([len(x.columns) for x in v if x is not None]))
        for k, v in tableinfo_summary.items()
    }
    max_mean_ser_length = max(series_lengths.values()) * cutoff_ratio
    for paml, param_len in series_lengths.items():
        if param_len < max_mean_ser_length:
            tableinfo_summary.pop(paml)

    series_lengths = {
        k: [len(x.columns) for x in v if x is not None]
        for k, v in tableinfo_summary.items()
    }
    highest_series_length_count = {
        pname: max(serl, key=serl.count) for pname, serl in series_lengths.items()
    }

    # next we filter by type of data, where flaot is prefered over int and int over other
    total_types = {}
    for k, v in tableinfo_summary.items():
        _max_length = highest_series_length_count[k]
        relevant_series = [
            x for x in v if x is not None and len(x.columns) == _max_length
        ]
        dtypes = []
        for s in relevant_series:
            dtypes.append(list(s.dtypes))

        dtypes = np.array(dtypes).T.tolist()
        number_ints = [sum([1 for x in d if x == np.dtype("int64")]) for d in dtypes]
        number_floats = [
            sum([1 for x in d if x == np.dtype("float64")]) for d in dtypes
        ]
        number_other = [
            sum([1 for x in d if x not in [np.dtype("int64"), np.dtype("float64")]])
            for d in dtypes
        ]

        total_types[k] = {
            "int": sum(number_ints),
            "float": sum(number_floats),
            "other": sum(number_other),
            "numerical": sum(number_ints) + sum(number_floats),
        }

    higest_numerical_value = max([v["numerical"] for v in total_types.values()])
    for k, v in list(total_types.items()):
        if v["numerical"] < higest_numerical_value:
            total_types.pop(k)

    highest_float_value = max([v["float"] for v in total_types.values()])
    count_cutff = int(np.ceil(highest_float_value * cutoff_ratio))
    for k, v in list(total_types.items()):
        if v["float"] < count_cutff:
            total_types.pop(k)

    highest_int_value = max([v["int"] for v in total_types.values()])
    count_cutff = int(np.ceil(highest_int_value * cutoff_ratio))
    for k, v in list(total_types.items()):
        if v["int"] < count_cutff:
            total_types.pop(k)

    highest_other_value = max([v["other"] for v in total_types.values()])
    count_cutff = int(np.ceil(highest_other_value * cutoff_ratio))
    for k, v in list(total_types.items()):
        if v["other"] < count_cutff:
            total_types.pop(k)

    # next we filter by the number of parameters that fullyfil the above criteria and take the longest one
    parameter_lengths = {
        k: len([x for x in v if x is not None]) for k, v in tableinfo_summary.items()
    }
    max_length = max(parameter_lengths.values())
    for paml, param_len in parameter_lengths.items():
        if param_len < max_length:
            total_types.pop(paml)

    parser_args = list(total_types.keys())[0]
    parser_args

    # now we check where the last line of header data is, that has other lengths than the data of the table
    last_drop_line = -1
    exp_series = highest_series_length_count[parser_args]

    for i, sers in enumerate(tableinfo_summary[parser_args]):
        if sers is None:
            last_drop_line = i
        else:
            if len(sers.columns) != exp_series:
                last_drop_line = i

    # lastly we estimate the header, if we expect numerical values, all initial rows without numerical values are header
    remaining_lines = checklines[last_drop_line + 1 :]
    headerlines = -1
    for line in remaining_lines:
        _df = pd.read_csv(
            StringIO(line),
            sep=parser_args[0],
            decimal=parser_args[1],
            thousands=parser_args[2],
            header=None,
            engine="python",
        )
        typeslist = _df.dtypes.to_list()

        if highest_float_value > 0:
            if np.dtype("float64") in typeslist:
                break
            else:
                headerlines += 1
                continue
        elif highest_int_value > 0:
            if np.dtype("int64") in typeslist:
                break
            else:
                headerlines += 1
                continue
        break

    if headerlines == -1:
        headerlines = None
    headerlines

    # return the estimated parameters
    return {
        "sep": parser_args[0],
        "decimal": parser_args[1],
        "thousands": parser_args[2],
        "header": headerlines,
        "skiprows": last_drop_line + 1,
    }, {
        "exp_series": exp_series,
        "last_drop_line": last_drop_line,
        "headerlines": headerlines,
    }


def auto_parse_table(
    source: Union[str, bytes],
    possible_delimiters: List[str] = None,
    possible_decimal_separators: List[str] = None,
    possible_thousands_separators: List[str] = None,
    max_lines: int = 200,
    cutoff_ratio: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    encoding = None
    if isinstance(source, bytes):
        encoding = detect_encoding(source)
        stringdata = source.decode(encoding).strip()
        lines = stringdata.split("\n")
    elif isinstance(source, str):
        source = source.strip()
        lines = source.split("\n")
        stringdata = source
    else:
        raise ValueError("source must be either a string or bytes")

    if not possible_delimiters:
        possible_delimiters = [
            ",",
            "\t",
            "\\t",
            " " * 4,
            " " * 3,
            " " * 2,
            " " * 1,
            ";",
            "|",
        ]

    if not possible_decimal_separators:
        possible_decimal_separators = [".", ",", None]
    if not possible_thousands_separators:
        possible_thousands_separators = [None, ",", ".", " "]

    if max_lines > len(lines):
        max_lines = len(lines) - 1

    checklines = lines[:max_lines]
    tableinfos = [
        guess_table_info(
            line=line,
            possible_delimiters=possible_delimiters,
            possible_decimal_separators=possible_decimal_separators,
            possible_thousands_separators=possible_thousands_separators,
        )
        for line in checklines
    ]

    parse_params, auto_params = guess_best_table_params(
        tableinfos=tableinfos,
        checklines=checklines,
        cutoff_ratio=cutoff_ratio,
    )
    if encoding:
        parse_params["encoding"] = encoding

    try:
        df = pd.read_csv(StringIO(stringdata), **parse_params)
    except Exception as e:
        raise ValueError(
            f"Could not parse table: {e} with params: {parse_params},{auto_params}"
        )
    return df, parse_params
