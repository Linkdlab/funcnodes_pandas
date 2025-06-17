import unittest
import funcnodes_pandas as fnpd
import pandas as pd
import funcnodes as fn
import numpy as np
from funcnodes_core import testing


class TestDataframeConvert(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        testing.setup()
        self.df = pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [1.1, 2.2, None],
            }
        )

        self.series = self.df.iloc[0]

    def tearDown(self):
        testing.teardown()

    async def test_to_dict(self):
        ins = fnpd.to_dict()
        ins.inputs["df"].value = self.df.fillna(0)
        await ins
        self.assertEqual(ins.outputs["dict"].value, self.df.fillna(0).to_dict("split"))

    async def test_to_orient_dict(self):
        for orient in ["dict", "list", "split", "tight", "records", "index"]:
            ins = fnpd.to_orient_dict()
            ins.inputs["df"].value = self.df.fillna(0)
            ins.inputs["orient"].value = orient
            await ins
            self.assertEqual(
                ins.outputs["dict"].value,
                self.df.fillna(0).to_dict(orient),
                {
                    "orient": orient,
                    "data": self.df.fillna(0).to_dict(orient),
                    "df": ins.outputs["dict"].value,
                },
            )

    async def test_from_dict(self):
        for orient in ["dict", "list", "split", "tight", "records", "index"]:
            ins = fnpd.from_dict()
            ins.inputs["data"].value = self.df.to_dict(orient)
            print(orient, self.df.to_dict(orient))
            await ins
            pd.testing.assert_frame_equal(
                ins.outputs["df"].value, self.df, check_dtype=False
            )

    async def test_from_orient_dict(self):
        for orient in ["dict", "list", "split", "tight", "records", "index"]:
            ins = fnpd.from_orient_dict()
            ins.inputs["data"].value = self.df.to_dict(orient)
            ins.inputs["orient"].value = orient
            print(orient, self.df.to_dict(orient))
            await ins
            pd.testing.assert_frame_equal(
                ins.outputs["df"].value, self.df, check_dtype=False
            )

    async def test_from_csv_str(self):
        ins = fnpd.from_csv_str()
        csv_string = self.df.to_csv(index=False)
        ins.inputs["source"].value = csv_string
        await ins
        pd.testing.assert_frame_equal(ins.outputs["df"].value, self.df)

    async def test_df_from_excel(self):
        ins = fnpd.DfFromExcelNode()
        toxls = fnpd.df_to_xls()
        toxls.inputs["df"].value = self.df
        toxls.outputs["xls"].connect(ins.inputs["data"])
        await toxls
        await ins
        print(ins.outputs["df"].value)
        print(self.df)
        pd.testing.assert_frame_equal(ins.outputs["df"].value, self.df)

    async def test_df_from_array(self):
        ins = fnpd.df_from_array()
        ins.inputs["data"].value = self.df.to_numpy()
        await ins
        df = self.df.copy()
        df.columns = [f"Col {i + 1}" for i in range(len(df.columns))]
        pd.testing.assert_frame_equal(ins.outputs["df"].value, df, check_dtype=False)

    async def test_to_csv_str(self):
        ins = fnpd.to_csv_str()
        ins.inputs["df"].value = self.df
        await ins
        self.assertEqual(ins.outputs["csv"].value, self.df.to_csv(index=False))

    async def test_df_from_csv_auto(self):
        ins = fnpd.from_csv_auto()
        larger_df = pd.concat([self.df] * 10).reset_index(drop=True)

        # gernate random header data
        headerdata = {
            "name": "test",
            "age": 20,
            "raddata": list(range(10)),
        }

        headerstring = ""
        for key, value in headerdata.items():
            if isinstance(value, list):
                value = ",".join(map(str, value))
            headerstring += f"{key}: {value}\n"

        possible_delimiters = [
            ",",
            "\t",
            " " * 1,
            ";",
            "|",
        ]

        possible_decimal_separators = [".", ","]

        for delimiters in possible_delimiters:
            for dec in possible_decimal_separators:
                if dec == delimiters:
                    continue
                csv_string = larger_df.to_csv(index=False, sep=delimiters, decimal=dec)
                self.assertIsInstance(csv_string, str)
                csv_string = headerstring + csv_string
                ins.inputs["source"].value = csv_string
                await ins
                params = ins.outputs["params"].value
                # print(params)
                # print(ins.outputs["df"].value)
                # print(csv_string)
                pd.testing.assert_frame_equal(
                    ins.outputs["df"].value,
                    larger_df,
                )
                self.assertEqual(params["sep"], delimiters, params)
                self.assertEqual(params["decimal"], dec, params)
                self.assertEqual(params["skiprows"], len(headerdata), params)


class TestDataframeManipulation(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        testing.setup()
        self.df = pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [1.1, 2.2, None],
            }
        )
        self.df.index.name = "my_index"

        self.series = self.df.iloc[0]

    def tearDown(self):
        testing.teardown()

    async def test_dropna(self):
        ins = fnpd.dropna()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_frame_equal(ins.outputs["out"].value, self.df.dropna())

    async def test_fillna(self):
        ins = fnpd.fillna()
        ins.inputs["df"].value = self.df
        ins.inputs["value"].value = 0
        await ins
        pd.testing.assert_frame_equal(ins.outputs["out"].value, self.df.fillna(0))

    async def test_ffill(self):
        ins = fnpd.ffill()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_frame_equal(ins.outputs["out"].value, self.df.ffill())

    async def test_bfill(self):
        ins = fnpd.bfill()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_frame_equal(ins.outputs["out"].value, self.df.bfill())

    async def test_drop_duplicates(self):
        ins = fnpd.drop_duplicates()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["out"].value, self.df.drop_duplicates()
        )

    async def test_numeric_only(self):
        df = self.df.copy()
        df["D"] = ["a", "b", "a"]
        ins = fnpd.numeric_only()
        ins.inputs["df"].value = df
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["out"].value, self.df.select_dtypes(include=[np.number])
        )
        self.assertEqual(ins.outputs["out"].value.columns.tolist(), ["A", "B", "C"])

        ins.get_input("label_encode").value = True
        await ins
        self.assertEqual(
            ins.outputs["out"].value.columns.tolist(), ["A", "B", "C", "D"]
        )
        self.assertEqual(ins.outputs["out"].value["D"].tolist(), [0, 1, 0])

    async def test_drop_columns(self):
        ins = fnpd.drop_columns()
        ins.inputs["df"].value = self.df
        ins.inputs["columns"].value = "A"
        await ins
        self.assertEqual(ins.outputs["out"].value.columns.tolist(), ["B", "C"])

    async def test_drop_rows(self):
        ins = fnpd.drop_rows()
        ins.inputs["df"].value = self.df
        ins.inputs["rows"].value = "0"
        await ins
        pd.testing.assert_frame_equal(ins.outputs["out"].value, self.df.iloc[1:])

    async def test_add_column(self):
        ins = fnpd.add_column()
        ins.inputs["df"].value = self.df
        ins.inputs["column"].value = "D"
        ins.inputs["data"].value = 1
        await ins
        self.assertEqual(
            ins.outputs["out"].value.columns.tolist(), ["A", "B", "C", "D"]
        )
        self.assertEqual(ins.outputs["out"].value["D"].tolist(), [1, 1, 1])

        ins = fnpd.add_column()
        ins.inputs["df"].value = self.df
        ins.inputs["column"].value = "D"
        ins.inputs["data"].value = [1, 2, 3]
        await ins
        self.assertEqual(
            ins.outputs["out"].value.columns.tolist(), ["A", "B", "C", "D"]
        )
        self.assertEqual(ins.outputs["out"].value["D"].tolist(), [1, 2, 3])

        ins = fnpd.add_column()
        ins.inputs["df"].value = self.df
        ins.inputs["column"].value = "D"
        ins.inputs["data"].value = [1, 2]
        with self.assertRaises(fn.NodeTriggerError):
            await ins
        self.assertEqual(ins.outputs["out"].value, fn.NoValue)

    async def test_add_row(self):
        ins = fnpd.add_row()
        ins.inputs["df"].value = self.df
        ins.inputs["row"].value = [1, 2, 3]
        await ins

        pd.testing.assert_frame_equal(
            ins.outputs["out"].value.iloc[-1:],
            pd.DataFrame([[1, 2, 3.0]], columns=["A", "B", "C"]),
        )

    async def test_drop_column(self):
        ins = fnpd.drop_column()
        ins.inputs["df"].value = self.df
        ins.inputs["column"].value = "A"
        await ins
        self.assertEqual(ins.outputs["out"].value.columns.tolist(), ["B", "C"])

    async def test_drop_row(self):
        ins = fnpd.drop_row()
        ins.inputs["df"].value = self.df
        ins.inputs["row"].value = 0
        await ins
        pd.testing.assert_frame_equal(ins.outputs["out"].value, self.df.iloc[1:])

    async def test_concat(self):
        ins = fnpd.df_concatenate()
        ins.inputs["df1"].value = self.df
        ins.inputs["df2"].value = self.df

        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["out"].value, pd.concat([self.df, self.df])
        )

    async def test_merge(self):
        ins = fnpd.df_merge()
        ins.inputs["df_left"].value = self.df
        ins.inputs["df_right"].value = self.df
        ins.inputs["left_on"].value = "A"
        ins.inputs["right_on"].value = "A"

        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["df"].value,
            pd.merge(self.df, self.df, left_on="A", right_on="A"),
        )

    async def test_join(self):
        ins = fnpd.df_join()
        ins.inputs["df_left"].value = self.df
        ins.inputs["df_right"].value = self.df
        ins.inputs["on"].value = "A"
        ins.inputs["rsuffix"].value = "_r"

        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["df"].value, self.df.join(self.df, on="A", rsuffix="_r")
        )

    async def test_df_reset_index(self):
        ins = fnpd.df_reset_index()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["out"].value, self.df.reset_index(drop=False)
        )
        self.assertEqual(
            ins.outputs["out"].value.columns.tolist(),
            ["my_index", "A", "B", "C"],
        )


class TestDataframeMask(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        testing.setup()
        self.df = pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [1.1, 2.2, None],
            }
        )

        self.series = self.df.iloc[0]

    def tearDown(self):
        testing.teardown()

    async def test_filter(self):
        ins = fnpd.filter()
        ins.inputs["df"].value = self.df
        ins.inputs["condition"].value = "A > 1"
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["filtered"].value, self.df[self.df["A"] > 1]
        )

    async def test_mask(self):
        ins = fnpd.mask()
        ins.inputs["df"].value = self.df
        ins.inputs["mask"].value = [True, False, True]
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["masked"].value, self.df[[True, False, True]]
        )


class TestDataframeMath(unittest.IsolatedAsyncioTestCase):
    """
     "df_corr",
    "df_cov",
    "df_mean",
    "df_median",
    "df_std",
    "df_sum",
    "df_var",
    "df_quantile",
    "df_describe",
    "df_value_counts",
    "df_eval",
    """

    def setUp(self) -> None:
        testing.setup()
        self.df = pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [1.1, 2.2, None],
            }
        )

    def tearDown(self):
        testing.teardown()

    async def test_corr(self):
        ins = fnpd.df_corr()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_frame_equal(ins.outputs["correlation"].value, self.df.corr())

    async def test_cov(self):
        ins = fnpd.df_cov()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_frame_equal(ins.outputs["covariance"].value, self.df.cov())

    async def test_mean(self):
        ins = fnpd.df_mean()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_series_equal(ins.outputs["mean"].value, self.df.mean())

    async def test_median(self):
        ins = fnpd.df_median()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_series_equal(ins.outputs["median"].value, self.df.median())

    async def test_std(self):
        ins = fnpd.df_std()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_series_equal(ins.outputs["std"].value, self.df.std())

    async def test_sum(self):
        ins = fnpd.df_sum()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_series_equal(ins.outputs["sum"].value, self.df.sum())

    async def test_var(self):
        ins = fnpd.df_var()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_series_equal(ins.outputs["var"].value, self.df.var())

    async def test_quantile(self):
        ins = fnpd.df_quantile()
        ins.inputs["df"].value = self.df
        ins.inputs["q"].value = 0.5
        await ins
        pd.testing.assert_series_equal(
            ins.outputs["quantile"].value, self.df.quantile(0.5)
        )

    async def test_describe(self):
        ins = fnpd.df_describe()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["description"].value, self.df.describe()
        )

    async def test_value_counts(self):
        ins = fnpd.df_value_counts()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["value_counts"].value, self.df.value_counts().reset_index()
        )

    async def test_eval(self):
        ins = fnpd.df_eval()
        ins.inputs["df"].value = self.df
        ins.inputs["expr"].value = "D = A + B"
        await ins
        exp = self.df.eval("D = A + B")
        pd.testing.assert_frame_equal(ins.outputs["result"].value, exp)

        ins.inputs["expr"].value = "A + B"
        await ins
        exp = self.df.eval("A + B")
        pd.testing.assert_series_equal(ins.outputs["result"].value, exp)

    async def test_eval_with_non_conform_cols(self):
        ins = fnpd.df_eval()
        df = self.df.rename(
            columns={"A": "2A", "B": "space col"},
        )
        ins.inputs["df"].value = df
        ins.inputs["expr"].value = "D = _2A + space_col"
        await ins

        print(ins.outputs["result"].value)

        exp = self.df.eval("D = A + B")
        exp = exp.rename(
            columns={"A": "2A", "B": "space col"},
        )
        pd.testing.assert_frame_equal(ins.outputs["result"].value, exp)


class TestDataFrameRowsCols(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        testing.setup()
        self.df = pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [1.1, 2.2, None],
            }
        )
        self.df.index.name = "my_index"

        self.series = self.df.iloc[0]

    def tearDown(self):
        testing.teardown()

    async def test_get_column(self):
        ins = fnpd.get_column()
        ins.inputs["df"].value = self.df
        await ins
        self.assertEqual(ins.outputs["series"].value, fn.NoValue)

        self.assertEqual(
            ins.get_input("column").value_options["options"],
            list(self.df.columns),
        )
        ins.inputs["column"].value = "A"
        await ins
        pd.testing.assert_series_equal(ins.outputs["series"].value, self.df["A"])

    async def test_set_column(self):
        ins = fnpd.set_column()
        ins.inputs["df"].value = self.df
        ins.inputs["column"].value = "D"
        ins.inputs["data"].value = 1
        await ins
        self.assertEqual(
            ins.outputs["out"].value.columns.tolist(), ["A", "B", "C", "D"]
        )
        self.assertEqual(ins.outputs["out"].value["D"].tolist(), [1, 1, 1])

    async def test_df_loc(self):
        ins = fnpd.df_loc()
        ins.inputs["df"].value = self.df
        ins.inputs["row"].value = "0"
        await ins
        pd.testing.assert_series_equal(ins.outputs["series"].value, self.df.loc[0])
        ins.inputs["row"].value = 0
        await ins
        pd.testing.assert_series_equal(ins.outputs["series"].value, self.df.loc[0])

    async def test_df_set_row(self):
        ins = fnpd.set_row()
        ins.inputs["df"].value = self.df
        ins.inputs["row"].value = 0
        ins.inputs["data"].value = 2
        await ins

        self.assertEqual(ins.outputs["out"].value.iloc[0].tolist(), [2, 2, 2])

    async def test_df_iloc(self):
        ins = fnpd.df_iloc()
        ins.inputs["df"].value = self.df
        ins.inputs["index"].value = 0
        await ins
        pd.testing.assert_series_equal(ins.outputs["row"].value, self.df.iloc[0])

        # check value options
        self.assertEqual(ins.get_input("index").value_options["min"], 0)
        self.assertEqual(ins.get_input("index").value_options["max"], len(self.df) - 1)
        self.assertEqual(ins.get_input("index").value_options["step"], 1)

    async def test_df_ilocs(self):
        ins = fnpd.df_ilocs()
        ins.inputs["df"].value = self.df
        ins.inputs["indices"].value = [0, 1.0]
        await ins
        pd.testing.assert_frame_equal(ins.outputs["rows"].value, self.df.iloc[[0, 1]])

    async def test_get_rows(self):
        ins = fnpd.get_rows()
        ins.inputs["df"].value = self.df
        ins.inputs["rows"].value = [0, 1]
        await ins
        pd.testing.assert_frame_equal(ins.outputs["out"].value, self.df.loc[[0, 1]])

    async def test_df_rename_col(self):
        ins = fnpd.df_rename_col()
        ins.inputs["df"].value = self.df
        ins.inputs["old_name"].value = "A"
        ins.inputs["new_name"].value = "new_A"
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["out"].value, self.df.rename(columns={"A": "new_A"})
        )

    async def test_df_rename_col_with_non_conform_cols(self):
        ins = fnpd.df_rename_cols_valid_identifier()
        df = self.df.rename(
            columns={"A": "2A", "B": "space col"},
        )
        ins.inputs["df"].value = df

        await ins
        self.assertEqual(
            ins.outputs["out"].value.columns.tolist(), ["_2A", "space_col", "C"]
        )

    async def test_df_get_index(self):
        ins = fnpd.df_get_index()
        ins.inputs["df"].value = self.df
        await ins
        self.assertEqual(ins.outputs["index"].value.tolist(), self.df.index.tolist())
        self.assertEqual(ins.outputs["index"].value.name, self.df.index.name)

    async def test_get_column_names(self):
        ins = fnpd.get_column_names()
        ins.inputs["df"].value = self.df
        await ins
        self.assertEqual(ins.outputs["columns"].value, self.df.columns.tolist())

    async def test_get_columns_by_names(self):
        ins = fnpd.get_columns_by_names()
        ins.inputs["df"].value = self.df
        ins.inputs["columns"].value = "A, B"
        await ins
        pd.testing.assert_frame_equal(ins.outputs["subdf"].value, self.df[["A", "B"]])

    async def test_get_columns_by_index(self):
        ins = fnpd.get_columns_by_index()
        ins.inputs["df"].value = self.df
        ins.inputs["index"].value = 1
        await ins
        pd.testing.assert_series_equal(ins.outputs["series"].value, self.df.iloc[:, 1])

        #    get_column_names,

    # get_columns_by_names,
    # get_columns_by_index,


class TestReduceDataFrameNode(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        testing.setup()
        # Set up common DataFrame for testing
        self.df = pd.DataFrame(
            {
                "A": [1, 1.02, 1.03, 1.10, 1.11, 120],
                "B": [5, 5, 5, 10, 10, 20],
                "C": [10, 20, 30, 40, 50, 60],
            }
        )

    def tearDown(self):
        testing.teardown()

    async def test_empty_dataframe(self):
        df_empty = pd.DataFrame(columns=["A", "B"])
        ins = fnpd.reduce_df()
        ins.inputs["df"].value = df_empty
        ins.inputs["on"].value = "A"

        with self.assertRaises(fn.NodeTriggerError):
            await ins

    async def test_single_row(self):
        df_single_row = pd.DataFrame({"A": [1], "B": [5]})
        ins = fnpd.reduce_df()
        ins.inputs["df"].value = df_single_row
        ins.inputs["on"].value = "A"

        await ins
        pd.testing.assert_frame_equal(ins.outputs["reduced df"].value, df_single_row)

    async def test_column_not_found(self):
        ins = fnpd.reduce_df()
        ins.inputs["df"].value = self.df
        ins.inputs["on"].value = "D"  # Non-existing column

        with self.assertRaises(fn.NodeTriggerError):
            await ins

    async def test_non_numeric_column(self):
        df_non_numeric = self.df.copy()
        df_non_numeric["D"] = ["a", "b", "c", "d", "e", "f"]

        ins = fnpd.reduce_df()
        ins.inputs["df"].value = df_non_numeric
        ins.inputs["on"].value = "D"  # Non-numeric column

        with self.assertRaises(fn.NodeTriggerError):
            await ins

    async def test_threshold_not_provided(self):
        ins = fnpd.reduce_df()
        ins.inputs["df"].value = self.df
        ins.inputs["on"].value = "A"

        await ins
        print(ins.outputs["reduced df"].value)
        expected = pd.DataFrame(
            {"A": [1, 1.11, 120], "B": [5, 10, 20], "C": [10, 50, 60]}
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(ins.outputs["reduced df"].value, expected)

    async def test_custom_threshold(self):
        ins = fnpd.reduce_df()
        ins.inputs["df"].value = self.df
        ins.inputs["on"].value = "A"
        ins.inputs["threshold"].value = 0.03

        await ins
        print(ins.outputs["reduced df"].value)
        expected = pd.DataFrame(
            {
                "A": [1, 1.03, 1.10, 1.11, 120],
                "B": [5, 5, 10, 10, 20],
                "C": [10, 30, 40, 50, 60],
            }
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(ins.outputs["reduced df"].value, expected)

        ins.inputs["threshold"].value = 0.1

        await ins
        print(ins.outputs["reduced df"].value)
        expected = pd.DataFrame(
            {
                "A": [1, 1.10, 1.11, 120],
                "B": [5, 10, 10, 20],
                "C": [10, 40, 50, 60],
            }
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(ins.outputs["reduced df"].value, expected)

    async def test_multiple_columns(self):
        ins = fnpd.reduce_df()
        ins.inputs["df"].value = self.df
        ins.inputs["on"].value = ["A", "B"]

        await ins

        expected = pd.DataFrame(
            {
                "A": [1, 1.03, 1.10, 1.11, 120],
                "B": [5, 5, 10, 10, 20],
                "C": [10, 30, 40, 50, 60],
            }
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(ins.outputs["reduced df"].value, expected)

    async def test_custom_threshold_multiple_columns(self):
        ins = fnpd.reduce_df()
        ins.inputs["df"].value = self.df
        ins.inputs["on"].value = ["A", "B"]
        ins.inputs["threshold"].value = [0.1, 5]

        await ins
        print(ins.outputs["reduced df"].value)
        expected = pd.DataFrame(
            {"A": [1, 1.1, 1.11, 120], "B": [5, 10, 10, 20], "C": [10, 40, 50, 60]}
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(ins.outputs["reduced df"].value, expected)

    async def test_percentage_threshold(self):
        ins = fnpd.reduce_df()
        ins.inputs["df"].value = self.df
        ins.inputs["on"].value = "B"
        ins.inputs["percentage_threshold"].value = 0.5

        await ins
        print(ins.outputs["reduced df"].value)
        expected = pd.DataFrame(
            {
                "A": [1, 1.11, 120],
                "B": [5, 10, 20],
                "C": [10, 50, 60],
            }
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(ins.outputs["reduced df"].value, expected)

    async def test_strong_reduce_df(self):
        ins = fnpd.reduce_df()
        x = np.arange(1000)
        y = np.random.rand(1000)
        y[200:400] += 200
        df = pd.DataFrame(
            {
                "x": x,
                "y": y,
            }
        )
        ins.inputs["df"].value = df
        ins.inputs["on"].value = "y"

        await ins

        print(ins.outputs["reduced df"].value)
        expected = pd.DataFrame(
            {
                "x": [0, 199, 200, 399, 400, 999],
                "y": df.iloc[[0, 199, 200, 399, 400, 999]]["y"],
            }
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(ins.outputs["reduced df"].value, expected)


class TestDataframeOther(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        testing.setup()
        self.df = pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [1.1, 2.2, None],
            }
        )

        self.series = self.df.iloc[0]

    def tearDown(self):
        testing.teardown()

    async def test_display_df(self):
        ins = fnpd.display_df()
        ins.inputs["df"].value = self.df
        await ins

        ins.inputs["df"].value = self.series
        with self.assertRaises(fn.NodeTriggerError):
            await ins
