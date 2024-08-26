from all_nodes_test_base import TestAllNodesBase
import funcnodes as fn
import funcnodes_pandas as fnpd

import pandas as pd
import numpy as np


class TestAllNodes(TestAllNodesBase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [1.1, 2.2, None],
            }
        )

        self.series = self.df.iloc[0]

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

    async def test_get_column(self):
        ins = fnpd.get_column()
        ins.inputs["df"].value = self.df
        await ins
        self.assertEqual(ins.outputs["series"].value, fn.io.NoValue)

        self.assertEqual(
            ins.get_input("column").value_options["options"],
            list(self.df.columns),
        )
        ins.inputs["column"].value = "A"
        await ins
        pd.testing.assert_series_equal(ins.outputs["series"].value, self.df["A"])

    async def test_df_loc(self):
        ins = fnpd.df_loc()
        ins.inputs["df"].value = self.df
        ins.inputs["row"].value = "0"
        await ins
        pd.testing.assert_series_equal(ins.outputs["series"].value, self.df.loc[0])
        ins.inputs["row"].value = 0
        await ins
        pd.testing.assert_series_equal(ins.outputs["series"].value, self.df.loc[0])

    async def test_df_iloc(self):
        ins = fnpd.df_iloc()
        ins.inputs["df"].value = self.df
        ins.inputs["index"].value = 0
        await ins
        pd.testing.assert_series_equal(ins.outputs["row"].value, self.df.iloc[0])

    async def test_to_csv_str(self):
        ins = fnpd.to_csv_str()
        ins.inputs["df"].value = self.df
        await ins
        self.assertEqual(ins.outputs["csv"].value, self.df.to_csv(index=False))

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
        df.columns = [f"Col {i+1}" for i in range(len(df.columns))]
        pd.testing.assert_frame_equal(ins.outputs["df"].value, df, check_dtype=False)

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

    async def test_corr(self):
        ins = fnpd.corr()
        ins.inputs["df"].value = self.df
        await ins
        pd.testing.assert_frame_equal(ins.outputs["correlation"].value, self.df.corr())

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
        ins = fnpd.concatenate()
        ins.inputs["df1"].value = self.df
        ins.inputs["df2"].value = self.df

        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["out"].value, pd.concat([self.df, self.df])
        )

    ## series

    async def test_ser_to_dict(self):
        ins = fnpd.ser_to_dict()
        ins.inputs["ser"].value = self.series
        await ins

        self.assertEqual(ins.outputs["dict"].value, self.series.to_dict())

    async def test_ser_to_list(self):
        ins = fnpd.ser_to_list()
        ins.inputs["ser"].value = self.series
        await ins
        self.assertEqual(ins.outputs["list"].value, self.series.to_list())

    async def test_ser_loc(self):
        ins = fnpd.ser_loc()
        ins.inputs["ser"].value = self.series
        ins.inputs["label"].value = "A"
        await ins
        self.assertEqual(ins.outputs["value"].value, self.series["A"])
        ins.inputs["label"].value = "B"
        await ins
        self.assertEqual(ins.outputs["value"].value, self.series["B"])

    async def test_ser_iloc(self):
        ins = fnpd.ser_iloc()
        ins.inputs["ser"].value = self.series
        ins.inputs["index"].value = 0
        await ins
        self.assertEqual(ins.outputs["value"].value, self.series[0])

    async def test_ser_from_dict(self):
        ins = fnpd.ser_from_dict()
        ins.inputs["data"].value = self.series.to_dict()
        ins.inputs["name"].value = self.series.name
        await ins
        pd.testing.assert_series_equal(ins.outputs["series"].value, self.series)

    async def test_ser_from_list(self):
        ins = fnpd.ser_from_list()
        ins.inputs["data"].value = self.series.to_list()
        ins.inputs["name"].value = self.series.name
        await ins
        pd.testing.assert_series_equal(
            ins.outputs["series"].value, self.series, check_index=False
        )

    async def test_ser_values(self):
        ins = fnpd.ser_values()
        ins.inputs["ser"].value = self.series
        await ins

        self.assertTrue(np.all(ins.outputs["values"].value == self.series.values))

    ## grouping
    async def test_groupby(self):
        ins = fnpd.group_by()
        ins.inputs["df"].value = self.df
        ins.inputs["by"].value = "A"
        await ins
        self.assertEqual(ins.outputs["grouped"].value.groups.keys(), {1, 2, 3})

    async def test_group_to_list(self):
        ins = fnpd.group_to_list()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        for i in range(3):
            pd.testing.assert_frame_equal(
                ins.outputs["list"].value[i], self.df[self.df["A"] == i + 1]
            )

    async def test_max(self):
        ins = fnpd.max()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["max"].value, self.df.groupby("A").max()
        )

    async def test_mean(self):
        ins = fnpd.mean()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["mean"].value, self.df.groupby("A").mean()
        )

    async def test_sum(self):
        ins = fnpd.sum()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["sum"].value, self.df.groupby("A").sum()
        )

    async def test_var(self):
        ins = fnpd.var()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["var"].value, self.df.groupby("A").var()
        )

    async def test_df_from_group(self):
        ins = fnpd.get_df_from_group()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        self.assertEqual(ins.inputs["name"].value_options["options"], [1, 2, 3])
        ins.inputs["name"].value = 1
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["df"].value, self.df[self.df["A"] == 1]
        )

    async def test_std(self):
        ins = fnpd.std()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["std"].value, self.df.groupby("A").std()
        )

    async def test_groupby_column(self):
        ins = fnpd.group_by_column()
        ins.inputs["df"].value = self.df
        ins.inputs["column"].value = "A"
        await ins
        self.assertEqual(ins.outputs["group"].value.groups.keys(), {1, 2, 3})

    async def test_min(self):
        ins = fnpd.min()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["min"].value, self.df.groupby("A").min()
        )

    async def test_count(self):
        ins = fnpd.count()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["count"].value, self.df.groupby("A").count()
        )

    async def test_describe(self):
        ins = fnpd.describe()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["description"].value, self.df.groupby("A").describe()
        )
