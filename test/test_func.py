import pandas as pd
import unittest
import funcnodes_pandas as fnpd
import asyncio
import funcnodes as fn
import numpy as np


class TestDF(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            }
        )

    async def test_to_dict(self):
        ins = fnpd.to_dict()
        ins.inputs["df"].value = self.df
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        self.assertEqual(ins.outputs["dict"].value, self.df.to_dict("split"))

    async def test_to_orient_dict(self):
        for orient in ["dict", "list", "split", "tight", "records", "index"]:
            ins = fnpd.to_orient_dict()
            ins.inputs["df"].value = self.df
            ins.inputs["orient"].value = orient
            timeout = 2
            await asyncio.wait_for(ins, timeout)
            self.assertEqual(ins.outputs["dict"].value, self.df.to_dict(orient))

    async def test_from_dict(self):
        for orient in ["dict", "list", "split", "tight", "records", "index"]:
            ins = fnpd.from_dict()
            ins.inputs["data"].value = self.df.to_dict(orient)
            print(orient, self.df.to_dict(orient))
            timeout = 2
            await asyncio.wait_for(ins, timeout)
            pd.testing.assert_frame_equal(ins.outputs["df"].value, self.df)

    async def test_from_orient_dict(self):
        for orient in ["dict", "list", "split", "tight", "records", "index"]:
            ins = fnpd.from_orient_dict()
            ins.inputs["data"].value = self.df.to_dict(orient)
            ins.inputs["orient"].value = orient
            print(orient, self.df.to_dict(orient))
            timeout = 2
            await asyncio.wait_for(ins, timeout)
            pd.testing.assert_frame_equal(ins.outputs["df"].value, self.df)

    async def test_get_column(self):
        ins = fnpd.get_column()
        ins.inputs["df"].value = self.df
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        self.assertEqual(ins.outputs["series"].value, fn.io.NoValue)
        self.assertEqual(
            ins.get_input("column").value_options, {c: c for c in self.df.columns}
        )
        ins.inputs["column"].value = "A"
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        pd.testing.assert_series_equal(ins.outputs["series"].value, self.df["A"])

    async def test_df_loc(self):
        ins = fnpd.df_loc()
        ins.inputs["df"].value = self.df
        ins.inputs["label"].value = "0"
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        pd.testing.assert_series_equal(ins.outputs["row"].value, self.df.loc[0])
        ins.inputs["label"].value = 0
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        pd.testing.assert_series_equal(ins.outputs["row"].value, self.df.loc[0])

    async def test_df_iloc(self):
        ins = fnpd.df_iloc()
        ins.inputs["df"].value = self.df
        ins.inputs["index"].value = 0
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        pd.testing.assert_series_equal(ins.outputs["row"].value, self.df.iloc[0])

    def test_all_tested(self):
        shelfenodes = fnpd.dataframe.NODE_SHELF["nodes"]
        modnodes = []
        for name in dir(fnpd.dataframe):
            attr = getattr(fnpd.dataframe, name)
            if isinstance(attr, type) and issubclass(attr, fn.Node):
                modnodes.append(attr)

        self.assertEqual(len(shelfenodes), len(modnodes))
        for node in shelfenodes:
            if node.node_id == "pd.get_column":
                continue
            self.assertTrue(
                hasattr(self, f"test_{node.func.__name__}"),
                f"Missing test_{node.func.__name__} for {node.node_name}",
            )

    async def test_to_csv_str(self):
        ins = fnpd.to_csv_str()
        ins.inputs["df"].value = self.df
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        self.assertEqual(ins.outputs["csv"].value, self.df.to_csv(index=False))

    async def test_from_csv_str(self):
        ins = fnpd.from_csv_str()
        csv_string = self.df.to_csv(index=False)
        ins.inputs["source"].value = csv_string
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        pd.testing.assert_frame_equal(ins.outputs["df"].value, self.df)

    async def test_df_from_array(self):
        ins = fnpd.df_from_array()
        ins.inputs["data"].value = self.df.to_numpy()
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        df = self.df.copy()
        df.columns = [f"Col {i+1}" for i in range(len(df.columns))]
        pd.testing.assert_frame_equal(ins.outputs["df"].value, df)


class TestSeries(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            }
        )
        self.series = self.df.iloc[0]

    async def test_ser_to_dict(self):
        ins = fnpd.ser_to_dict()
        ins.inputs["ser"].value = self.series
        timeout = 2
        await asyncio.wait_for(ins, timeout)

        self.assertEqual(ins.outputs["dict"].value, self.series.to_dict())

    async def test_ser_to_list(self):
        ins = fnpd.ser_to_list()
        ins.inputs["ser"].value = self.series
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        self.assertEqual(ins.outputs["list"].value, self.series.to_list())

    async def test_ser_loc(self):
        ins = fnpd.ser_loc()
        ins.inputs["ser"].value = self.series
        ins.inputs["label"].value = "A"
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        self.assertEqual(ins.outputs["value"].value, self.series["A"])
        ins.inputs["label"].value = "B"
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        self.assertEqual(ins.outputs["value"].value, self.series["B"])

    async def test_ser_iloc(self):
        ins = fnpd.ser_iloc()
        ins.inputs["ser"].value = self.series
        ins.inputs["index"].value = 0
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        self.assertEqual(ins.outputs["value"].value, self.series[0])

    async def test_ser_from_dict(self):
        ins = fnpd.ser_from_dict()
        ins.inputs["data"].value = self.series.to_dict()
        ins.inputs["name"].value = self.series.name
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        pd.testing.assert_series_equal(ins.outputs["series"].value, self.series)

    async def test_ser_from_list(self):
        ins = fnpd.ser_from_list()
        ins.inputs["data"].value = self.series.to_list()
        ins.inputs["name"].value = self.series.name
        timeout = 2
        await asyncio.wait_for(ins, timeout)
        pd.testing.assert_series_equal(
            ins.outputs["series"].value, self.series, check_index=False
        )

    async def test_ser_values(self):
        ins = fnpd.ser_values()
        ins.inputs["ser"].value = self.series
        timeout = 2
        await asyncio.wait_for(ins, timeout)

        self.assertTrue(np.all(ins.outputs["values"].value == self.series.values))
