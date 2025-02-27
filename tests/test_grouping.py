import unittest
import funcnodes_pandas as fnpd
import pandas as pd
from funcnodes_core import testing


class TestGrouping(unittest.IsolatedAsyncioTestCase):
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
        ins = fnpd.gr_max()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["max"].value, self.df.groupby("A").max()
        )

    async def test_mean(self):
        ins = fnpd.gr_mean()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["mean"].value, self.df.groupby("A").mean()
        )

    async def test_sum(self):
        ins = fnpd.gr_sum()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["sum"].value, self.df.groupby("A").sum()
        )

    async def test_var(self):
        ins = fnpd.gr_var()
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
        ins = fnpd.gr_std()
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
        ins = fnpd.gr_min()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["min"].value, self.df.groupby("A").min()
        )

    async def test_count(self):
        ins = fnpd.gr_count()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["count"].value, self.df.groupby("A").count()
        )

    async def test_describe(self):
        ins = fnpd.gr_describe()
        ins.inputs["group"].value = self.df.groupby("A")
        await ins
        pd.testing.assert_frame_equal(
            ins.outputs["description"].value, self.df.groupby("A").describe()
        )
