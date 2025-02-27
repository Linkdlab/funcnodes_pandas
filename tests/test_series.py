import unittest
import funcnodes_pandas as fnpd
import pandas as pd
import numpy as np
from funcnodes_core import testing


class TestSeriesStrConvert(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        testing.setup()
        self.df = pd.DataFrame(
            data={
                "A": ["foo", "bar", "baz"],
                "B": ["hello", "world", "foo"],
                "C": ["function", "method", "class"],
            }
        )

        self.col = self.df["A"]

    def tearDown(self):
        testing.teardown()

    async def test_ser_str_contains(self):
        ins = fnpd.ser_str_contains()
        ins.inputs["series"].value = self.col
        ins.inputs["pat"].value = "ba"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), [False, True, True])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.contains("ba")
        )

    async def test_ser_str_startswith(self):
        ins = fnpd.ser_str_startswith()
        ins.inputs["series"].value = self.col
        ins.inputs["pat"].value = "b"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), [False, True, True])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.startswith("b")
        )

    async def test_ser_str_endswith(self):
        ins = fnpd.ser_str_endswith()
        ins.inputs["series"].value = self.col
        ins.inputs["pat"].value = "o"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), [True, False, False])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.endswith("o")
        )

    async def test_ser_str_count(self):
        ins = fnpd.ser_str_count()
        ins.inputs["series"].value = self.col
        ins.inputs["pat"].value = "o"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), [2, 0, 0])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.count("o")
        )

    async def test_ser_str_len(self):
        ins = fnpd.ser_str_len()
        ins.inputs["series"].value = self.col
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), [3, 3, 3])
        pd.testing.assert_series_equal(ins.outputs["out"].value, self.col.str.len())

    async def test_ser_str_extract(self):
        ins = fnpd.ser_str_extract()
        ins.inputs["series"].value = self.col
        ins.inputs["pat"].value = "(\\w)(\\w)"
        await ins
        print(ins.outputs["out"].value)
        self.assertEqual(ins.outputs["out"].value[0].tolist(), ["f", "b", "b"])
        self.assertEqual(ins.outputs["out"].value[1].tolist(), ["o", "a", "a"])
        pd.testing.assert_frame_equal(
            ins.outputs["out"].value, self.col.str.extract(r"(\w)(\w)")
        )

    async def test_ser_str_find(self):
        ins = fnpd.ser_str_find()
        ins.inputs["series"].value = self.col
        ins.inputs["pat"].value = "o"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), [1, -1, -1])
        pd.testing.assert_series_equal(ins.outputs["out"].value, self.col.str.find("o"))

    async def test_ser_str_findall(self):
        ins = fnpd.ser_str_findall()
        ins.inputs["series"].value = self.col
        ins.inputs["pat"].value = "[fo]"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), [["f", "o", "o"], [], []])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.findall("[fo]")
        )

    async def test_ser_str_get(self):
        ins = fnpd.ser_str_get()
        ins.inputs["series"].value = self.col
        ins.inputs["index"].value = 1
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["o", "a", "a"])
        pd.testing.assert_series_equal(ins.outputs["out"].value, self.col.str.get(1))

    async def test_ser_str_replace(self):
        ins = fnpd.ser_str_replace()
        ins.inputs["series"].value = self.col
        ins.inputs["pat"].value = "o"
        ins.inputs["repl"].value = "a"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["faa", "bar", "baz"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.replace("o", "a")
        )

    async def test_ser_str_slice(self):
        ins = fnpd.ser_str_slice()
        ins.inputs["series"].value = self.col
        ins.inputs["start"].value = 1
        ins.inputs["stop"].value = 3
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["oo", "ar", "az"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.slice(1, 3)
        )

    async def test_ser_str_split(self):
        ins = fnpd.ser_str_split()
        ins.inputs["series"].value = self.col
        ins.inputs["pat"].value = "o"
        await ins
        self.assertEqual(
            ins.outputs["out"].value.tolist(), [["f", "", ""], ["bar"], ["baz"]]
        )
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.split("o")
        )

    async def test_ser_str_strip(self):
        ins = fnpd.ser_str_strip()
        ins.inputs["series"].value = self.col
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["foo", "bar", "baz"])
        pd.testing.assert_series_equal(ins.outputs["out"].value, self.col.str.strip())

    async def test_ser_str_zfill(self):
        ins = fnpd.ser_str_zfill()
        ins.inputs["series"].value = self.col
        ins.inputs["width"].value = 5
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["00foo", "00bar", "00baz"])
        pd.testing.assert_series_equal(ins.outputs["out"].value, self.col.str.zfill(5))

    async def test_ser_str_pad(self):
        ins = fnpd.ser_str_pad()
        ins.inputs["series"].value = self.col
        ins.inputs["width"].value = 5
        ins.inputs["side"].value = "right"
        ins.inputs["fillchar"].value = "!"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["foo!!", "bar!!", "baz!!"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.pad(5, side="right", fillchar="!")
        )

    async def test_ser_str_center(self):
        ins = fnpd.ser_str_center()
        ins.inputs["series"].value = self.col
        ins.inputs["width"].value = 5
        ins.inputs["fillchar"].value = "!"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["!foo!", "!bar!", "!baz!"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.center(5, fillchar="!")
        )

    async def test_ser_str_ljust(self):
        ins = fnpd.ser_str_ljust()
        ins.inputs["series"].value = self.col
        ins.inputs["width"].value = 5
        ins.inputs["fillchar"].value = "!"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["foo!!", "bar!!", "baz!!"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.ljust(5, fillchar="!")
        )

    async def test_ser_str_rjust(self):
        ins = fnpd.ser_str_rjust()
        ins.inputs["series"].value = self.col
        ins.inputs["width"].value = 5
        ins.inputs["fillchar"].value = "!"
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["!!foo", "!!bar", "!!baz"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value, self.col.str.rjust(5, fillchar="!")
        )

    async def test_ser_str_wrap(self):
        ins = fnpd.ser_str_wrap()
        ins.inputs["series"].value = self.col
        ins.inputs["width"].value = 2
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["fo\no", "ba\nr", "ba\nz"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value,
            self.col.str.wrap(width=2),
        )

    async def test_ser_str_repeat(self):
        ins = fnpd.ser_str_repeat()
        ins.inputs["series"].value = self.col
        ins.inputs["repeats"].value = 2
        await ins
        self.assertEqual(
            ins.outputs["out"].value.tolist(), ["foofoo", "barbar", "bazbaz"]
        )
        pd.testing.assert_series_equal(
            ins.outputs["out"].value,
            self.col.str.repeat(2),
        )

    async def test_ser_str_upper(self):
        ins = fnpd.ser_str_upper()
        ins.inputs["series"].value = self.col
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["FOO", "BAR", "BAZ"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value,
            self.col.str.upper(),
        )

    async def test_ser_str_lower(self):
        ins = fnpd.ser_str_lower()
        ins.inputs["series"].value = self.col
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["foo", "bar", "baz"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value,
            self.col.str.lower(),
        )

    async def test_ser_str_title(self):
        ins = fnpd.ser_str_title()
        ins.inputs["series"].value = self.col
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["Foo", "Bar", "Baz"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value,
            self.col.str.title(),
        )

    async def test_ser_str_capitalize(self):
        ins = fnpd.ser_str_capitalize()
        ins.inputs["series"].value = self.col
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["Foo", "Bar", "Baz"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value,
            self.col.str.capitalize(),
        )

    async def test_ser_str_swapcase(self):
        ins = fnpd.ser_str_swapcase()
        ins.inputs["series"].value = self.col
        await ins
        self.assertEqual(ins.outputs["out"].value.tolist(), ["FOO", "BAR", "BAZ"])
        pd.testing.assert_series_equal(
            ins.outputs["out"].value,
            self.col.str.swapcase(),
        )

    async def test_ser_str_cat(self):
        ins = fnpd.ser_str_cat()
        ins.inputs["series"].value = self.col
        ins.inputs["sep"].value = "-"
        await ins
        self.assertEqual(ins.outputs["out"].value, "foo-bar-baz")

    async def test_ser_str_cat_others(self):
        ins = fnpd.ser_str_cat()
        ins.inputs["series"].value = self.col
        ins.inputs["sep"].value = "-"
        ins.inputs["others"].value = ["hello", "world", "foo"]
        await ins
        self.assertEqual(
            ins.outputs["out"].value.tolist(), ["foo-hello", "bar-world", "baz-foo"]
        )

        pd.testing.assert_series_equal(
            ins.outputs["out"].value,
            self.col.str.cat(["hello", "world", "foo"], sep="-"),
        )

    async def test_ser_str_cat_no_sep(self):
        ins = fnpd.ser_str_cat()
        ins.inputs["series"].value = self.col
        await ins
        self.assertEqual(ins.outputs["out"].value, "foobarbaz")

    async def test_ser_str_cat_no_sep_others(self):
        ins = fnpd.ser_str_cat()
        ins.inputs["series"].value = self.col
        ins.inputs["others"].value = ["hello", "world", "foo"]
        await ins
        self.assertEqual(
            ins.outputs["out"].value.tolist(), ["foohello", "barworld", "bazfoo"]
        )
        pd.testing.assert_series_equal(
            ins.outputs["out"].value,
            self.col.str.cat(["hello", "world", "foo"]),
        )


class TestSeries(unittest.IsolatedAsyncioTestCase):
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

    async def test_ser_to_dict(self):
        ins = fnpd.ser_to_dict()
        ins.inputs["ser"].value = self.series
        await ins
        self.assertEqual(ins.outputs["dict"].value, self.series.to_dict())
