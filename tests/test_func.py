import pandas as pd
import unittest
import funcnodes_pandas as fnpd
import asyncio
import funcnodes as fn
import numpy as np


class TestSeries(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            }
        )
        self.series = self.df.iloc[0]

    