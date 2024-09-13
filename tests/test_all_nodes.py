from all_nodes_test_base import TestAllNodesBase


from test_df import (
    TestDataframeConvert,
    TestDataframeManipulation,
    TestDataframeMask,
    TestDataframeMath,
    TestDataFrameRowsCols,
)

from test_series import TestSeriesStrConvert, TestSeries
from test_grouping import TestGrouping

import funcnodes as fn

fn.config.IN_NODE_TEST = True


class TestAllNodes(TestAllNodesBase):
    sub_test_classes = [
        TestDataframeConvert,
        TestDataframeManipulation,
        TestDataframeMask,
        TestDataframeMath,
        TestDataFrameRowsCols,
        TestSeriesStrConvert,
        TestSeries,
        TestGrouping,
    ]
